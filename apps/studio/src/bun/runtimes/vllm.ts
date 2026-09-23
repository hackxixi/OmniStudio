import type { Subprocess } from "bun";
import { existsSync } from "fs";
import { getSetting, getServerPort, ENGINE_EXTRA_ARGS_KEYS } from "../db/settings";
import { resolveManagedPython } from "../python-engine";
import { modelNameForPath } from "../model-scan";
import { slugModelFileName } from "../model-store";
import { markServerStarted } from "../stats";
import { extractStartupError } from "./errors";
import { MAX_LOG_CHARS, killProcessTree, probeCommand, pumpServerOutput, readHelpText, spawnServerProcess } from "./proc";
import { getModelParams } from "../db/model-params";
import {
  cachedVllmHelpSupport,
  parseVllmHelp,
  resolveSamplingOrNull,
  setVllmHelpSupport,
  vllmSamplingArgs,
} from "./engine-sampling";
import { shellJoin, splitShellArgs } from "./shell-args";
import type {
  BinaryCheckResult,
  LogListener,
  Runtime,
  RuntimeOverrides,
  ServerStatus,
  StartResult,
  StatusListener,
} from "./types";

const DOWNLOAD_PATTERN = /downloading|fetching|(\d+(\.\d+)?)\s*%|progress/i;



export class VllmRuntime implements Runtime {
  readonly id = "vllm";
  readonly label = "vLLM";

  constructor(private readonly overrides: RuntimeOverrides = {}) {}

  private serverProcess: Subprocess | null = null;
  private serverStatus: ServerStatus = "stopped";
  private serverLogs = "";
  private lastError = "";
  private lastDownloadActivityAt = 0;

  private logListeners = new Set<LogListener>();
  private statusListeners = new Set<StatusListener>();

  private setStatus(status: ServerStatus) {
    this.serverStatus = status;
    for (const cb of this.statusListeners) cb(status);
  }

  private appendLog(text: string) {
    this.serverLogs += text;
    if (this.serverLogs.length > MAX_LOG_CHARS) {
      this.serverLogs = this.serverLogs.slice(-MAX_LOG_CHARS);
    }
    if (DOWNLOAD_PATTERN.test(text)) {
      this.lastDownloadActivityAt = Date.now();
      if (this.serverStatus === "starting") this.setStatus("downloading");
    }
    for (const cb of this.logListeners) cb(text);
  }

  onLog(cb: LogListener): () => void {
    this.logListeners.add(cb);
    return () => this.logListeners.delete(cb);
  }

  onStatusChange(cb: StatusListener): () => void {
    this.statusListeners.add(cb);
    return () => this.statusListeners.delete(cb);
  }

  getStatus(): ServerStatus {
    return this.serverStatus;
  }

  getPid(): number | undefined {
    return this.serverProcess?.pid;
  }

  getLogs(): string {
    return this.serverLogs;
  }

  getLastError(): string {
    return this.lastError;
  }

  clearLogs() {
    this.serverLogs = "";
  }

  async checkBinary(): Promise<BinaryCheckResult> {
    // 应用自己装的托管 venv 优先（引导页 / 设置里的「一键安装」）。
    const managed = resolveManagedPython("vllm", "vllm");
    if (managed) return { found: true, path: managed, mode: "python" };

    // Check for vllm CLI
    const vllmPath = Bun.which("vllm");
    if (vllmPath) return { found: true, path: vllmPath };

    // Check for python -m vllm
    const pythonPath = Bun.which("python3") ?? Bun.which("python");
    if (pythonPath) {
      // 退出码为 0 才算数（`python3 -m vllm --help` 在没有 vllm 时退出码 1）：
      // 只判"进程退出了"会让任何一台装了 python3 的机器都报"已安装"。
      if (await probeCommand([pythonPath, "-m", "vllm", "--help"], 5000)) {
        return { found: true, path: pythonPath, mode: "python" };
      }
    }

    return { found: false };
  }

  private resolveModel(): { model: string; servedName?: string } {
    // 显式覆盖（已启动模型注册表）优先：同引擎多实例时不能读「当前活动模型」。
    if (this.overrides.model) {
      const target = this.overrides.model;
      const fallbackName = existsSync(target)
        ? slugModelFileName(modelNameForPath(target))
        : undefined;
      return { model: target, servedName: this.overrides.servedName ?? fallbackName };
    }

    const localPath = getSetting("LOCAL_MODEL_PATH");
    if (localPath) {
      const localName = getSetting("LOCAL_MODEL_NAME");
      const servedName = localName
        ? localName.toLowerCase().replace(/[^a-z0-9_.-]/g, "-")
        : undefined;
      return { model: localPath, servedName };
    }

    const chatModel = getSetting("CHAT_MODEL");
    if (chatModel) return { model: chatModel };

    const profileId = getSetting("VLLM_MODEL_PROFILE");
    if (profileId && profileId !== "none") {
      // For vLLM, we use the HF model ID directly (not GGUF)
      const customHf = getSetting("CUSTOM_HF_MODEL");
      if (customHf) return { model: customHf.split(":")[0] ?? customHf };
    }

    const customHf = getSetting("CUSTOM_HF_MODEL");
    if (customHf) return { model: customHf.split(":")[0] ?? customHf };

    return { model: "" };
  }

  buildCommandLine(modelOverride?: string): string {
    let model: string;
    let servedName: string | undefined;
    if (modelOverride) {
      model = modelOverride;
      if (existsSync(modelOverride)) {
        servedName = slugModelFileName(modelNameForPath(modelOverride));
      }
    } else {
      const resolved = this.resolveModel();
      model = resolved.model;
      servedName = resolved.servedName;
    }

    const args = this.buildArgs(model, servedName);
    const vllmPath = Bun.which("vllm");
    if (vllmPath) return shellJoin([vllmPath, ...args]);
    const python = Bun.which("python3") ?? Bun.which("python");
    if (python) return shellJoin([python, "-m", "vllm.entrypoints.openai.api_server", ...args.slice(1)]);
    return shellJoin(["vllm", ...args]);
  }

  /**
   * 采样开关探测缓存的 key：启动时实际用的那份可执行文件（没启动过 = null → 一个采样参数都不发，
   * 所以没启动过时复制的命令里没有它们；启动过一次后预览与实际一致）。
   */
  private helpKey: string | null = null;
  /** 本次运行实际发出去的 argv（没启动过 = null），needsRestart 的比对基准。 */
  private launchedArgs: string[] | null = null;

  needsRestart(): boolean {
    if (this.launchedArgs === null || this.serverStatus !== "running") return false;
    const { model, servedName } = this.resolveModel();
    return shellJoin(this.buildArgs(model, servedName)) !== shellJoin(this.launchedArgs);
  }

  /**
   * 探测 --override-generation-config / --default-chat-template-kwargs。vLLM 新版 `serve --help`
   * 只列参数组，要 `--help=all` 才有全量；老版不认 `=all`，再退回 `--help`。python 形态直接问
   * api_server（它的 parser 一次列全）。vLLM 导入很重，给足超时；成功才落缓存。
   */
  private async probeHelp(binaryPath: string, isPython: boolean): Promise<void> {
    const key = `${isPython ? "py" : "cli"}:${binaryPath}`;
    this.helpKey = key;
    if (cachedVllmHelpSupport(key) !== null) return;
    const cmds = isPython
      ? [[binaryPath, "-m", "vllm.entrypoints.openai.api_server", "--help"]]
      : [
          [binaryPath, "serve", "--help=all"],
          [binaryPath, "serve", "--help"],
        ];
    for (const cmd of cmds) {
      const help = await readHelpText(cmd, 60_000);
      if (help === null) continue;
      const support = parseVllmHelp(help);
      // 第一条没认出来可能只是「不认 =all、打了用法错误」，继续试下一条；最后一条的结论照单全收。
      if (support.overrideGenerationConfig || cmd === cmds[cmds.length - 1]) {
        setVllmHelpSupport(key, support);
        return;
      }
    }
  }

  private buildArgs(model: string, servedName?: string): string[] {
    const port = this.overrides.port ?? getServerPort(this.id);
    const host = getSetting("SERVER_HOST") || "127.0.0.1";
    // 按模型参数（key = 模型 target）：ctxSize → --max-model-len，追加参数，采样 / 思考默认值。
    const mp = getModelParams(model);
    const maxModelLen =
      mp?.ctxSize !== undefined ? String(mp.ctxSize) : getSetting("VLLM_MAX_MODEL_LEN") || "8192";
    const tensorParallel = getSetting("VLLM_TENSOR_PARALLEL_SIZE") || "1";
    const gpuMemUtil = getSetting("VLLM_GPU_MEMORY_UTILIZATION") || "0.9";
    const enforceEager = getSetting("VLLM_ENFORCE_EAGER") === "1";
    const dtype = getSetting("VLLM_DTYPE") || "auto";

    const args: string[] = [
      "serve",
      model,
      "--host",
      host,
      "--port",
      port,
      "--max-model-len",
      maxModelLen,
      "--tensor-parallel-size",
      tensorParallel,
      "--gpu-memory-utilization",
      gpuMemUtil,
      "--dtype",
      dtype,
    ];

    if (servedName) args.push("--served-model-name", servedName);
    if (enforceEager) args.push("--enforce-eager");

    args.push(
      ...vllmSamplingArgs(resolveSamplingOrNull(model, mp), mp?.thinking, cachedVllmHelpSupport(this.helpKey)),
    );

    // 追加参数放最后：全局在前、按模型在后（argparse 同名参数后者生效）；按 shell 引号规则切分。
    args.push(...splitShellArgs(getSetting(ENGINE_EXTRA_ARGS_KEYS[this.id]) || ""));
    if (mp?.extraArgs) args.push(...splitShellArgs(mp.extraArgs));

    return args;
  }

  async start(): Promise<StartResult> {
    if (this.serverStatus === "running" || this.serverStatus === "starting" || this.serverStatus === "downloading") {
      return { ok: false, error: "Server already running" };
    }

    const { model, servedName } = this.resolveModel();
    if (!model) {
      return { ok: false, error: "No model configured" };
    }

    const binary = await this.checkBinary();
    if (!binary.found) {
      return { ok: false, error: "vLLM 未安装。可在引导页 / 设置里点「一键安装」（仅 Linux），或手动执行 pip install vllm" };
    }

    // venv / 系统 python 走 `-m vllm.entrypoints…`，`vllm` CLI 直接执行。
    const isPython =
      binary.mode === "python" ||
      binary.path?.endsWith("python3") ||
      binary.path?.endsWith("python") ||
      binary.path?.endsWith("python.exe");

    await this.probeHelp(binary.path!, Boolean(isPython));
    const args = this.buildArgs(model, servedName);
    this.launchedArgs = args;
    this.lastError = "";
    this.setStatus("starting");
    const cmd = isPython
      ? [binary.path!, "-m", "vllm.entrypoints.openai.api_server", ...args.slice(1)]
      : [binary.path!, ...args];

    this.appendLog(`$ ${shellJoin(cmd)}\n`);

    try {
      this.serverProcess = spawnServerProcess(cmd, undefined, "vllm");
      pumpServerOutput(this.serverProcess, this.appendLog.bind(this));

      const self = this;
      this.serverProcess.exited
        .then((code) => {
          self.serverProcess = null;
          if (code === 0 || self.getStatus() === "stopped") {
            self.appendLog(`\n[server exited with code ${code}]\n`);
            self.setStatus("stopped");
          } else {
            self.lastError = extractStartupError(
              self.serverLogs,
              `Process exited with code ${code ?? 1}`,
            );
            self.appendLog(`\n[server exited with code ${code}]\n`);
            self.setStatus("error");
          }
        })
        .catch(() => {
          self.serverProcess = null;
          self.setStatus("error");
        });

      const port = this.overrides.port ?? getServerPort(this.id);
      const healthUrl = `http://localhost:${port}/health`;
      const maxIdleAttempts = 180; // vLLM may take longer to load
      let idleCount = 0;
      this.lastDownloadActivityAt = 0;

      while (true) {
        await Bun.sleep(1000);
        const status = this.getStatus();
        if (status !== "starting" && status !== "downloading") break;
        try {
          const res = await fetch(healthUrl, { signal: AbortSignal.timeout(2000) });
          if (res.ok) {
            this.setStatus("running");
            this.appendLog("\n[server is ready]\n");
            markServerStarted();
            return { ok: true };
          }
        } catch {
          // not ready yet
        }

        const downloadActive = Date.now() - this.lastDownloadActivityAt < 5000;
        if (downloadActive) {
          idleCount = 0;
        } else {
          idleCount += 1;
          if (idleCount >= maxIdleAttempts) break;
        }
      }

      const status = this.getStatus();
      if (status === "starting" || status === "downloading") {
        this.lastError = extractStartupError(
          this.serverLogs,
          "Server failed to become ready within timeout",
        );
        this.setStatus("error");
        return { ok: false, error: this.lastError };
      }

      return this.getStatus() === "running"
        ? { ok: true }
        : { ok: false, error: extractStartupError(this.serverLogs, this.lastError) };
    } catch (e) {
      this.lastError = String(e);
      this.setStatus("error");
      return { ok: false, error: this.lastError };
    }
  }

  async stop(): Promise<void> {
    if (!this.serverProcess) {
      this.setStatus("stopped");
      return;
    }

    const proc = this.serverProcess;
    this.setStatus("stopped");
    this.appendLog("\n[stopping server...]\n");

    killProcessTree(proc, "SIGTERM");

    const exited = await Promise.race([
      proc.exited.then(() => true),
      Bun.sleep(5000).then(() => false),
    ]);

    if (!exited) {
      killProcessTree(proc, "SIGKILL");
      await proc.exited.catch(() => {});
    }

    this.serverProcess = null;
  }

  async restart(): Promise<StartResult> {
    await this.stop();
    return this.start();
  }

  forceKill() {
    if (this.serverProcess) {
      try {
        killProcessTree(this.serverProcess, "SIGKILL");
      } catch {
        // already dead
      }
      this.serverProcess = null;
    }
  }
}
