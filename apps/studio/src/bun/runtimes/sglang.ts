import type { Subprocess } from "bun";
import { getSetting, getServerPort, ENGINE_EXTRA_ARGS_KEYS } from "../db/settings";
import { resolveManagedPython } from "../python-engine";
import { markServerStarted } from "../stats";
import { extractStartupError } from "./errors";
import { MAX_LOG_CHARS, killProcessTree, probeCommand, pumpServerOutput, spawnServerProcess } from "./proc";
import { getModelParams } from "../db/model-params";
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



export class SglangRuntime implements Runtime {
  readonly id = "sglang";
  readonly label = "SGLang";

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
    const managed = resolveManagedPython("sglang", "sglang");
    if (managed) return { found: true, path: managed, mode: "python" };

    // Check for python3 with sglang installed
    const pythonPath = Bun.which("python3") ?? Bun.which("python");
    if (!pythonPath) return { found: false };

    // 退出码为 0 才算数：只判"进程退出了"（`waitExit`）会让任何装了 python3 的机器
    // 都报"已安装" —— 导入失败同样是"退出了"。
    if (await probeCommand([pythonPath, "-c", "import sglang; print(sglang.__version__)"], 5000)) {
      return { found: true, path: pythonPath, mode: "python" };
    }

    return { found: false };
  }

  private resolveModel(): { model: string; servedName?: string } {
    // 显式覆盖（已启动模型注册表）优先：同引擎多实例时不能读「当前活动模型」。
    if (this.overrides.model) {
      return { model: this.overrides.model, servedName: this.overrides.servedName };
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
      const customHf = getSetting("CUSTOM_HF_MODEL");
      if (customHf) return { model: customHf.split(":")[0] ?? customHf };
    }

    const customHf = getSetting("CUSTOM_HF_MODEL");
    if (customHf) return { model: customHf.split(":")[0] ?? customHf };

    return { model: "" };
  }

  /** 本次运行实际发出去的 argv（没启动过 = null），needsRestart 的比对基准。 */
  private launchedArgs: string[] | null = null;

  needsRestart(): boolean {
    if (this.launchedArgs === null || this.serverStatus !== "running") return false;
    const { model, servedName } = this.resolveModel();
    return shellJoin(this.buildArgs(model, servedName)) !== shellJoin(this.launchedArgs);
  }

  /**
   * 按模型参数（key = 模型 target）只用得上 ctxSize（→ --context-length）与追加参数：
   * SGLang 没有服务端级的数值采样默认值开关（只有 --sampling-defaults model|openai，
   * 默认就是读模型的 generation_config），采样交给请求体；思考开关同样没有服务端默认。
   */
  private buildArgs(model: string, servedName?: string): string[] {
    const port = this.overrides.port ?? getServerPort(this.id);
    const host = getSetting("SERVER_HOST") || "127.0.0.1";
    const mp = getModelParams(model);
    const contextLength =
      mp?.ctxSize !== undefined ? String(mp.ctxSize) : getSetting("SGLANG_CONTEXT_LENGTH") || "8192";
    const tpSize = getSetting("SGLANG_TP_SIZE") || "1";
    const memFraction = getSetting("SGLANG_MEM_FRACTION_STATIC") || "0.88";
    const chunkedPrefill = getSetting("SGLANG_CHUNKED_PREFILL_SIZE") || "";

    const args: string[] = [
      "-m",
      "sglang.launch_server",
      "--model-path",
      model,
      "--host",
      host,
      "--port",
      port,
      "--context-length",
      contextLength,
      "--tp",
      tpSize,
      "--mem-fraction-static",
      memFraction,
    ];

    if (servedName) args.push("--served-model-name", servedName);

    if (chunkedPrefill && chunkedPrefill !== "0") {
      args.push("--chunked-prefill-size", chunkedPrefill);
    }

    // 追加参数放最后：全局在前、按模型在后（后者盖过前者）；按 shell 引号规则切分。
    args.push(...splitShellArgs(getSetting(ENGINE_EXTRA_ARGS_KEYS[this.id]) || ""));
    if (mp?.extraArgs) args.push(...splitShellArgs(mp.extraArgs));

    return args;
  }

  buildCommandLine(modelOverride?: string): string {
    let model: string;
    let servedName: string | undefined;
    if (modelOverride) {
      model = modelOverride;
    } else {
      const resolved = this.resolveModel();
      model = resolved.model;
      servedName = resolved.servedName;
    }
    // 复制的命令要和实际启动的一致：托管 venv 的 python 优先（见 checkBinary）。
    const python =
      resolveManagedPython("sglang", "sglang") ??
      Bun.which("python3") ??
      Bun.which("python") ??
      "python3";
    return shellJoin([python, ...this.buildArgs(model, servedName)]);
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
      return { ok: false, error: "SGLang 未安装。可在引导页 / 设置里点「一键安装」（仅 Linux），或手动执行 pip install sglang" };
    }

    const args = this.buildArgs(model, servedName);
    this.launchedArgs = args;
    this.lastError = "";
    this.setStatus("starting");

    const cmd = [binary.path!, ...args];
    this.appendLog(`$ ${shellJoin(cmd)}\n`);

    try {
      this.serverProcess = spawnServerProcess(cmd, undefined, "sglang");
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
      const maxIdleAttempts = 180;
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
