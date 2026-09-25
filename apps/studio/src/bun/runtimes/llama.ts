import type { Subprocess } from "bun";
import { existsSync, statSync } from "fs";
import { basename } from "path";
import { EMBEDDING_PORT_BASE } from "../../shared/engines";
import { getModelProfile, type ServerArgs } from "../../shared/model-profiles";
import { parseComputeBufferBytes, parseFlashAttnState, parseKvCacheBytes } from "../../shared/llama-log";
import { logEvent } from "../app-log";
import {
  EMBEDDING_POOLING_VALUES,
  getSetting,
  setServerFlashAttnEffective,
  type SettingsKey,
} from "../db/settings";
import {
  autoTuneMinCtx,
  buildLaunchPlanKeyFromSettings,
  cachedLaunchPlan,
  pairedMmprojPath,
  refreshLaunchPlan,
  type LaunchPlan,
  type LaunchPlanKey,
} from "../launch-plan";
import { llamaCppBinaryPath } from "../engine-paths";
import { KV_CACHE_TYPES, getModelParams } from "../db/model-params";
import { refreshSamplingMetadata, resolveSampling } from "../model-sampling";
import type { ModelParams, ResolvedSampling } from "../../shared/model-params";
import { mainGgufInDir, modelNameForPath } from "../model-scan";
import { slugModelFileName } from "../model-store";
import { markServerStarted } from "../stats";
import { readGgufMeta } from "../gguf-meta";
import { extractStartupError, isOutOfMemoryLog } from "./errors";
import {
  MAX_DEGRADE_RETRIES,
  nextDegradeStep,
  type DegradeStepKind,
  type DegradedLaunch,
  type LaunchAdjust,
  type MemoryKnobs,
} from "./llama-degrade";
import {
  COMMON_LLAMA_SERVER_PATHS,
  cachedFlashAttnSupport,
  cachedKvUnifiedSupport,
  cachedMemoryFlagSupport,
  cachedReasoningSupport,
  cachedServerHelpSupport,
  defaultLlamaServerBinary,
  flashAttnArgs,
  probeServerHelp,
  reasoningArgs,
  type FlashAttnSupport,
  type MemoryFlagSupport,
} from "./llama-flash-attn";
import { shellJoin, splitShellArgs } from "./shell-args";
import { loadModeArgs, loadModeUnsupported, type LoadModeSupport } from "./llama-load-mode";
import { MAX_LOG_CHARS, downloadSourceEnv, killProcessTree, pumpServerOutput, spawnServerProcess, waitExit } from "./proc";
import type {
  BinaryCheckResult,
  LogListener,
  Runtime,
  RuntimeOverrides,
  ServerStatus,
  StartResult,
  StatusListener,
} from "./types";

const DOWNLOAD_PATTERN = /download|fetch|pulling|(\d+(\.\d+)?)\s*%/i;

const COMMON_BINARY_PATHS = COMMON_LLAMA_SERVER_PATHS;

/**
 * 按二进制路径进程级缓存的 `--help` 探测结果（load-mode 与 flash-attn 共用，一次子进程）。
 * 应用托管的那份与 `brew install` 那份版本可能不同，各自探各自的；同一份二进制不必
 * 每次启动都跑 --help。探测失败不落缓存（下回再试），同步路径按「还没探过」处理：
 * load-mode 记 unknown（按默认启动，不赌开关存在），flash-attn 记 none（不发参数，
 * 行为与加这个开关前逐字节一致）。
 */

export async function probeLoadModeSupport(binaryPath: string): Promise<LoadModeSupport> {
  return (await probeServerHelp(binaryPath)).loadMode;
}

/**
 * 同步读已探测到的 load-mode 结果（null = 还没探过）。
 *
 * 给 `buildCommandLine` 用：那个函数是同步的（界面要「可复制的命令」立刻出字），而探测
 * 要跑一次 `--help`。只要本次会话里启动过一次（探测结果按路径缓存），界面复制的命令就
 * 与实际发出去的一致；没启动过则按默认（不发参数）显示 —— 不为了好看去赌版本。
 * 实现委托给 llama-flash-attn 的合并缓存（两个开关共用同一次 --help）。
 */
export function cachedLoadModeSupport(binaryPath: string): LoadModeSupport | null {
  const support = cachedServerHelpSupport(binaryPath);
  if (support === null) return null;
  return support.loadMode;
}

/**
 * 用户设置（三态）+ 上次实测 → 规划用的布尔（T4d 的计价规则）：
 *   off → false；on → true；
 *   auto → 实测 on→true，实测 off→false，未知（从没启动过）→ false（保守，
 *           宁可窗口小也不 OOM）。
 * 非法值（手改设置行塞进来的）按 auto 处理 —— 与 buildArgs 的白名单回落同口径。
 */
export function effectiveFlashAttnForPlan(
  setting: string | null | undefined,
  effective: "" | "on" | "off" | null | undefined,
): boolean {
  const s = (setting ?? "") as string;
  if (s === "off") return false;
  if (s === "on") return true;
  if (s !== "auto") {
    // 非枚举值 → 读侧回落 auto（与 buildArgs / updateSettings 的白名单一致）
  }
  const e = effective ?? "";
  if (e === "on") return true;
  if (e === "off") return false;
  return false; // 从没启动过 → 保守按关计价
}

/**
 * 设置表里持久化的「上次实测」FA（SERVER_FLASH_ATTN_EFFECTIVE）收成三态；
 * 手改进来的非法值当未知（""）。
 *
 * 为什么运行时也要读它：每个 LlamaRuntime 实例的内存态初始都是 ""（未知），而预览 RPC
 * 读的是这条持久化设置 —— 只看内存态的话，重启应用 / 新建实例后运行时永远按「FA 关」
 * 计价，预览却按「FA 开」算，两边 key 不同、预览到的计划不是真正启动用的那份。
 */
export function persistedEffectiveFlashAttn(raw: string | null | undefined): "" | "on" | "off" {
  return raw === "on" || raw === "off" ? raw : "";
}

type LlamaModel = { kind: "local"; path: string; alias: string } | { kind: "hf"; ref: string };

/**
 * 「全部层放 GPU」的 --n-gpu-layers 值。用 999 而不是 `all` / `-1`：数字在所有版本里都认
 * （超过层数会被钳到全部），`all` 是新版才有的写法，`-1` 在带 --fit 的版本里表示「auto」。
 */
const ALL_GPU_LAYERS = "999";

/**
 * MoE 专家下放的 `-ot` 回落写法（没有 --n-cpu-moe 的版本）：前 n 层的专家张量
 * （blk.<i>.ffn_{gate,up,down,gate_up}_exps.weight）放 CPU。与 --n-cpu-moe N 同义。
 */
export function cpuMoeOverrideTensor(n: number): string {
  const layers = Array.from({ length: Math.max(0, Math.floor(n)) }, (_, i) => String(i));
  return `blk\\.(${layers.join("|")})\\.ffn_.*_exps\\.=CPU`;
}

/** buildArgs 里显存相关那几个参数的最终取值（已叠过 launchAdjust），拼 argv 与选降级步骤共用。 */
type ResolvedMemory = {
  ctxSize: string;
  cacheTypeK: string | null;
  cacheTypeV: string | null;
  /** --n-gpu-layers / --n-cpu-moe / -ot / --fit 这一组，按顺序原样进 argv */
  gpuArgs: string[];
  knobs: MemoryKnobs;
};

/**
 * 自动启动参数的缓存 key（运行时 / 预览 RPC 共用这一处，保证逐字段相同）：
 * 设置 + 按模型参数（ctx 作 ctxOverride、parallel / KV 类型盖过设置）+ FA 折算。
 * FA 的「设置值」同样是按模型的优先（它既决定发什么参数，也决定按开还是关计价）。
 */
export function launchPlanKeyForModel(
  modelPath: string,
  modelParams: ModelParams | null,
  get: (key: string) => string,
  effectiveFa: "" | "on" | "off" | null | undefined,
  supportsKvUnified?: boolean | null,
): LaunchPlanKey {
  return buildLaunchPlanKeyFromSettings(
    modelPath,
    get,
    effectiveFlashAttnForPlan(modelParams?.flashAttn ?? get("SERVER_FLASH_ATTN"), effectiveFa),
    { supportsKvUnified, modelParams },
  );
}

/**
 * 采样参数的最终值（按模型 > 模型自带 > 家族推荐 > 全局，见 model-sampling.ts）。
 * 解析器抛错时回落「设置优先、档案兜底」的旧规则 —— 采样值算不出来不能挡住启动。
 */
function samplingValuesFor(
  target: string,
  mp: ModelParams | null,
  serverArgs: ServerArgs,
): ResolvedSampling["values"] {
  try {
    return resolveSampling(target, { override: mp?.sampling, thinking: mp?.thinking }).values;
  } catch (e) {
    logEvent({
      level: "warn",
      source: "server",
      event: "sampling.resolve_failed",
      message: `采样参数解析失败（${e instanceof Error ? e.message : String(e)}），按全局设置`,
    });
    const num = (key: SettingsKey, fallback: number) => {
      const n = Number(getSetting(key));
      return getSetting(key) !== "" && Number.isFinite(n) ? n : fallback;
    };
    return {
      temperature: num("SERVER_TEMP", serverArgs.temp),
      topP: num("SERVER_TOP_P", serverArgs.topP),
      topK: num("SERVER_TOP_K", serverArgs.topK),
      minP: num("SERVER_MIN_P" as SettingsKey, 0),
      presencePenalty: num("SERVER_PRESENCE_PENALTY" as SettingsKey, 0),
      repeatPenalty: num("SERVER_REPEAT_PENALTY", serverArgs.repeatPenalty),
    };
  }
}

/** 数值 → argv 字符串：非有限数不发（返回 null），避免 NaN 进命令行。 */
function numArg(v: number): string | null {
  return Number.isFinite(v) ? String(v) : null;
}

function getSearchPath(): string {
  const home = process.env.HOME ?? "";
  const extra = [
    "/opt/homebrew/bin",
    "/usr/local/bin",
    home ? `${home}/.local/bin` : "",
    home ? `${home}/bin` : "",
  ].filter(Boolean);
  const current = process.env.PATH ?? "";
  return [...extra, current].join(":");
}

export const DEFAULT_CUSTOM_SERVER_ARGS: ServerArgs = {
  ctxSize: 8192,
  imageMaxTokens: 2048,
  batchSize: 256,
  ubatchSize: 64,
  parallel: 1,
  temp: 0.2,
  topP: 0.9,
  topK: 40,
  repeatPenalty: 1.12,
  repeatLastN: 256,
  noMmprojOffload: true,
};



/**
 * llama.cpp 的 `-m` 只收文件。目标是目录时（扫描器把「主模型 + mmproj」的 GGUF 仓库
 * 聚成了目录条目），换成目录里的主 GGUF；找不到就原样返回，让 llama-server 报它的错。
 * 见 `mainGgufInDir` 的注释 —— 那是一个真实起不来的模型。
 */
export function llamaLoadablePath(target: string): string {
  try {
    if (!statSync(target).isDirectory()) return target;
  } catch {
    return target;
  }
  return mainGgufInDir(target) ?? target;
}

export class LlamaRuntime implements Runtime {
  readonly id = "llama.cpp";
  readonly label = "llama-server";

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
    this.attemptLog += text;
    if (this.attemptLog.length > MAX_LOG_CHARS) {
      this.attemptLog = this.attemptLog.slice(-MAX_LOG_CHARS);
    }
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

  /** 本次运行是不是降级起来的、用了什么（没降级 / 没在跑 = null）。界面据此提示「本次临时调小了…」。 */
  getDegradedLaunch(): DegradedLaunch | null {
    return this.degraded;
  }

  /**
   * `--load-mode` / `--flash-attn` 的支持形态（null = 还没探测过）。探测在 start() 里、
   * buildArgs 之前做（一次 --help 出两个），结果按二进制路径缓存在进程级（同一份
   * llama-server 不必每次启动都跑 --help）。
   */
  private loadModeSupport: LoadModeSupport | null = null;
  private flashAttnSupport: FlashAttnSupport | null = null;
  /** `--kv-unified` 支持（同一次 --help 探测；null = 本实例还没探测，回落进程级缓存）。 */
  private kvUnifiedSupport: boolean | null = null;
  /**
   * 本实例上次启动实测的 flash attention（"" = 本实例还没测到）。规划时 "" 回落到设置里
   * 持久化的 SERVER_FLASH_ATTN_EFFECTIVE（见 planFlashAttnEffective），与预览 RPC 同源。
   */
  private lastEffectiveFlashAttn: "" | "on" | "off" = "";
  /**
   * 自动启动参数（SERVER_AUTO_TUNE）的缓存 key：buildArgs / start 共用同一份，
   * 保证「复制的命令」与实际发出去的一致。
   */
  private launchPlanKey: LaunchPlanKey | null = null;
  /** `--reasoning` 支持（同一次 --help 探测；null = 本实例还没探测，回落进程级缓存）。 */
  private reasoningSupport: boolean | null = null;
  /** 本次运行实际发出去的 argv（没在跑 / 没启动过 = null），「改了参数要不要重启」的比对基准。 */
  private launchedArgs: string[] | null = null;
  /** 显存相关开关（--fit / --n-cpu-moe / -ot / --no-context-shift）的探测结果；null = 本实例没探过。 */
  private memoryFlags: MemoryFlagSupport | null = null;
  /**
   * 显存不足降级重试的临时调整（见 llama-degrade.ts）：buildArgs 最后一层叠上它。
   * 只活在本次运行里 —— 每次 start() 清空，不写设置、不改按模型参数。
   */
  private launchAdjust: LaunchAdjust = {};
  /** 本次运行是降级起来的：用了什么（界面 / 日志展示）；正常启动 = null。 */
  private degraded: DegradedLaunch | null = null;
  /** 用户在启动过程中点了停止：降级重试到此为止。 */
  private stopRequested = false;
  /** 正在「启动阶段」的那个子进程：它退出时由 start() 的循环决定是重试还是报错。 */
  private startupProc: Subprocess | null = null;
  /** 本次尝试（一次 spawn）的日志：判断是不是 OOM、挑错误行都只看这一次的。 */
  private attemptLog = "";
  /** 模型层数（GGUF 头；读不到 = null），GPU 层数降级时没有计划也能算 75%。 */
  private modelBlockCount: number | null = null;
  /** 健康检查轮询间隔 / 进程退出后等日志落地 / 两次尝试之间的间隔（测试调小）。 */
  protected healthPollMs = 1000;
  protected exitGraceMs = 250;
  protected retryDelayMs = 1000;

  clearLogs() {
    this.serverLogs = "";
  }

  async checkBinary(): Promise<BinaryCheckResult> {
    // 托管安装优先（引导页 / 设置里「一键安装」下到 <dataDir>/engines/llama.cpp/current）：
    // 官方构建、版本可查；用户自己装过的（Homebrew / PATH）仍然照用，不重复下载。
    const managed = llamaCppBinaryPath();
    if (existsSync(managed)) return { found: true, path: managed, mode: "managed" };
    for (const p of COMMON_BINARY_PATHS) {
      try {
        const f = Bun.file(p);
        if (await f.exists()) return { found: true, path: p };
      } catch {
        // continue
      }
    }
    const p = Bun.which("llama-server", { PATH: getSearchPath() });
    return p ? { found: true, path: p } : { found: false };
  }

  /**
   * Resolve the model this instance serves.
   * Priority: explicit override (served-model registry) → locally installed GGUF path
   * → HF model reference (CUSTOM_HF_MODEL → profile).
   */
  private resolveModel(): { kind: "local"; path: string; alias: string } | { kind: "hf"; ref: string } {
    const target = this.overrides.model;
    if (target) {
      if (existsSync(target)) {
        return {
          kind: "local",
          path: llamaLoadablePath(target),
          alias: this.overrides.servedName ?? slugModelFileName(modelNameForPath(target)),
        };
      }
      return { kind: "hf", ref: target };
    }

    const localPath = getSetting("LOCAL_MODEL_PATH");
    if (localPath) {
      const name = getSetting("LOCAL_MODEL_NAME");
      const alias = name || localPath.split(/[\\/]/).pop()?.replace(/\.gguf$/i, "") || "model";
      return {
        kind: "local",
        path: llamaLoadablePath(localPath),
        alias: alias.toLowerCase().replace(/[^a-z0-9_.-]/g, "-"),
      };
    }

    const profileId = getSetting("VLLM_MODEL_PROFILE");
    const profile = getModelProfile(profileId);
    const hfModel = getSetting("CUSTOM_HF_MODEL") || profile?.hfModel;
    if (hfModel) return { kind: "hf", ref: hfModel };
    return { kind: "hf", ref: "" };
  }

  /**
   * 本实例的模型身份（按模型参数 / 采样解析的 key）：与 resolveModel 同一优先级，但返回
   * **原始 target**（目录条目不换成主 GGUF，HF 引用原样）—— 与已服务模型注册表的
   * modelRef 同一口径，界面按这个 key 存的参数才能被启动时读到。
   */
  private resolveTarget(): string {
    if (this.overrides.model) return this.overrides.model;
    const localPath = getSetting("LOCAL_MODEL_PATH");
    if (localPath) return localPath;
    const profile = getModelProfile(getSetting("VLLM_MODEL_PROFILE"));
    return getSetting("CUSTOM_HF_MODEL") || profile?.hfModel || "";
  }

  /** 公开（同进程内的 UI / 测试入口与 `buildCommandLine` 共用）：模型档案的 serverArgs。 */
  getProfileServerArgs(): ServerArgs {
    const profileId = getSetting("VLLM_MODEL_PROFILE");
    const profile = getModelProfile(profileId);
    return profile?.serverArgs ?? DEFAULT_CUSTOM_SERVER_ARGS;
  }

  /** 设置里是否开了「自动启动参数」：只认 "1"（set 时已做白名单，这里再收一遍）。 */
  private static isAutoTuneEnabled(): boolean {
    return getSetting("SERVER_AUTO_TUNE" as SettingsKey) === "1";
  }

  /**
   * 自动启动参数的缓存 key：设置（并发 / batch / ubatch / 缓存类型 / FA）+ 已解析的模型路径。
   * ctxOverride 固定为 null —— 自动模式下不把设置里的 ctx 当成「用户显式指定」，
   * 否则规划器会尊重它、自动推算就失去意义。flashAttn 按「设置值 + 上次实测」折算
   * （见 effectiveFlashAttnForPlan），unknown 时保守按关计价。
   * 非本地 GGUF（HF 引用 / 未解析到模型）没有可算的对象，返回 null。
   */
  private buildLaunchPlanKey(model: LlamaModel, target: string): LaunchPlanKey | null {
    if (model.kind !== "local") return null;
    // key 的计算收敛在 launchPlanKeyForModel（与预览 RPC、测试共用同一份）。
    return launchPlanKeyForModel(
      model.path,
      getModelParams(target),
      (k) => getSetting(k as SettingsKey),
      this.planFlashAttnEffective(),
      // start() 里刚探测过就用当次结果（与实际启动的那份二进制一致）；没探过由
      // launch-plan 读进程级缓存 —— 与预览 RPC 同一条规则。
      this.kvUnifiedSupport,
    );
  }

  /**
   * 规划用的「上次实测」FA：本实例测到过就用本实例的（最新），否则读持久化设置。
   * 与预览 RPC（直接读 SERVER_FLASH_ATTN_EFFECTIVE）同源 —— 回读时两者同时写，必然相等。
   */
  private planFlashAttnEffective(): "" | "on" | "off" {
    return this.lastEffectiveFlashAttn || persistedEffectiveFlashAttn(getSetting("SERVER_FLASH_ATTN_EFFECTIVE"));
  }

  /** 数值型设置收成非负整数串，非法 / 空 → fallback（argv 注入防御，见 buildArgs）。 */
  private static cleanNum(raw: string, fallback: string): string {
    const n = Number(raw);
    return raw.trim() !== "" && Number.isFinite(n) && n >= 0 && Number.isSafeInteger(n)
      ? String(n)
      : fallback;
  }

  /**
   * 本实例监听的端口 —— 拼参数（--port）、健康检查、启动回读（/props）**只认这一处**。
   * 优先 overrides.port（served-model 注册表分配的）；否则嵌入实例（purpose=embedding）
   * 回落嵌入端口段（EMBEDDING_PORT，默认 18190），不碰聊天默认端点；聊天实例用 SERVER_PORT。
   * 以前健康检查自己按 SERVER_PORT 算，嵌入实例起在 18190 却去轮询 8080：要么一直等到超时，
   * 要么撞上正在跑的聊天实例、把没起来的嵌入实例误报成 running。
   */
  private resolvePort(): string {
    if (this.overrides.port) return this.overrides.port;
    return this.overrides.purpose === "embedding"
      ? LlamaRuntime.cleanNum(getSetting("EMBEDDING_PORT") || "", String(EMBEDDING_PORT_BASE))
      : LlamaRuntime.cleanNum(getSetting("SERVER_PORT") || "", "8080");
  }

  /**
   * 启动后的实测回读（T4e）：健康检查通过后从运行中的实例把「预测」之外的另一半
   * 数字捞回来 —— `/props` 的 n_ctx（每 slot 窗口）× total_slots = 总窗口、启动日志里
   * 的 flash attention 实际状态与 KV / compute buffer 实际字节数，然后记一条
   * `launch_plan.measured`（预测 vs 实测同框，以后校准公式的全部依据），并把
   * flash attention 实测值写进 SERVER_FLASH_ATTN_EFFECTIVE（下次 auto 规划就不再保守计价）。
   *
   * 任何一步失败都只是记 `launch_plan.readback_failed`（warn），**绝不影响启动结果**：
   * 回读是对已完成启动的补充记录，没有它启动依然是成功的。
   */
  private async readbackMeasured(): Promise<void> {
    const port = this.resolvePort();
    const detail: Record<string, string | number | null> = {
      model: basename(this.resolvedModelPath()),
    };

    const plan = this.autoPlan(this.resolveModel(), this.resolveTarget());
    if (plan !== null) {
      detail.predictedCtx = plan.ctxTokens;
      detail.predictedPerSlot = plan.ctxPerSlot;
      detail.predictedKv = plan.estimates.kvBytes;
    }

    // 降级起来的：预测与实测对不上是预期的，记下用了什么，校准时别把它当成公式误差
    if (this.degraded !== null) detail.degraded = this.degraded.summary;

    const log = this.getLogs();
    const flash = parseFlashAttnState(log);
    detail.flashAttn = flash;
    if (flash !== null) {
      setServerFlashAttnEffective(flash);
      this.lastEffectiveFlashAttn = flash; // 本次会话的下一次规划立刻受益
    }

    const kv = parseKvCacheBytes(log);
    if (kv !== null) detail.actualKv = kv;
    const compute = parseComputeBufferBytes(log);
    if (compute !== null) detail.actualCompute = compute;

    let actualCtxPerSlot: number | null = null;
    let actualSlots: number | null = null;
    const propsRes = await fetch(`http://localhost:${port}/props`, {
      signal: AbortSignal.timeout(3000),
    });
    if (propsRes.ok) {
      const props = (await propsRes.json()) as {
        default_generation_settings?: { n_ctx?: number } | null;
        total_slots?: number;
      };
      actualCtxPerSlot = Number.isFinite(props.default_generation_settings?.n_ctx)
        ? Number(props.default_generation_settings?.n_ctx)
        : null;
      actualSlots = Number.isFinite(props.total_slots) ? Number(props.total_slots) : null;
      // n_ctx 是**每 slot**的窗口（--ctx-size 会在 slot 间均分），总窗口要乘回去
      if (actualCtxPerSlot !== null) detail.actualCtxPerSlot = actualCtxPerSlot;
      if (actualSlots !== null) detail.actualSlots = actualSlots;
      if (actualCtxPerSlot !== null && actualSlots !== null) {
        detail.actualCtxTotal = actualCtxPerSlot * actualSlots;
      }
    }

    logEvent({
      level: "info",
      source: "server",
      event: "launch_plan.measured",
      message: `llama.cpp 启动回读（${detail.model}）`,
      detail,
    });

    if (plan !== null && typeof detail.actualCtxTotal === "number") {
      const diff = Math.abs(plan.ctxTokens - detail.actualCtxTotal) / plan.ctxTokens;
      if (diff > 0.05) {
        logEvent({
          level: "warn",
          source: "server",
          event: "launch_plan.mismatch",
          message: `预测窗口与实测相差超过 5%（${detail.model}）`,
          detail: { model: detail.model, predictedCtx: plan.ctxTokens, actualCtxTotal: detail.actualCtxTotal },
        });
      }
    }
  }

  /** 本次启动解析到的本地模型路径（HF 引用 / 未解析 → 空串），回读日志的 model 字段用。 */
  private resolvedModelPath(): string {
    const model = this.resolveModel();
    return model.kind === "local" ? model.path : "";
  }

  /** 本次自动推算用的计划（同步读缓存）；未开自动 / 非 GGUF / 没算过 → null，调用方回落到设置值。
   *  key 由调用方传入（start 用自己缓存的这份；buildCommandLine 用模型现建），
   *  两处 key 同一模型时逐字段相同，保证「复制的命令」与实际发出去的一致。 */
  private autoPlan(model: LlamaModel, target: string): LaunchPlan | null {
    if (!LlamaRuntime.isAutoTuneEnabled()) return null;
    const fresh = this.buildLaunchPlanKey(model, target);
    if (fresh === null) return null;
    // key 每次现建（按模型参数改了 → key 变 → 预览 / 重启判定立刻看到新值）；唯一沿用启动那份的
    // 是 FA 折算：启动回读会把「实测 FA」写回，只因为它就换 key 会让正在跑的实例读不到自己的计划。
    const launched = this.launchPlanKey;
    const key =
      launched !== null && launched.modelPath === fresh.modelPath
        ? { ...fresh, flashAttn: launched.flashAttn }
        : fresh;
    const plan = cachedLaunchPlan(key);
    return plan !== null && plan.ctxTokens > 0 ? plan : null;
  }

  /** 一行中文摘要（GiB 保留一位小数，reasons 只列 code）。 */
  private autoPlanLogLine(plan: LaunchPlan): string {
    const giB = (bytes: number) => (bytes / (1024 ** 3)).toFixed(1);
    const reasons = plan.reasons.map((r) => r.code);
    return (
      `[auto] 上下文 ${plan.ctxTokens}（每请求 ${plan.ctxPerSlot}，${plan.parallel} 并发）` +
      ` · KV ${giB(plan.estimates.kvBytes)} GiB · 预算 ${giB(plan.estimates.budgetBytes)} GiB` +
      (reasons.length > 0 ? ` · 依据: ${reasons.join(", ")}` : "")
    );
  }

  buildCommandLine(modelOverride?: string): string {
    let model: LlamaModel;
    if (modelOverride) {
      if (existsSync(modelOverride)) {
        // 与 resolveModel 同一规则：目录条目换成主 GGUF（`-m` 只收文件），
        // 否则复制出来的命令与 start() 实际发的不一致。
        model = {
          kind: "local",
          path: llamaLoadablePath(modelOverride),
          alias: slugModelFileName(modelNameForPath(modelOverride)),
        };
      } else {
        model = { kind: "hf", ref: modelOverride };
      }
    } else {
      model = this.resolveModel();
    }
    // 用户终端直接跑原生命令，不带 macOS PTY 包装。按 shell 规则加引号：路径带空格 /
    // JSON 参数原样粘进终端也能跑，且切回来与 argv 逐项相等。
    const bin = defaultLlamaServerBinary();
    const target = modelOverride || this.resolveTarget();
    return shellJoin([bin, ...this.buildArgs(model, this.getProfileServerArgs(), target)]);
  }

  /**
   * 正在跑的实例与「按现在的设置 / 按模型参数再起一次」的 argv 是否不同（界面据此提示
   * 「改动需重启生效」）。不自动重启 —— 重启会打断进行中的请求，由用户决定。
   * 没在跑 / 没有启动记录 → false。
   */
  needsRestart(): boolean {
    if (this.launchedArgs === null || this.serverStatus !== "running") return false;
    const current = this.buildArgs(this.resolveModel(), this.getProfileServerArgs(), this.resolveTarget());
    return shellJoin(current) !== shellJoin(this.launchedArgs);
  }

  /** 显存相关开关的支持：本实例探过用本实例的，否则读进程级缓存（没探过 = 全不支持）。 */
  private memoryFlagSupport(): MemoryFlagSupport {
    return this.memoryFlags ?? cachedMemoryFlagSupport(defaultLlamaServerBinary());
  }

  /**
   * 显存相关参数的最终取值：上下文、KV 类型、GPU 层数 / MoE 专家下放 / --fit。
   *
   * 优先级（低 → 高）：档案默认 < 全局设置 < 自动计划 < 按模型参数 < **launchAdjust**（降级重试的
   * 临时调整，永远最后叠，所以复制的命令就是降级后实际发出去的那条）。
   *
   * GPU 这一组的决策（只在用户没固定层数、且有自动计划时才自动决定）：
   *  - 计划说没有 GPU（gpuLayers === 0）：一个参数都不发（引擎自己决定）；
   *  - MoE 专家下放（计划给了 moeOffload，且引擎认 --n-cpu-moe 或 -ot）：
   *    计划装得下 → 层全留 GPU + 前 N 层专家放 CPU（认 --fit 的再加 --fit off，别让它再挪）；
   *    装不下 → 认 --fit 就整组交给 `--fit on`，不认照样按 N 下放；
   *  - 其它：计划「证明装得下」（fits 且不用按层卸载）→ 认 --fit 就 `-ngl 999 --fit off`
   *    （新版 --fit 默认 on，装得下时它仍会为留余量把约 1 GiB 挪出 GPU —— Unsloth 实测）；
   *    证明不了（按层卸载 / 装不下）→ 认 --fit 就只发 `--fit on`、不发层数（llama.cpp 按真实空闲
   *    显存放，比我们估得准，给了层数它反而不动）；不认 --fit 回落计划的层数（老行为）。
   * 没有计划（关了自动 / 非 GGUF）时与加这些开关之前逐字节一致：只有用户固定的层数。
   */
  private resolveMemory(mp: ModelParams | null, autoPlan: LaunchPlan | null, serverArgs: ServerArgs): ResolvedMemory {
    const cleanNum = LlamaRuntime.cleanNum;
    const adj = this.launchAdjust;
    const mem = this.memoryFlagSupport();

    // 自动启动参数（SERVER_AUTO_TUNE，默认开）：有计划时 --ctx-size 用计划值。
    // 按模型固定的窗口：自动规划开着时它已作为 ctxOverride 进了计划（计划的 ctxTokens 就是它，
    // 其余参数照样拟合）；没有计划（关了自动 / 非 GGUF / 没算过）时直接用它。
    let ctxSize = autoPlan
      ? String(autoPlan.ctxTokens)
      : mp?.ctxSize !== undefined
        ? String(mp.ctxSize)
        : cleanNum(getSetting("SERVER_CTX_SIZE") || "", String(serverArgs.ctxSize));
    if (adj.ctxCap !== undefined && Number(ctxSize) > adj.ctxCap) ctxSize = String(adj.ctxCap);

    // llama.cpp 的 KV 量化枚举（本机 `--help` 的 allowed values，与规划器 KV 表同源）：手改设置行
    // 塞进来的值不能进 argv，白名单之外一律不落参数（引擎用自己的默认）。按模型的优先，降级调整再盖过它。
    const kvTypes = new Set<string>(KV_CACHE_TYPES);
    const pickKv = (adjusted: string | undefined, perModel: string | undefined, key: SettingsKey): string | null => {
      const v = adjusted ?? perModel ?? getSetting(key);
      return kvTypes.has(v) ? v : null;
    };
    const cacheTypeK = pickKv(adj.cacheTypeK, mp?.cacheTypeK, "SERVER_CACHE_TYPE_K");
    const cacheTypeV = pickKv(adj.cacheTypeV, mp?.cacheTypeV, "SERVER_CACHE_TYPE_V");

    // 「自动（全卸载）」哨兵值：与 llama.cpp 引擎自身的 -1 同值（不传参数 = 引擎自己决定），
    // 这里额外接受 "" / "auto" 作为同义写法 —— 自动推算的 gpuLayers 建议只在用户没
    // 显式指定时采纳（哨兵 → 计划值；具体数字 → 听用户的）。具体数字要能被解析成
    // 有限整数才进 argv（手改设置行塞进来的垃圾值丢弃，行为等同哨兵 = 引擎自己决定）。
    // 按模型的层数优先：-1 = 显式「交给引擎 / 自动规划」（盖过全局的固定层数），>=0 = 固定。
    const rawGpuLayers = mp?.gpuLayers !== undefined ? String(mp.gpuLayers) : getSetting("SERVER_GPU_LAYERS");
    const gpuAuto = rawGpuLayers === "-1" || rawGpuLayers === "" || rawGpuLayers.toLowerCase() === "auto";
    const pinnedLayers =
      !gpuAuto && /^-?\d+$/.test(rawGpuLayers.trim()) && Number.isSafeInteger(Number(rawGpuLayers))
        ? rawGpuLayers
        : "";

    let ngl: string | null = null;
    let nCpuMoe: number | null = null;
    let fit: "on" | "off" | null = null;
    if (pinnedLayers) {
      ngl = pinnedLayers;
    } else if (gpuAuto && autoPlan !== null && autoPlan.gpuLayers !== 0) {
      const moe = autoPlan.moeOffload ?? null;
      if (moe !== null && (mem.nCpuMoe || mem.overrideTensor)) {
        if (mem.fit && !autoPlan.fits) {
          fit = "on";
        } else {
          ngl = ALL_GPU_LAYERS;
          nCpuMoe = moe.cpuMoeLayers;
          if (mem.fit) fit = "off";
        }
      } else {
        // 引擎不认专家下放时回落按层卸载的建议值
        const layers = moe !== null ? moe.fallbackGpuLayers : autoPlan.gpuLayers;
        const proven = autoPlan.fits && layers === null;
        if (mem.fit) {
          if (proven) {
            ngl = ALL_GPU_LAYERS;
            fit = "off";
          } else {
            fit = "on";
          }
        } else if (layers !== null && layers !== 0) {
          ngl = String(layers);
        }
      }
    }

    // 降级重试的临时调整（最后叠）
    if (adj.fit === "on" && mem.fit) {
      fit = "on";
      ngl = null;
      nCpuMoe = null;
    }
    if (adj.nCpuMoe !== undefined && nCpuMoe !== null) nCpuMoe = adj.nCpuMoe;
    if (adj.gpuLayers !== undefined) ngl = String(adj.gpuLayers);

    const gpuArgs: string[] = [];
    if (ngl !== null) gpuArgs.push("--n-gpu-layers", ngl);
    if (nCpuMoe !== null && nCpuMoe > 0) {
      if (mem.nCpuMoe) gpuArgs.push("--n-cpu-moe", String(nCpuMoe));
      else gpuArgs.push("--override-tensor", cpuMoeOverrideTensor(nCpuMoe));
    }
    if (fit !== null) gpuArgs.push("--fit", fit);

    const partialDetail = autoPlan?.reasons.find((r) => r.code === "gpu.partial-offload")?.detail?.blockCount;
    const blockCount =
      autoPlan?.moeOffload?.blockCount ??
      (typeof partialDetail === "number" ? partialDetail : null) ??
      this.modelBlockCount;
    const rawFlash = mp?.flashAttn ?? getSetting("SERVER_FLASH_ATTN");
    const knobs: MemoryKnobs = {
      ctx: Number(ctxSize),
      ctxPinned: mp?.ctxSize !== undefined,
      cacheTypeK,
      cacheTypeV,
      gpuLayers: ngl !== null ? Number(ngl) : null,
      gpuLayersPinned: pinnedLayers !== "",
      nCpuMoe: nCpuMoe !== null && nCpuMoe > 0 ? nCpuMoe : null,
      fit,
      blockCount,
      fitSupported: mem.fit,
      flashAttnOff: rawFlash === "off" || this.planFlashAttnEffective() === "off",
      minCtx: autoTuneMinCtx(),
    };
    return { ctxSize, cacheTypeK, cacheTypeV, gpuArgs, knobs };
  }

  /** 公开（`buildCommandLine` / `start` / 同进程内 UI 入口共用）：拼装 llama-server 参数。 */
  buildArgs(model: LlamaModel, serverArgs: ServerArgs, target?: string): string[] {
    const cleanNum = LlamaRuntime.cleanNum;
    // 按模型参数的 key：调用方给的原始 target；没给（测试直调）按实例 target → 模型本身兜底。
    const modelKey = target ?? (this.overrides.model || (model.kind === "local" ? model.path : model.ref));
    // 按模型参数（已校验收敛，见 db/model-params）：逐字段盖过自动规划与全局设置。
    const mp = getModelParams(modelKey);
    // 数值型设置（ctx / batch / parallel / port / …）的注入防御：这些值会原样进 argv，而
    // 设置行可能被手改数据库 / 旧版本写入，所以统一收一遍 —— 解析不出有限非负整数就丢弃
    // （空串保持空，让 `|| 档案默认` 兜底；非法值同样落回默认而不是把字符串塞给引擎）。
    const embedding = this.overrides.purpose === "embedding";
    // 端口与健康检查 / 回读同源（resolvePort）：嵌入实例落嵌入端口段，聊天实例 SERVER_PORT。
    const port = this.resolvePort();
    const host = getSetting("SERVER_HOST") || "127.0.0.1";
    // 自动启动参数（SERVER_AUTO_TUNE，默认开）：有计划时 --ctx-size / --batch-size /
    // --ubatch-size（仅聊天实例）用计划值；--parallel 仍用设置值（并发是用户的业务选择），
    // --cache-type-k/-v 保持设置值（它们参与计划 key 的计算，改了自然重算）。
    // 按模型固定的窗口：自动规划开着时它已作为 ctxOverride 进了计划（计划的 ctxTokens 就是它，
    // 其余参数照样拟合）；没有计划（关了自动 / 非 GGUF / 没算过）时直接用它。
    const autoPlan = this.autoPlan(model, modelKey);
    // 上下文 / KV 类型 / GPU 层数这一组收在 resolveMemory（同一份取值也给降级重试挑下一步用）。
    const memory = this.resolveMemory(mp, autoPlan, serverArgs);
    const ctxSize = memory.ctxSize;
    const imageMaxTokens = cleanNum(
      getSetting("SERVER_IMAGE_MAX_TOKENS") || "",
      String(serverArgs.imageMaxTokens),
    );
    const batchSize = autoPlan
      ? String(autoPlan.batch)
      : cleanNum(getSetting("SERVER_BATCH_SIZE") || "", String(serverArgs.batchSize));
    const ubatchSize = autoPlan
      ? String(autoPlan.ubatch)
      : cleanNum(getSetting("SERVER_UBATCH_SIZE") || "", String(serverArgs.ubatchSize));
    const parallel =
      mp?.parallel !== undefined
        ? String(mp.parallel)
        : cleanNum(getSetting("SERVER_PARALLEL") || "", String(serverArgs.parallel));
    const { cacheTypeK, cacheTypeV } = memory;
    // 池化方式同理：设置键在 set 时有枚举校验，这里再收一遍（db 被手改 / 旧值兜底）。
    const pooling = (EMBEDDING_POOLING_VALUES as readonly string[]).includes(
      getSetting("EMBEDDING_POOLING"),
    )
      ? (getSetting("EMBEDDING_POOLING") as string)
      : "last";
    // 嵌入模式的物理 batch（n_batch = n_ubatch）：取 ctx-size，即「塞得进上下文的
    // 文本就一定嵌得进去」。**不能沿用聊天调优的 256/64** —— llama.cpp 在
    // `--embeddings` 下会强制 n_batch = n_ubatch（显式传值才不会被压到 512），
    // 而物理 batch 就是单次能喂进模型的 token 上限：超过它的文档在 pooling=last 时
    // 触发 GGML 断言直接崩进程（SIGTRAP / exit 5），pooling=mean 时返回 500。
    // KB 导入的 markdown 轻松超过 512 token，这正是「导入即崩」的根因。
    // ctxSize 这里已是 cleanNum 收敛过的纯数字串，直接解析即可。
    const embedBatch = String(Number(ctxSize) || 8192);

    const args: string[] = [];

    // 这条命令里有没有视觉投影：本地模型看同目录有没有配对到 mmproj；hf ref 由 llama-server
    // 自己按仓库拉（-hf 会顺带下载 mmproj），沿用旧行为按「有」处理。
    // 没有投影时 --image-max-tokens / --no-mmproj-offload 都是空转参数，不发。
    let hasVision = model.kind === "hf";

    if (model.kind === "local") {
      args.push("-m", model.path, "--alias", model.alias);
      // 多模态（mmproj）：本地模型按同目录自动配对投影文件（规则见 pairedMmprojPath，
      // 与规划器扣的那份字节数同源）。聊天实例也要注入 —— 以前「聊天实例永不注入」，
      // 本地视觉模型收到图片也看不见。嵌入实例 spike（llama-server b9410）实证
      // `--embeddings --pooling last --mmproj` 共存可用。
      // hf ref 走 -hf 自管缓存拿不到本地路径，不注入（Non-Goal）。
      const mmproj = pairedMmprojPath(model.path);
      if (mmproj !== null) {
        args.push("--mmproj", mmproj);
        hasVision = true;
      }
    } else if (model.ref) {
      args.push("-hf", model.ref);
    }

    args.push(
      "--host",
      host,
      "--port",
      port,
      "--ctx-size",
      ctxSize,
    );

    // 嵌入模式裁剪的聊天参数：--temp/--top-p/--top-k/--repeat-penalty/--repeat-last-n/--image-max-tokens。
    // --image-max-tokens 只在真有投影文件时发（纯文本模型发它没有意义）。
    if (!embedding && hasVision) {
      args.push(
        "--image-max-tokens",
        imageMaxTokens,
      );
    }

    // KV 缓存类型只有白名单内的值才落参数（嵌入实例也发，两边都需要）；
    // 手改设置行塞进来的值一律不发，让引擎用自己的默认。
    args.push(
      "--parallel",
      parallel,
      "--batch-size",
      embedding ? embedBatch : batchSize,
      "--ubatch-size",
      embedding ? embedBatch : ubatchSize,
    );
    // --kv-unified：多个 slot 共用一整块 KV，每个请求都能用满 --ctx-size；不加的话
    // llama.cpp 把总窗口按 slot 均分（parallel=4 时每个请求只剩 1/4）。
    // 有计划时听计划（规划器已按「引擎支持与否」定了口径），没计划时 parallel > 1 就要；
    // 两种情况都只在这台 llama-server 的 --help 里确实有这个开关时才发（没探过不赌）。
    const kvuSupported =
      this.kvUnifiedSupport ?? cachedKvUnifiedSupport(defaultLlamaServerBinary()) ?? false;
    const wantKvUnified = autoPlan ? autoPlan.kvUnified : Number(parallel) > 1;
    if (wantKvUnified && kvuSupported) args.push("--kv-unified");
    if (cacheTypeK) args.push("--cache-type-k", cacheTypeK);
    if (cacheTypeV) args.push("--cache-type-v", cacheTypeV);

    // 采样默认值（请求没带时引擎用它）：按模型 > 模型自带 > 家族推荐 > 全局（model-sampling.ts）。
    // 嵌入实例不采样，整组不发。
    if (!embedding) {
      const s = samplingValuesFor(modelKey, mp, serverArgs);
      const sampling: [string, string | null][] = [
        ["--repeat-penalty", numArg(s.repeatPenalty)],
        ["--repeat-last-n", String(serverArgs.repeatLastN)],
        ["--temp", numArg(s.temperature)],
        ["--top-p", numArg(s.topP)],
        ["--top-k", numArg(s.topK)],
        ["--min-p", numArg(s.minP)],
        ["--presence-penalty", numArg(s.presencePenalty)],
      ];
      for (const [flag, v] of sampling) if (v !== null) args.push(flag, v);
      // 思考开关（按模型）：认 --reasoning 就直接交给引擎，老版关思考回落 chat-template-kwargs。
      const reaSupported =
        this.reasoningSupport ?? cachedReasoningSupport(defaultLlamaServerBinary()) ?? false;
      args.push(...reasoningArgs(mp?.thinking, reaSupported));
      // 超窗报错而不是悄悄丢前文（context shift 会把最早的对话挤掉，模型「忘了」开头却没人知道）。
      // 新版默认已经是关，老版默认开 —— 显式发出去两边行为一致；只在 --help 认这个开关时发。
      if (this.memoryFlagSupport().noContextShift) args.push("--no-context-shift");
    }

    // 嵌入实例追加嵌入开关与池化方式（llama.cpp 默认禁用嵌入端点，这就是 501 的根因）。
    if (embedding) {
      args.push(
        "--embeddings",
        "--pooling",
        pooling,
      );
    }

    // GPU 层数 / MoE 专家下放 / --fit（决策见 resolveMemory）
    args.push(...memory.gpuArgs);

    // 加载模式（PERF-02）：权重 mmap / 锁内存的取舍 —— 系统内存紧张时是「换出去一点」
    // 还是「整机卡住」，由它决定。按 --help 探测结果决定发新版 --load-mode 还是旧版
    // 的 --mlock / --no-mmap（见 llama-load-mode.ts 的等价表）；界面复制的命令读同一份
    // 缓存，所以只要启动过一次，显示与实际发出去的就是同一串。
    const loadModeSupport =
      this.loadModeSupport ?? cachedLoadModeSupport(defaultLlamaServerBinary()) ?? "unknown";
    args.push(...loadModeArgs(getSetting("SERVER_LOAD_MODE"), loadModeSupport));

    // flash attention（T4d）：三态开关按 --help 探测结果折算 —— 新版发 [--flash-attn, 值]，
    // 老版布尔开关只有 on 才发，没探测到就一个参数都不发（与加这个开关前逐字节一致）。
    // flashAttnArgs 自己把设置值收进白名单（非法值回落 auto），这里直传原始值。
    const rawFlash = mp?.flashAttn ?? getSetting("SERVER_FLASH_ATTN");
    const faSupport =
      this.flashAttnSupport ?? cachedFlashAttnSupport(defaultLlamaServerBinary()) ?? "none";
    args.push(...flashAttnArgs(rawFlash, faSupport));

    // 投影留在 CPU：只有真带了投影时才有意义（见上面 hasVision）。
    if (serverArgs.noMmprojOffload && hasVision) {
      args.push("--no-mmproj-offload");
    }

    // 追加参数放最后（llama.cpp 同名参数后者生效）：全局在前、按模型在后，所以按模型的能盖过全局。
    // 按 shell 引号规则切分（'{"enable_thinking": false}' 这类带空格的 JSON 不能被拆开）。
    args.push(...splitShellArgs(getSetting("SERVER_EXTRA_ARGS") || ""));
    if (mp?.extraArgs) args.push(...splitShellArgs(mp.extraArgs));

    return args;
  }

  async start(): Promise<StartResult> {
    if (this.serverStatus === "running" || this.serverStatus === "starting" || this.serverStatus === "downloading") {
      return { ok: false, error: "Server already running" };
    }

    const serverArgs = this.getProfileServerArgs();

    const model = this.resolveModel();
    const target = this.resolveTarget();
    if (model.kind === "hf" && !model.ref) {
      return { ok: false, error: "No model configured" };
    }

    const binary = await this.checkBinary();
    if (!binary.found) {
      return { ok: false, error: "llama-server not found on PATH" };
    }
    const llamaPath = binary.path!;

    // 加载模式 + flash attention 共用一次 --help 探测（新版 --load-mode / 旧版 --mlock；
    // 三态 --flash-attn / 布尔 -fa / 没有），再拼参数。
    const helpSupport = await probeServerHelp(llamaPath);
    this.loadModeSupport = helpSupport.loadMode;
    this.flashAttnSupport = helpSupport.flashAttn;
    this.kvUnifiedSupport = helpSupport.kvUnified ?? false;
    this.reasoningSupport = helpSupport.reasoning ?? false;
    this.memoryFlags = {
      fit: helpSupport.fit ?? false,
      nCpuMoe: helpSupport.nCpuMoe ?? false,
      overrideTensor: helpSupport.overrideTensor ?? false,
      noContextShift: helpSupport.noContextShift ?? false,
    };
    const loadMode = getSetting("SERVER_LOAD_MODE");
    if (loadModeUnsupported(loadMode, this.loadModeSupport)) {
      const message = `加载模式 ${loadMode} 在这台 llama-server（${this.loadModeSupport}）上不支持，本次按默认加载模式启动`;
      this.appendLog(`\n[omni] ${message}\n`);
      logEvent({
        level: "warn",
        source: "server",
        event: "engine.load_mode.unsupported",
        message,
        detail: { mode: loadMode, support: this.loadModeSupport, binary: llamaPath },
      });
    }

    // 自动启动参数：异步算一次、写进按 key 的缓存（buildArgs / buildCommandLine 只同步读）。
    // 任何失败都不能挡住启动 —— 计划缺失时全部回落设置值，与不开自动时逐字节一致。
    this.launchPlanKey = null;
    if (
      LlamaRuntime.isAutoTuneEnabled() &&
      model.kind === "local" &&
      /\.gguf$/i.test(model.path)
    ) {
      const key = this.buildLaunchPlanKey(model, target);
      if (key !== null) {
        try {
          await refreshLaunchPlan(key);
          this.launchPlanKey = key;
          const plan = cachedLaunchPlan(key);
          if (plan !== null && plan.ctxTokens > 0) {
            this.appendLog(`\n[omni] ${this.autoPlanLogLine(plan)}\n`);
          } else {
            this.appendLog("\n[omni] [auto] 未得到可用计划，本次启动按设置值\n");
          }
        } catch (e) {
          logEvent({
            level: "warn",
            source: "server",
            event: "launch_plan.failed",
            message: `自动启动参数计算失败（${e instanceof Error ? e.message : String(e)}），按设置值启动`,
            detail: { file: model.path },
          });
        }
      }
    }

    // 模型自带的采样元数据（generation_config / GGUF general.sampling.*）先异步读进缓存，
    // buildArgs 里同步解析才拿得到；读不到不影响启动（回落家族推荐 / 全局）。
    if (target) {
      try {
        await refreshSamplingMetadata(target);
      } catch {
        // 元数据读取失败：按没有处理
      }
    }

    // 模型层数：GPU 层数降级在没有计划时也要能算「75%」。读不到（不是 GGUF / HF 引用）按未知。
    this.modelBlockCount = null;
    if (model.kind === "local") {
      try {
        const read = await readGgufMeta(model.path);
        if (read.ok) this.modelBlockCount = read.data.meta.blockCount;
      } catch {
        // 元数据读取失败：按未知
      }
    }

    // —— 启动 + 显存不足时的降级重试（见 llama-degrade.ts）——
    // 每次 start() 都从计划重新开始：上次降级时被占的显存现在可能已经空出来了。
    this.launchAdjust = {};
    this.degraded = null;
    this.stopRequested = false;
    const used = new Set<DegradeStepKind>();
    const steps: string[] = [];
    let retries = 0;
    while (true) {
      const attempt = await this.launchOnce(llamaPath, model, serverArgs, target);
      if (attempt.ok) {
        if (retries > 0) {
          logEvent({
            level: "warn",
            source: "server",
            event: "launch.degrade.succeeded",
            message: `显存不足，降级 ${retries} 次后启动成功（${basename(this.resolvedModelPath())}）：${steps.join("；")}`,
            detail: { model: basename(this.resolvedModelPath()), attempts: retries, ...this.launchAdjust },
          });
        }
        return { ok: true };
      }
      if (attempt.aborted || this.stopRequested) {
        // 用户停了：不再重试，也不把「停止」记成错误
        this.degraded = null;
        if (retries > 0) this.appendLog("\n[omni] 已停止，放弃降级重试\n");
        return { ok: false, error: attempt.error };
      }

      const oom = isOutOfMemoryLog(this.attemptLog);
      const step =
        oom && retries < MAX_DEGRADE_RETRIES
          ? nextDegradeStep(
              // start() 总是把 target 传给 buildArgs，这里用同一个 key 读按模型参数与计划
              this.resolveMemory(getModelParams(target), this.autoPlan(model, target), serverArgs).knobs,
              this.launchAdjust,
              used,
            )
          : null;
      if (step === null) {
        if (oom && retries > 0) {
          logEvent({
            level: "error",
            source: "server",
            event: "launch.degrade.exhausted",
            message: `显存不足，降级 ${retries} 次后仍起不来（${basename(this.resolvedModelPath())}）：${attempt.error}`,
            detail: { model: basename(this.resolvedModelPath()), attempts: retries, steps, ...this.launchAdjust },
          });
        }
        this.degraded = null;
        this.lastError = attempt.error;
        // 与加重试之前同一口径：启动阶段正常退出（code 0）记「已停止」，其余记错误
        this.setStatus(attempt.exitCode === 0 ? "stopped" : "error");
        return { ok: false, error: attempt.error };
      }

      // 超时那条路径进程可能还活着（卡在分配上）：先收掉，显存才会还回来
      await this.killStartupLeftover();
      retries += 1;
      used.add(step.kind);
      steps.push(step.summary);
      this.launchAdjust = step.adjust;
      // 在下一次尝试进入 running 之前就记好：注册表在状态变成 running 的回调里读它
      this.degraded = { attempts: retries, adjust: { ...step.adjust }, summary: steps.join("；") };
      const line = `显存不足，第 ${retries} 次降级重试：${step.summary}`;
      this.appendLog(`\n[omni] ${line}\n`);
      logEvent({
        level: "warn",
        source: "server",
        event: "launch.degrade.retry",
        message: `${line}（${basename(this.resolvedModelPath())}）`,
        detail: {
          model: basename(this.resolvedModelPath()),
          attempt: retries,
          step: step.kind,
          reason: attempt.error,
          ...step.adjust,
        },
      });
      if (this.retryDelayMs > 0) await Bun.sleep(this.retryDelayMs);
      if (this.stopRequested) {
        this.degraded = null;
        this.appendLog("\n[omni] 已停止，放弃降级重试\n");
        return { ok: false, error: "Server stopped before becoming ready" };
      }
    }
  }


  /** 启动阶段超时后进程还活着：杀掉整组并等它退出（重试前必须把显存还回来）。 */
  private async killStartupLeftover(): Promise<void> {
    const proc = this.serverProcess;
    if (!proc) return;
    killProcessTree(proc, "SIGTERM");
    if (!(await waitExit(proc, 5000))) {
      killProcessTree(proc, "SIGKILL");
      await proc.exited.catch(() => {});
    }
    if (this.serverProcess === proc) this.serverProcess = null;
  }

  /**
   * 一次启动尝试：拼 argv → spawn → 等健康检查通过 / 进程退出 / 超时 / 用户停止。
   *
   * 失败时**不改状态**（保持 starting）：是报错还是降级重试由 start() 决定 —— 否则每次重试
   * 之间界面会闪一下「错误」，注册表还会各记一条 crashed。进入 running 之后才挂掉的照旧由
   * exited 回调记错误。
   */
  private async launchOnce(
    llamaPath: string,
    model: LlamaModel,
    serverArgs: ServerArgs,
    target: string,
  ): Promise<{ ok: boolean; error: string; exitCode?: number | null; aborted?: boolean }> {
    const args = this.buildArgs(model, serverArgs, target);
    this.launchedArgs = args;
    this.lastError = "";
    this.attemptLog = "";
    this.setStatus("starting");
    this.appendLog(`$ llama-server ${shellJoin(args)}\n`);

    let exited = false;
    let exitCode: number | null = null;
    let signalExit: () => void = () => {};
    const exitSignal = new Promise<void>((resolve) => {
      signalExit = resolve;
    });

    try {
      const usePty = process.platform === "darwin";
      const cmd = usePty
        ? ["script", "-q", "/dev/null", llamaPath, ...args]
        : [llamaPath, ...args];

      // `-hf repo:quant` 由 llama-server 自己拉权重：HF_ENDPOINT / MODEL_ENDPOINT 按下载源路由给
      // （国内镜像或官方直连）。env 不进 argv，复制出来的命令不受影响。
      const proc = spawnServerProcess(cmd, await downloadSourceEnv());
      this.serverProcess = proc;
      this.startupProc = proc;
      pumpServerOutput(proc, this.appendLog.bind(this));

      const self = this;
      proc.exited
        .then((code) => {
          if (self.serverProcess === proc) self.serverProcess = null;
          exited = true;
          exitCode = code;
          signalExit();
          if (self.startupProc === proc && self.getStatus() !== "stopped") {
            // 启动阶段退出：由 start() 的循环决定重试还是报错
            self.appendLog(`\n[server exited with code ${code}]\n`);
            return;
          }
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
          if (self.serverProcess === proc) self.serverProcess = null;
          exited = true;
          signalExit();
          if (self.startupProc !== proc) self.setStatus("error");
        });

      // 与 --port 同源（嵌入实例在嵌入端口段，不能去轮询聊天端口）
      const port = this.resolvePort();
      const healthUrl = `http://localhost:${port}/health`;
      const maxIdleAttempts = 120;
      let idleCount = 0;
      this.lastDownloadActivityAt = 0;

      try {
        while (true) {
          // 进程一退出就不必等满这一拍（OOM 通常几秒内就退，重试要尽快开始）
          await Promise.race([Bun.sleep(this.healthPollMs), exitSignal]);
          if (exited) break;
          const status = this.getStatus();
          if (status !== "starting" && status !== "downloading") break;
          try {
            const res = await fetch(healthUrl, { signal: AbortSignal.timeout(2000) });
            if (res.ok) {
              this.startupProc = null;
              this.setStatus("running");
              this.appendLog("\n[server is ready]\n");
              markServerStarted();
              // 启动后回读实测值（T4e）：任何失败只记日志，不影响上面的成功结果。
              try {
                await this.readbackMeasured();
              } catch (e) {
                logEvent({
                  level: "warn",
                  source: "server",
                  event: "launch_plan.readback_failed",
                  message: `启动回读失败（${e instanceof Error ? e.message : String(e)}），不影响已完成的启动`,
                });
              }
              return { ok: true, error: "" };
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
      } finally {
        this.startupProc = null;
      }

      if (this.getStatus() === "stopped" || this.stopRequested) {
        return {
          ok: false,
          aborted: true,
          error: extractStartupError(this.attemptLog, "Server stopped before becoming ready"),
        };
      }
      if (exited) {
        // 退出后日志管道里可能还有最后几行（根因行往往就在最后），等它落地再挑
        if (this.exitGraceMs > 0) await Bun.sleep(this.exitGraceMs);
        return {
          ok: false,
          exitCode,
          error: extractStartupError(this.attemptLog, `Process exited with code ${exitCode ?? 1}`),
        };
      }
      return {
        ok: false,
        error: extractStartupError(this.attemptLog, "Server failed to become ready within timeout"),
      };
    } catch (e) {
      this.startupProc = null;
      return { ok: false, error: String(e) };
    }
  }

  async stop(): Promise<void> {
    // 启动 / 降级重试进行中：让 start() 的循环就此收手（不再起下一次）
    this.stopRequested = true;
    this.launchAdjust = {};
    this.degraded = null;
    if (!this.serverProcess) {
      this.setStatus("stopped");
      return;
    }

    const proc = this.serverProcess;
    this.setStatus("stopped");
    this.appendLog("\n[stopping server...]\n");

    killProcessTree(proc, "SIGTERM");

    const exited = await waitExit(proc, 5000);

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
    this.stopRequested = true;
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