/**
 * llama.cpp 的 flash attention 开关（`--flash-attn`）、模型加载模式（`--load-mode`）与
 * `--kv-unified` 的版本探测：它们都只能从 `--help` 的输出里认出来，所以**合并成一次
 * 子进程调用**（`probeServerHelp`），而不是为了每个开关各跑一遍。
 *
 * 为什么要探测而不是赌版本：
 *  - 新版（本机 llama.cpp 实测）是**三态**：`-fa, --flash-attn [on|off|auto]`
 *    （env: LLAMA_ARG_FLASH_ATTN，default auto）；
 *  - 老一些的 build 里 `--flash-attn` 是**不带值的布尔开关**（只认 on）；
 *  - 更老的没有这个开关。
 *
 * `auto` 是 llama.cpp 自己的默认（N 卡上自动开），也是本应用的默认 —— 所以「探测不到
 * 三态」时宁可一个参数都不发，行为与加这个开关之前逐字节一致；「探测到三态」才按用户
 * 选择发 `--flash-attn <auto|on|off>`。
 *
 * 缓存规则与 llama.ts 里 load-mode 那套一致：按二进制路径进程级缓存（托管安装与
 * brew 那份版本可能不同，各自探各自的）；探测**失败**不落缓存（下回再试），探测**成功**
 * （哪怕是 "none"）落缓存 —— "none" 是真实答案，重试没有意义。
 */

import { existsSync } from "fs";

import { llamaCppBinaryPath } from "../engine-paths";

/** 托管安装之外、按固定路径查找的 llama-server（Homebrew 两种前缀）。 */
export const COMMON_LLAMA_SERVER_PATHS = [
  "/opt/homebrew/bin/llama-server",
  "/usr/local/bin/llama-server",
];

/**
 * 同步路径（`buildCommandLine` / 计划 key / 预览 RPC）读探测缓存时用的二进制路径：
 * 托管安装 → Homebrew → 裸名。缓存按路径分，所以**所有同步读者必须用同一条规则**，
 * 否则「复制的命令」与预览、实际启动各读各的缓存条目、互相对不上。
 */
export function defaultLlamaServerBinary(): string {
  return [llamaCppBinaryPath(), ...COMMON_LLAMA_SERVER_PATHS].find((p) => existsSync(p)) ?? "llama-server";
}

/** 真值表（llama-server --help 实测行）：
 *   `-fa, --flash-attn [on|off|auto]    set Flash Attention use ('on', 'off', or 'auto', default: 'auto')`
 */

export type FlashAttnSupport = "tristate" | "boolean" | "none";

export type ServerHelpSupport = {
  loadMode: "load-mode" | "legacy" | "unknown";
  flashAttn: FlashAttnSupport;
  /**
   * 是否认 `--kv-unified`（`-kvu`）。不认的老版在 `--parallel N` 下会把 `--ctx-size`
   * 按 slot 均分，每个请求只剩 1/N 的窗口 —— 规划器据此决定按哪种口径计价。
   * 可选只是为了兼容旧调用方（测试注缓存）；缺省按不支持处理（不赌开关存在）。
   */
  kvUnified?: boolean;
  /**
   * 是否认 `-rea, --reasoning [on|off|auto]`（按模型「思考开关」直接交给引擎）。
   * 不认的老版回落 `--chat-template-kwargs '{"enable_thinking":false}'`（只有关能这样表达）。
   * 可选同 kvUnified：缺省按不支持处理。
   */
  reasoning?: boolean;
};

/** `--load-mode` 的 `--help` 识别（照抄 llama-load-mode.ts 的正则，保持两处判定一致）。 */
function parseLoadModeSupportText(help: string): ServerHelpSupport["loadMode"] {
  if (/--load-mode\b/.test(help) || /(^|\s)-lm\b/.test(help)) return "load-mode";
  if (/--mlock\b/.test(help) || /--no-mmap\b/.test(help)) return "legacy";
  return "unknown";
}

/** `--flash-attn` 的 `--help` 识别：三态看取值列表，布尔只看开关名。 */
function parseFlashAttnSupportText(help: string): FlashAttnSupport {
  const lines = help.split("\n");
  for (const line of lines) {
    if (line.includes("--flash-attn") || /(^|\s)-fa\b/.test(line)) {
      if (/[oO]n\|\s*[oO]ff\|\s*[aA]uto/.test(line) || /\[\s*[oO]n/.test(line)) return "tristate";
      return "boolean";
    }
  }
  return "none";
}

/**
 * `--kv-unified` 的 `--help` 识别：`-kvu, --kv-unified` 那一行。
 * 注意 `--no-kv-unified`（反向开关）不能算数 —— 它前面是 `no-` 而不是空白 / 逗号，
 * 所以锚在「行首 / 空白 / 逗号」之后。
 */
function parseKvUnifiedSupportText(help: string): boolean {
  return /(^|[\s,])--kv-unified\b/m.test(help) || /(^|[\s,])-kvu\b/m.test(help);
}

/**
 * `--reasoning` 的 `--help` 识别：`-rea, --reasoning [on|off|auto]` 那一行。
 * 同名前缀的 `--reasoning-format` / `--reasoning-budget` / `--reasoning-effort` 是另外的开关
 * （老版早就有 --reasoning-format），不能算数 —— 所以要求后面不再跟 `-` / 单词字符。
 */
function parseReasoningSupportText(help: string): boolean {
  return /(^|[\s,])--reasoning(?![-\w])/m.test(help);
}

/** 一次 `--help` 解析出各开关的支持形态。 */
export function parseServerHelpSupport(help: string): ServerHelpSupport {
  return {
    loadMode: parseLoadModeSupportText(help),
    flashAttn: parseFlashAttnSupportText(help),
    kvUnified: parseKvUnifiedSupportText(help),
    reasoning: parseReasoningSupportText(help),
  };
}

/**
 * `--help` 输出 → 启动参数。
 *
 *  - `tristate`：把用户的三态选择原样发出去（`auto` 与不传等价，但显式发出去无害，
 *    且能让 llama-server 的启动日志把实际取值打印出来 —— 这是 effective 状态的来源）；
 *  - `boolean`：老版只认「开」，所以只有 `"on"` 才发（不带值）；`"auto"`/`"off"` 都不发
 *    （老版没有「关」这个选择，auto 听它自己的默认，行为与加开关前一致）；
 *  - `none`：一个参数都不发。
 */
/** 合法取值白名单：拼命令行的地方自己校验，不依赖上游 updateSettings 的枚举检查。 */
const FLASH_ATTN_VALUES = new Set(["auto", "on", "off"]);

export function flashAttnArgs(setting: string | null | undefined, support: FlashAttnSupport): string[] {
  const value = setting != null && FLASH_ATTN_VALUES.has(setting) ? setting : "auto";
  if (support === "tristate") return ["--flash-attn", value];
  if (support === "boolean") return value === "on" ? ["--flash-attn"] : [];
  return [];
}

/**
 * 按模型的思考开关 → 启动参数（仅聊天实例）。
 *
 *  - auto / 没设：一个参数都不发（交给模板默认，与加这个开关前逐字节一致）；
 *  - 认 `--reasoning`：原样发 `--reasoning on|off`；
 *  - 不认（老版 / 没探过）：关 → `--chat-template-kwargs {"enable_thinking":false}`（Qwen3 /
 *    GLM 等模板都认这个 kwarg，老版也有这个开关）；开 → 不发（模板默认本来就是开）。
 */
export function reasoningArgs(mode: string | null | undefined, supported: boolean): string[] {
  if (mode !== "on" && mode !== "off") return [];
  if (supported) return ["--reasoning", mode];
  return mode === "off" ? ["--chat-template-kwargs", JSON.stringify({ enable_thinking: false })] : [];
}

// —— 进程级缓存（llama.ts 持有） ——

const flashAttnSupportCache = new Map<string, FlashAttnSupport>();
const loadModeSupportCache = new Map<string, ServerHelpSupport["loadMode"]>();
const kvUnifiedSupportCache = new Map<string, boolean>();
const reasoningSupportCache = new Map<string, boolean>();

export function cachedFlashAttnSupport(binaryPath: string): FlashAttnSupport | null {
  return flashAttnSupportCache.get(binaryPath) ?? null;
}

export function cachedLoadModeSupport(binaryPath: string): ServerHelpSupport["loadMode"] | null {
  return loadModeSupportCache.get(binaryPath) ?? null;
}

/** 同步读已探测到的 `--kv-unified` 支持（null = 还没探过）。 */
export function cachedKvUnifiedSupport(binaryPath: string): boolean | null {
  return kvUnifiedSupportCache.get(binaryPath) ?? null;
}

/** 同步读已探测到的 `--reasoning` 支持（null = 还没探过）。 */
export function cachedReasoningSupport(binaryPath: string): boolean | null {
  return reasoningSupportCache.get(binaryPath) ?? null;
}

/**
 * 同步读已缓存的合并探测结果（null = 还没探过）。
 * llama.ts 的 `cachedLoadModeSupport` 委托到这里，避免两份 Map 各自演化。
 */
export function cachedServerHelpSupport(
  binaryPath: string,
): {
  loadMode: ServerHelpSupport["loadMode"];
  flashAttn: FlashAttnSupport;
  kvUnified: boolean;
  reasoning: boolean;
} | null {
  const load = loadModeSupportCache.get(binaryPath);
  const flash = flashAttnSupportCache.get(binaryPath);
  const kvu = kvUnifiedSupportCache.get(binaryPath);
  const rea = reasoningSupportCache.get(binaryPath);
  if (load === undefined && flash === undefined && kvu === undefined && rea === undefined) return null;
  return {
    loadMode: load ?? "unknown",
    flashAttn: flash ?? "none",
    kvUnified: kvu ?? false,
    reasoning: rea ?? false,
  };
}

export function clearServerHelpSupportCache(): void {
  flashAttnSupportCache.clear();
  loadModeSupportCache.clear();
  kvUnifiedSupportCache.clear();
  reasoningSupportCache.clear();
}

/**
 * 同步把一份探测结果写进缓存（仅供测试注用；生产路径永远走 probeServerHelp）。
 * loadMode === "unknown" 不落缓存（与 probeServerHelp 同一规则），flashAttn 三种都落；
 * kvUnified 缺省不落（等价于「没探过」→ 不支持）。
 */
export function setCachedServerHelpSupport(binaryPath: string, support: ServerHelpSupport): void {
  if (support.loadMode !== "unknown") loadModeSupportCache.set(binaryPath, support.loadMode);
  flashAttnSupportCache.set(binaryPath, support.flashAttn);
  if (support.kvUnified !== undefined) kvUnifiedSupportCache.set(binaryPath, support.kvUnified);
  if (support.reasoning !== undefined) reasoningSupportCache.set(binaryPath, support.reasoning);
}

/**
 * 一次 `llama-server --help`，解析出 load-mode 与 flash-attn 的支持形态并按路径缓存
 * （成功才落缓存；子进程失败 / 超时不落 —— 下回启动再试）。超时 2s 后 kill（照 load-mode
 * 原实现的保护）。
 *
 * 未探测过（或探测失败）时：load-mode 回落到各自的已有缓存 / "unknown"（与旧实现
 * 「探测失败记 unknown」一致）；flash-attn 回落到已有缓存 / "none"（没探过就不赌开关
 * 存在，一个参数都不发）。
 */
export async function probeServerHelp(binaryPath: string): Promise<ServerHelpSupport> {
  const cachedFlash = flashAttnSupportCache.get(binaryPath);
  const cachedLoad = loadModeSupportCache.get(binaryPath);
  const cachedKvu = kvUnifiedSupportCache.get(binaryPath);
  const cachedRea = reasoningSupportCache.get(binaryPath);
  if (cachedFlash && cachedLoad && cachedKvu !== undefined && cachedRea !== undefined) {
    return { loadMode: cachedLoad, flashAttn: cachedFlash, kvUnified: cachedKvu, reasoning: cachedRea };
  }

  let help = "";
  let ok = false;
  try {
    const proc = Bun.spawn({ cmd: [binaryPath, "--help"], stdout: "pipe", stderr: "pipe" });
    const timer = setTimeout(() => {
      try {
        proc.kill();
      } catch {
        // 已经退出了
      }
    }, 2_000);
    try {
      const [out, err] = await Promise.all([
        new Response(proc.stdout).text().catch(() => ""),
        new Response(proc.stderr).text().catch(() => ""),
      ]);
      help = `${out}\n${err}`;
      ok = true;
    } finally {
      clearTimeout(timer);
    }
  } catch {
    ok = false;
  }

  if (!ok) {
    // 探测失败：不落缓存，同步路径按「还没探过」处理。
    return {
      loadMode: cachedLoad ?? "unknown",
      flashAttn: cachedFlash ?? "none",
      kvUnified: cachedKvu ?? false,
      reasoning: cachedRea ?? false,
    };
  }

  const parsed = parseServerHelpSupport(help);
  if (parsed.loadMode !== "unknown") loadModeSupportCache.set(binaryPath, parsed.loadMode);
  flashAttnSupportCache.set(binaryPath, parsed.flashAttn);
  kvUnifiedSupportCache.set(binaryPath, parsed.kvUnified ?? false);
  reasoningSupportCache.set(binaryPath, parsed.reasoning ?? false);
  return parsed;
}
