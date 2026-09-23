/**
 * llama.cpp 启动计划的主进程侧缓存层（Part B）。
 *
 * 主进程在两处消费同一份计划，而这两处的同步性不一样：
 *   - `runtime.start()` 是 async，可以现算（`refreshLaunchPlan`）；
 *   - `buildCommandLine()` 是**同步的**（UI「复制启动命令」要立刻出字），只能读缓存
 *     （`cachedLaunchPlan`）。
 *
 * 照 `runtimes/llama.ts` 里 `probeLoadModeSupport` / `cachedLoadModeSupport` 的套路：
 * 异步算一次、按 key 缓存、同步读。启动前先 `refreshLaunchPlan`，之后 UI 复制到的命令
 * 就是实际发出去的那条；没算过 / key 变了读缓存返回 null，调用方回落到设置里的值。
 *
 * 缓存 key **只由 `LaunchPlanKey` 决定**（`planCacheKey`，全模块唯一入口），
 * GGUF 文件的 mtimeMs/size 指纹存在**条目的值**里，`cachedLaunchPlan` 命中时重新 stat
 * 校验（文件被换掉但名字不变时 mtime 兜住；stat 失败不丢弃计划——文件可能在网络盘上）。
 * 只保留最近 4 条，Map 的插入顺序天然就是 LRU（命中时移到队尾）。
 */
import { lstatSync, readdirSync, statSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { getSetting, type SettingsKey } from "./db/settings";
import { logEvent } from "./app-log";
import { readGgufMeta } from "./gguf-meta";
import { isMmprojFile } from "./model-scan";
import { cachedKvUnifiedSupport, defaultLlamaServerBinary } from "./runtimes/llama-flash-attn";
import { MIN_FIT_CTX } from "../shared/launch-planner";
import { plannerHardwareSnapshot } from "./planner-hardware";
import { planLlamaLaunch, type LaunchPlan } from "../shared/launch-planner";

export type { LaunchPlan };

export type LaunchPlanKey = {
  modelPath: string;
  parallel: number;
  ubatch: number | null;
  batch: number | null;
  cacheTypeK: string | null;
  cacheTypeV: string | null;
  /** 用户显式指定的窗口，null = 自动。 */
  ctxOverride: number | null;
  flashAttn: boolean | null;
  /**
   * 这台 llama-server 认不认 `--kv-unified`（来自 `--help` 探测缓存；没探过 = false）。
   * 不认时 parallel > 1 会把总窗口按 slot 均分，规划器要按「每 slot 窗口」计价 ——
   * 不进 key 的话，同一模型在支持 / 不支持两种二进制下会命中同一份计划。
   */
  supportsKvUnified: boolean;
  /** 同目录自动配对的 mmproj 文件字节数（没有 = null）：它也要占显存，预算要扣掉。 */
  mmprojBytes: number | null;
};

const CACHE_MAX_ENTRIES = 4;

type CacheEntry = {
  /** 写入缓存时模型文件的指纹（`<mtimeMs>:<size>`），`cachedLaunchPlan` 命中时重新 stat 比对 */
  mtimeMs: number;
  size: number;
  plan: LaunchPlan;
};

const cache = new Map<string, CacheEntry>();

/** 缓存 key 的唯一计算入口：只由 `LaunchPlanKey` 决定，不含文件指纹。 */
function planCacheKey(key: LaunchPlanKey): string {
  return JSON.stringify({
    p: key.modelPath,
    par: key.parallel,
    ub: key.ubatch,
    b: key.batch,
    ck: key.cacheTypeK,
    cv: key.cacheTypeV,
    ctx: key.ctxOverride,
    fa: key.flashAttn,
    kvu: key.supportsKvUnified,
    mm: key.mmprojBytes,
  });
}

function statModelFile(
  modelPath: string,
): { mtimeMs: number; size: number } | null {
  try {
    // lstat 而不是 stat：符号链接指向的源文件（比如引擎目录的 `current`）被替换时
    // 链接自身的 mtime 会变，能兜住「模型被升级但 GGUF 文件名不变」的情况。
    const st = lstatSync(resolve(modelPath));
    return { mtimeMs: st.mtimeMs, size: st.size };
  } catch {
    return null;
  }
}

/**
 * 异步算一份计划并写进缓存。启动前调用。
 * 读不到 GGUF 元数据（不是 GGUF / 文件不存在）→ 返回 null（调用方回落到设置里的值）。
 */
export async function refreshLaunchPlan(key: LaunchPlanKey): Promise<LaunchPlan | null> {
  const read = await readGgufMeta(key.modelPath);
  if (!read.ok) {
    logEvent({
      level: "info",
      source: "server",
      event: "launch_plan.skipped",
      message: `启动计划跳过：${basename(key.modelPath)}（${read.reason}）`,
      detail: { reason: read.reason, file: basename(key.modelPath) },
    });
    return null;
  }

  const hardware = await plannerHardwareSnapshot();
  const minCtx = autoTuneMinCtx();
  const plan = planLlamaLaunch({
    meta: read.data.meta,
    weightsBytes: read.data.totalFileBytes,
    mmprojBytes: key.mmprojBytes,
    supportsKvUnified: key.supportsKvUnified,
    hardware,
    minCtx,
    overrides: {
      ctxTokens: key.ctxOverride,
      parallel: key.parallel,
      batch: key.batch,
      ubatch: key.ubatch,
      cacheTypeK: key.cacheTypeK,
      cacheTypeV: key.cacheTypeV,
      flashAttn: key.flashAttn,
    },
  });

  // 这条日志是「预测 vs 实测」校准的依据：ctxTokens / budgetBytes / kvBytes 与
  // 启动日志里的实际占用对照，才能看出估算系数偏乐观还是偏保守。
  logEvent({
    level: "info",
    source: "server",
    event: "launch_plan.computed",
    message: `启动计划：${basename(read.data.filePath)} ctx=${plan.ctxTokens} fits=${plan.fits}`,
    detail: {
      file: basename(read.data.filePath),
      ctxTokens: plan.ctxTokens,
      ctxPerSlot: plan.ctxPerSlot,
      fits: plan.fits,
      budgetBytes: plan.estimates.budgetBytes,
      kvBytes: plan.estimates.kvBytes,
      totalBytes: plan.estimates.totalBytes,
      reasons: plan.reasons.map((r) => r.code),
    },
  });

  const mapKey = planCacheKey(key);
  const entry: CacheEntry = {
    // 指纹来自 GGUF 头读取（`filePath` 可能是分片的第一片，与 modelPath 相同时就是它自己）
    mtimeMs: read.data.mtimeMs,
    size: read.data.size,
    plan,
  };
  // LRU：命中时先删后插，把这条移到队尾。
  cache.delete(mapKey);
  cache.set(mapKey, entry);
  while (cache.size > CACHE_MAX_ENTRIES) {
    const oldest = cache.keys().next().value;
    if (oldest === undefined) break;
    cache.delete(oldest);
  }
  return plan;
}

/**
 * 同步读缓存。没算过、或 key 变了就返回 null —— 调用方回落到设置里的值。
 * 命中后再同步 stat 一次模型文件：文件被换过（mtime/size 变了）删掉这条返回 null；
 * stat 失败（网络盘抖动 / 文件临时不可见）不丢弃计划，照旧返回。
 */
export function cachedLaunchPlan(key: LaunchPlanKey): LaunchPlan | null {
  const mapKey = planCacheKey(key);
  const entry = cache.get(mapKey);
  if (entry === undefined) return null;
  const st = statModelFile(key.modelPath);
  if (st !== null && (st.mtimeMs !== entry.mtimeMs || st.size !== entry.size)) {
    cache.delete(mapKey);
    return null;
  }
  // 命中：移到队尾
  cache.delete(mapKey);
  cache.set(mapKey, entry);
  return entry.plan;
}

/**
 * 自动推算时上下文的下限（token）：设置 `SERVER_AUTO_TUNE_MIN_CTX`，解析不了 / 非正数
 * 回落规划器的 MIN_FIT_CTX（4096）。只影响自动拟合（见 llama.ts），不影响任何设置值。
 */
function autoTuneMinCtx(): number {
  const raw = Number(getSetting("SERVER_AUTO_TUNE_MIN_CTX" as SettingsKey));
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : MIN_FIT_CTX;
}

export function clearLaunchPlanCache(): void {
  cache.clear();
}

/**
 * 本地 GGUF 模型同目录下自动配对的多模态投影文件（mmproj-*.gguf），没有就 null。
 *
 * 聊天实例与嵌入实例共用这一条规则（llama.ts 拼 `--mmproj`、这里给规划器算字节数），
 * 保证「命令里带的那个文件」就是「预算里扣掉的那个文件」。
 * 多文件优先 f16：bf16 的名字里也含 "f16" 子串，直接 includes 会选错，所以要求 f16
 * 前面不是字母；没有 f16 时按字典序取第一个兜底。
 * `modelPath` 通常是主 GGUF 文件（目录条目已被 llamaLoadablePath 换成主文件），
 * 投影文件在它的 dirname 里；万一传进来的就是目录（预览入口），直接在目录里找。
 * 目录读不到（模型被移走等）按「无投影文件」处理。
 */
export function pairedMmprojPath(modelPath: string): string | null {
  let dir: string;
  try {
    dir = statSync(modelPath).isDirectory() ? modelPath : dirname(modelPath);
  } catch {
    dir = dirname(modelPath);
  }
  try {
    const names = readdirSync(dir);
    // 平铺目录防误配：只有目录里的主 GGUF 都是「同一个模型的不同量化 / 分片」时才配对。
    // 用户把几个不相干的模型平铺在一个文件夹里时，别人的 mmproj 会让纯文本模型直接加载失败
    // （投影维度对不上），宁可不配 —— 看不了图比起不来强。
    if (!singleModelFamily(names)) return null;
    const candidates = names.filter(isMmprojFile).sort();
    const f16 = candidates.find((n) => /(?:^|[^a-z])f16/i.test(n));
    const picked = f16 ?? candidates[0];
    return picked ? join(dir, picked) : null;
  } catch {
    return null;
  }
}

/** 量化 / 精度 / 分片后缀（Qwen3-VL-8B-Instruct-UD-Q4_K_XL-00001-of-00002 → Qwen3-VL-8B-Instruct）。 */
const QUANT_SUFFIX_RE =
  /(?:[-._](?:\d{5}-of-\d{5}))?(?:[-._](?:UD[-_])?(?:I?Q\d[\w]*|B?F16|F32|MXFP4\w*))+(?:[-._]\d{5}-of-\d{5})?$/i;

/** 目录里除 mmproj 外的 GGUF 是否同属一个模型（去掉量化后缀后名字一致）；一个都没有也算。 */
export function singleModelFamily(names: string[]): boolean {
  const stems = new Set(
    names
      .filter((n) => /\.gguf$/i.test(n) && !isMmprojFile(n))
      .map((n) => n.replace(/\.gguf$/i, "").replace(QUANT_SUFFIX_RE, "").toLowerCase()),
  );
  return stems.size <= 1;
}

/** 配对到的 mmproj 字节数（没有 / stat 失败 = null），规划器的 `mmprojBytes` 输入。 */
function pairedMmprojBytes(modelPath: string): number | null {
  const mmproj = pairedMmprojPath(modelPath);
  if (mmproj === null) return null;
  try {
    return statSync(mmproj).size;
  } catch {
    return null;
  }
}

/**
 * 自动启动参数缓存 key 的唯一计算入口（llama.ts 与测试共用，从根上消灭两份 key
 * 构造规则各自演化导致不一致的这类 bug）。字段语义见 `LaunchPlanKey` 注释。
 *
 * `flashAttn` 参数是「用户设置 + 上次实测」折算后的布尔（llama.ts 的
 * `effectiveFlashAttnForPlan`），null = 调用方不知道（默认按关，保守）。
 *
 * `supportsKvUnified` / `mmprojBytes` 不靠调用方传：前者读 `--help` 探测缓存（按
 * `defaultLlamaServerBinary` 那条同步路径规则，与 buildCommandLine 同源；没探过 = 不支持，
 * 与「没探过就不发 --kv-unified」一致），后者按 `pairedMmprojPath` 现 stat。这样预览 RPC
 * 与运行时不用各自补参数也拿到逐字段相同的 key。运行时在 start() 里刚探测过时可以用
 * `opts.supportsKvUnified` 传当次探测值（探测成功也会写进同一份缓存，两者一致）。
 */
export function buildLaunchPlanKeyFromSettings(
  modelPath: string,
  get: (key: string) => string,
  flashAttn?: boolean | null,
  opts?: { supportsKvUnified?: boolean | null },
): LaunchPlanKey {
  const parse = (raw: string): number | null => {
    if (!raw.trim()) return null;
    const n = Number(raw);
    return Number.isFinite(n) && n >= 0 ? Math.floor(n) : null;
  };
  return {
    modelPath,
    parallel: parse(get("SERVER_PARALLEL")) ?? 1,
    ubatch: parse(get("SERVER_UBATCH_SIZE")),
    batch: parse(get("SERVER_BATCH_SIZE")),
    cacheTypeK: get("SERVER_CACHE_TYPE_K") || null,
    cacheTypeV: get("SERVER_CACHE_TYPE_V") || null,
    ctxOverride: null,
    flashAttn: flashAttn ?? false,
    supportsKvUnified:
      opts?.supportsKvUnified ?? cachedKvUnifiedSupport(defaultLlamaServerBinary()) ?? false,
    mmprojBytes: pairedMmprojBytes(modelPath),
  };
}

/**
 * 仅供测试：把一份现成的计划直接写进缓存，走的与 `refreshLaunchPlan` 完全相同的
 * key / 指纹路径（指纹缺省取当前文件的真实 stat；文件不存在存 0/0，命中时的
 * 「stat 失败不丢弃」规则会让它照常命中）。生产路径永远走 `refreshLaunchPlan`，
 * 这是测试唯一注入口。
 */
export function __setLaunchPlanForTest(
  key: LaunchPlanKey,
  plan: LaunchPlan,
  stat?: { mtimeMs: number; size: number },
): void {
  const mapKey = planCacheKey(key);
  const fingerprint = stat ?? statModelFile(key.modelPath) ?? { mtimeMs: 0, size: 0 };
  cache.delete(mapKey);
  cache.set(mapKey, { mtimeMs: fingerprint.mtimeMs, size: fingerprint.size, plan });
}
