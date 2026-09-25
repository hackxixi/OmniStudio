/**
 * 本地模型的采样参数解析：按字段逐项决定「用哪个值、为什么是它」。
 *
 * 优先级（与 shared/model-params.ts 一致）：
 *   按模型设置（调用方传入的 override）> 模型自带（GGUF general.sampling.* /
 *   generation_config.json）> 内置家族推荐表（shared/sampling-presets.ts）> 全局设置。
 *
 * 一个例外：显式指定了思考模式（on / off），而家族表恰好为该模式单列了变体时，
 * 变体压过模型自带值。模型自带的只有「默认模式」那一份 —— Qwen3 的 generation_config
 * 写的是思考档 0.6 / 0.95，关掉思考还用它，就违背了模型卡给非思考模式的 0.7 / 0.8。
 *
 * 必须是同步的：llama 的 buildArgs / buildCommandLine 是同步路径。generation_config.json
 * 用同步 fs 读；GGUF 头的读取器（bun/gguf-meta.ts）只有异步版，所以这里维护一份
 * 按路径 + mtime + size 的采样缓存，由 refreshSamplingMetadata 预热；resolveSampling 未命中时
 * 顺手在后台预热一次，本次先当「模型没自带」处理（下一次启动 / 下一条消息就能用上）。
 */
import { readFileSync, statSync } from "fs";
import path from "path";
import { getSetting, type SettingsKey } from "./db/settings";
import { readGgufMeta } from "./gguf-meta";
import { firstSplitShardPath, mainGgufInDir, modelNameForPath } from "./model-scan";
import {
  SAMPLING_FIELDS,
  type ResolvedSampling,
  type SamplingField,
  type SamplingParams,
  type SamplingSource,
  type ThinkingMode,
} from "../shared/model-params";
import { matchSamplingPreset, matchSamplingPresetByArch, presetHasExplicitVariant, presetVariant } from "../shared/sampling-presets";

// ---------- 全局设置（最后一级） ----------

const GLOBAL_KEYS: Record<SamplingField, SettingsKey> = {
  temperature: "SERVER_TEMP",
  topP: "SERVER_TOP_P",
  topK: "SERVER_TOP_K",
  minP: "SERVER_MIN_P",
  presencePenalty: "SERVER_PRESENCE_PENALTY",
  repeatPenalty: "SERVER_REPEAT_PENALTY",
};

/** 设置行被清空 / 写坏时的兜底，与 db/settings.ts 的 DEFAULTS 同值。 */
const GLOBAL_FALLBACK: Required<SamplingParams> = {
  temperature: 0.7,
  topP: 0.9,
  topK: 40,
  minP: 0.05,
  presencePenalty: 0,
  repeatPenalty: 1.0,
};

/** 逐项把关：越界值一律当没写（它们最终会进 argv / 请求体）。 */
function validField(field: SamplingField, v: unknown): number | undefined {
  if (typeof v !== "number" || !Number.isFinite(v)) return undefined;
  switch (field) {
    case "temperature":
      return v >= 0 ? v : undefined;
    case "topP":
      return v > 0 && v <= 1 ? v : undefined;
    case "topK":
      // HF / vLLM 用 -1 表示不截断，llama.cpp 是 0，统一成 0
      if (v === -1) return 0;
      return Number.isInteger(v) && v >= 0 ? v : undefined;
    case "minP":
      return v >= 0 && v <= 1 ? v : undefined;
    case "presencePenalty":
      return v >= -2 && v <= 2 ? v : undefined;
    case "repeatPenalty":
      return v > 0 ? v : undefined;
  }
}

function cleanParams(p: SamplingParams | null | undefined): SamplingParams {
  const out: SamplingParams = {};
  if (!p) return out;
  for (const f of SAMPLING_FIELDS) {
    const v = validField(f, p[f]);
    if (v !== undefined) out[f] = v;
  }
  return out;
}

function globalSampling(): Required<SamplingParams> {
  const out = { ...GLOBAL_FALLBACK };
  for (const f of SAMPLING_FIELDS) {
    const raw = (getSetting(GLOBAL_KEYS[f]) ?? "").trim();
    const v = raw === "" ? undefined : validField(f, Number(raw));
    if (v !== undefined) out[f] = v;
  }
  return out;
}

// ---------- 模型身份 ----------

const PATH_RE = /^(?:[/~.]|[A-Za-z]:[\\/])/;
/** 量化 / 精度后缀（`-Q4_K_M`、`.UD-IQ2_XXS`、`-BF16`、`-MXFP4_MOE`）：展示名里不要。 */
const QUANT_SUFFIX_RE = /[-.](?:UD-)?(?:I?Q\d\w*|[BF]F?16|F32|MXFP4\w*)$/i;

function stripWeightName(name: string): string {
  return name.replace(/\.(gguf|ggml|safetensors|bin)$/i, "").replace(QUANT_SUFFIX_RE, "");
}

/**
 * 用于家族匹配的候选名，按可信度排序：文件 / 目录名 → HF 缓存的 repo 名 → 上一级目录名。
 * HF 缓存的 snapshot 目录名是 commit hash，靠 `models--org--repo` 那一段认；
 * 纯 GGUF 仓库里的文件名有时是 `model-q4.gguf`，靠父目录（`Qwen3-8B-GGUF`）认。
 */
function nameCandidates(target: string): string[] {
  const t = target.trim();
  if (!t) return [];
  if (!PATH_RE.test(t)) {
    // HF 引用：`unsloth/Qwen3-8B-GGUF:Q4_K_M`
    const repo = t.split(":")[0] ?? t;
    return [stripWeightName(repo.split("/").pop() ?? repo), repo];
  }
  const out = [stripWeightName(modelNameForPath(t))];
  const parts = t.split(/[/\\]+/);
  const hfRepo = parts.find((p) => p.startsWith("models--"));
  if (hfRepo) out.push(hfRepo.slice("models--".length).split("--").pop() ?? hfRepo);
  const parent = parts.length >= 2 ? parts[parts.length - 2] : undefined;
  if (parent && parent !== "snapshots") out.push(parent);
  // 应用下载目录用 `org__repo` 命名
  return out.map((n) => n.split("__").pop() ?? n).filter(Boolean);
}

/** 展示用的模型名（去掉扩展名与量化后缀）。 */
export function samplingModelName(target: string): string {
  return nameCandidates(target)[0] ?? target;
}

// ---------- 模型自带：generation_config.json ----------

type GenCacheEntry = { mtimeMs: number; params: SamplingParams };
const genConfigCache = new Map<string, GenCacheEntry>();

function statSafe(p: string): { mtimeMs: number; size: number; isDir: boolean } | null {
  try {
    const s = statSync(p);
    return { mtimeMs: s.mtimeMs, size: s.size, isDir: s.isDirectory() };
  } catch {
    return null;
  }
}

/** HF generation_config.json → SamplingParams（vLLM `--generation-config auto` 同口径，不看 do_sample）。 */
export function parseGenerationConfig(json: unknown): SamplingParams {
  if (!json || typeof json !== "object") return {};
  const g = json as Record<string, unknown>;
  return cleanParams({
    temperature: g.temperature as number,
    topP: g.top_p as number,
    topK: g.top_k as number,
    minP: g.min_p as number,
    presencePenalty: g.presence_penalty as number,
    repeatPenalty: g.repetition_penalty as number,
  });
}

function readGenerationConfig(dir: string): SamplingParams {
  const file = path.join(dir, "generation_config.json");
  const st = statSafe(file);
  if (!st || st.isDir) {
    genConfigCache.delete(file);
    return {};
  }
  const cached = genConfigCache.get(file);
  if (cached && cached.mtimeMs === st.mtimeMs) return cached.params;
  let params: SamplingParams = {};
  try {
    params = parseGenerationConfig(JSON.parse(readFileSync(file, "utf8")));
  } catch {
    // 写坏的 JSON 当没有：采样还有家族表和全局兜底
  }
  genConfigCache.set(file, { mtimeMs: st.mtimeMs, params });
  return params;
}

// ---------- 模型自带：GGUF general.sampling.* ----------

type GgufCacheEntry = {
  mtimeMs: number;
  size: number;
  sampling: SamplingParams;
  name: string | null;
  architecture: string | null;
};
const ggufCache = new Map<string, GgufCacheEntry>();
const ggufInflight = new Map<string, Promise<void>>();

/** target 对应的 GGUF 文件（分片取第一片；目录取主文件）；不是 GGUF 返回 null。 */
function ggufPathFor(target: string, isDir: boolean): string | null {
  if (isDir) return mainGgufInDir(target);
  if (!/\.gguf$/i.test(target)) return null;
  return path.resolve(firstSplitShardPath(target) ?? target);
}

function peekGguf(file: string): GgufCacheEntry | null {
  const st = statSafe(file);
  const cached = ggufCache.get(file);
  if (!st || !cached) return null;
  return cached.mtimeMs === st.mtimeMs && cached.size === st.size ? cached : null;
}

/**
 * 预热 GGUF 采样缓存（异步读文件头；读取器自带 mtime+size 缓存，重复调用很便宜）。
 * 启动模型前 await 它一次，同步的 resolveSampling 就能拿到文件里写的推荐值。
 */
export async function refreshSamplingMetadata(target: string): Promise<void> {
  const st = statSafe(target);
  if (!st) return;
  const file = ggufPathFor(target, st.isDir);
  if (!file) return;
  const running = ggufInflight.get(file);
  if (running) return running;
  const job = (async () => {
    const res = await readGgufMeta(file);
    if (!res.ok) return;
    ggufCache.set(file, {
      mtimeMs: res.data.mtimeMs,
      size: res.data.size,
      sampling: cleanParams(res.data.meta.sampling),
      name: res.data.meta.name,
      architecture: res.data.meta.architecture,
    });
  })().finally(() => ggufInflight.delete(file));
  ggufInflight.set(file, job);
  return job;
}

/** 测试用：清掉两份缓存。 */
export function clearSamplingCache(): void {
  genConfigCache.clear();
  ggufCache.clear();
}

type ModelMetadata = { sampling: SamplingParams; ggufName: string | null; ggufArch: string | null };

function modelMetadata(target: string): ModelMetadata {
  const st = statSafe(target);
  if (!st) return { sampling: {}, ggufName: null, ggufArch: null };
  const gen = readGenerationConfig(st.isDir ? target : path.dirname(target));
  const file = ggufPathFor(target, st.isDir);
  let gguf: GgufCacheEntry | null = null;
  if (file) {
    gguf = peekGguf(file);
    // 未预热：后台读一次，本次先按「没自带」算（失败由读取器记日志，这里不再抛）
    if (!gguf) void refreshSamplingMetadata(target).catch(() => {});
  }
  // 两者都有时文件头优先：它跟着这一个量化文件走，比同目录的 config 更具体
  return { sampling: { ...gen, ...(gguf?.sampling ?? {}) }, ggufName: gguf?.name ?? null, ggufArch: gguf?.architecture ?? null };
}

// ---------- 解析 ----------

export function resolveSampling(
  target: string,
  opts: { thinking?: ThinkingMode; override?: SamplingParams } = {},
): ResolvedSampling {
  const mode: ThinkingMode = opts.thinking ?? "auto";
  const override = cleanParams(opts.override);

  const meta = modelMetadata(target);
  let match = null as ReturnType<typeof matchSamplingPreset>;
  for (const name of nameCandidates(target)) {
    match = matchSamplingPreset(name);
    if (match) break;
  }
  // 文件名什么都看不出来（`model-q4.gguf`）时，再拿 GGUF 里的 general.name 认一次
  if (!match && meta.ggufName) match = matchSamplingPreset(meta.ggufName);
  // 名字都认不出（微调改了名）：按架构认底座家族
  if (!match) match = matchSamplingPresetByArch(meta.ggufArch);

  const variant = match ? cleanParams(presetVariant(match.preset, mode)) : {};
  const variantBeatsMeta = match ? presetHasExplicitVariant(match.preset, mode) : false;
  const global = globalSampling();

  const values = {} as Required<SamplingParams>;
  const sources = {} as Record<SamplingField, SamplingSource>;
  for (const f of SAMPLING_FIELDS) {
    const pick = (v: number | undefined, source: SamplingSource): boolean => {
      if (v === undefined) return false;
      values[f] = v;
      sources[f] = source;
      return true;
    };
    if (pick(override[f], "model-override")) continue;
    if (variantBeatsMeta && pick(variant[f], "family-preset")) continue;
    if (pick(meta.sampling[f], "model-metadata")) continue;
    if (pick(variant[f], "family-preset")) continue;
    pick(global[f], "global");
  }
  return { values, sources, preset: match?.id ?? null };
}
