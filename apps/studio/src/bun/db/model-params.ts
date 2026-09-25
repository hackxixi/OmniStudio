/**
 * 按模型保存的参数（model_params 表）的读写入口。
 *
 * 形状与优先级见 shared/model-params.ts。这里只管两件事：
 *  1. 校验收敛：这些值最终会原样进引擎 argv，所以写入时逐字段按白名单 / 范围收一遍，
 *     非法字段直接丢弃（不钳位 —— 钳出来的值不是用户填的，界面上会对不上）；
 *     读出时再收一遍（库被手改 / 旧版本写入的也挡住）。
 *  2. 同步读缓存：buildArgs / buildCommandLine 是同步的，而且一次启动会读好几次同一个 key，
 *     按 key 进程级缓存（含「没有」这个答案），写入 / 清除时失效。
 */
import { eq } from "drizzle-orm";
import { existsSync } from "fs";
import { resolveRuntimeTarget } from "../model-scan";
import { db } from "./index";
import { modelParams as modelParamsTable } from "./schema";
import {
  SAMPLING_FIELDS,
  type ModelParams,
  type SamplingField,
  type SamplingParams,
} from "../../shared/model-params";

/**
 * llama.cpp `--cache-type-k/-v` 认的取值（本机 llama-server --help 实测的 allowed values），
 * 与规划器的 KV 字节表一致。buildArgs 的 argv 白名单也用这一份。
 */
export const KV_CACHE_TYPES = [
  "f32",
  "f16",
  "bf16",
  "q8_0",
  "q4_0",
  "q4_1",
  "iq4_nl",
  "q5_0",
  "q5_1",
] as const;

const KV_CACHE_TYPE_SET = new Set<string>(KV_CACHE_TYPES);
const FLASH_ATTN_VALUES = new Set(["auto", "on", "off"]);
const THINKING_VALUES = new Set(["auto", "on", "off"]);

/** 追加参数的长度上限：再长就不是「几个开关」而是粘错了东西。 */
const EXTRA_ARGS_MAX_CHARS = 4096;

type Range = { min: number; max: number; int?: boolean };

const LAUNCH_RANGES = {
  ctxSize: { min: 256, max: 1_048_576, int: true },
  parallel: { min: 1, max: 64, int: true },
  gpuLayers: { min: -1, max: 999, int: true },
} satisfies Record<string, Range>;

const SAMPLING_RANGES: Record<SamplingField, Range> = {
  temperature: { min: 0, max: 5 },
  topP: { min: 0, max: 1 },
  topK: { min: 0, max: 1000, int: true },
  minP: { min: 0, max: 1 },
  presencePenalty: { min: -2, max: 2 },
  repeatPenalty: { min: 0.5, max: 2 },
};

function inRange(v: unknown, r: Range): number | undefined {
  // 界面表单可能送字符串过来（"0.6"），能干净解析成数字的也收；空串 / 非数一律丢。
  const n = typeof v === "string" && v.trim() !== "" ? Number(v) : v;
  if (typeof n !== "number" || !Number.isFinite(n)) return undefined;
  if (r.int && !Number.isInteger(n)) return undefined;
  if (n < r.min || n > r.max) return undefined;
  return n;
}

/**
 * 校验收敛一份（可能来自界面 / 手改库的）参数：白名单之外、越界、类型不对的字段一律丢弃，
 * 只留下能安全进 argv 的。返回的对象没有 undefined 字段；全被丢光时返回 `{}`。
 */
export function sanitizeModelParams(input: unknown): ModelParams {
  if (input === null || typeof input !== "object" || Array.isArray(input)) return {};
  const raw = input as Record<string, unknown>;
  const out: ModelParams = {};

  for (const [field, range] of Object.entries(LAUNCH_RANGES) as [keyof typeof LAUNCH_RANGES, Range][]) {
    const v = inRange(raw[field], range);
    if (v !== undefined) out[field] = v;
  }
  for (const field of ["cacheTypeK", "cacheTypeV"] as const) {
    const v = raw[field];
    if (typeof v === "string" && KV_CACHE_TYPE_SET.has(v)) out[field] = v;
  }
  if (typeof raw.flashAttn === "string" && FLASH_ATTN_VALUES.has(raw.flashAttn)) {
    out.flashAttn = raw.flashAttn as ModelParams["flashAttn"];
  }
  if (typeof raw.thinking === "string" && THINKING_VALUES.has(raw.thinking)) {
    out.thinking = raw.thinking as ModelParams["thinking"];
  }

  if (raw.sampling !== null && typeof raw.sampling === "object" && !Array.isArray(raw.sampling)) {
    const s = raw.sampling as Record<string, unknown>;
    const sampling: SamplingParams = {};
    for (const field of SAMPLING_FIELDS) {
      const v = inRange(s[field], SAMPLING_RANGES[field]);
      if (v !== undefined) sampling[field] = v;
    }
    if (Object.keys(sampling).length > 0) out.sampling = sampling;
  }

  if (typeof raw.extraArgs === "string") {
    // 换行在 shell 里是命令分隔，这里一律当空白（tokenizer 本来也按空白切）；控制字符丢掉。
    const v = raw.extraArgs.replace(/[\r\n\t]+/g, " ").replace(/[\x00-\x1f\x7f]/g, "").trim();
    if (v && v.length <= EXTRA_ARGS_MAX_CHARS) out.extraArgs = v;
  }

  return out;
}

/** 按 key 的同步读缓存（null = 查过，没有这一行）。 */
const cache = new Map<string, ModelParams | null>();

/**
 * 模型身份归一：与已服务模型注册表（model-servers.startServedModel）同一规则 —— 本地路径过一遍
 * resolveRuntimeTarget（分批 GGUF → 第一片、仓库里的权重文件 → 仓库目录），HF repo id 原样。
 * 界面拿扫描到的路径来存、运行时拿注册表的 target 来读，两边落到同一个 key。
 */
export function modelParamsKey(raw: string): string {
  const t = typeof raw === "string" ? raw.trim() : "";
  if (!t) return "";
  try {
    return existsSync(t) ? resolveRuntimeTarget(t) : t;
  } catch {
    return t;
  }
}

function normalizeKey(key: string): string {
  return modelParamsKey(key);
}

/** 读某个模型的参数；没有 / key 为空 → null。结果已校验收敛（库被手改也不会带出非法值）。 */
export function getModelParams(key: string): ModelParams | null {
  const k = normalizeKey(key);
  if (!k) return null;
  if (cache.has(k)) return cache.get(k) ?? null;
  let parsed: ModelParams | null = null;
  try {
    const row = db.select().from(modelParamsTable).where(eq(modelParamsTable.modelKey, k)).get();
    if (row) {
      const clean = sanitizeModelParams(JSON.parse(row.paramsJson));
      parsed = Object.keys(clean).length > 0 ? clean : null;
    }
  } catch {
    // JSON 坏了 / 表还没建（极早期调用）：当作没有，启动照常按全局设置走。
    parsed = null;
  }
  cache.set(k, parsed);
  return parsed;
}

/**
 * 写某个模型的参数（整份替换，不是合并 —— 界面提交的就是完整表单）。
 * 校验后为空对象 → 删除这一行（「全部恢复默认」）。返回实际落库的那份（null = 已删除）。
 */
export function setModelParams(key: string, params: ModelParams): ModelParams | null {
  const k = normalizeKey(key);
  if (!k) throw new Error("model key is required");
  const clean = sanitizeModelParams(params);
  if (Object.keys(clean).length === 0) {
    clearModelParams(k);
    return null;
  }
  const paramsJson = JSON.stringify(clean);
  const updatedAt = Date.now();
  db.insert(modelParamsTable)
    .values({ modelKey: k, paramsJson, updatedAt })
    .onConflictDoUpdate({ target: modelParamsTable.modelKey, set: { paramsJson, updatedAt } })
    .run();
  cache.set(k, clean);
  return clean;
}

export function clearModelParams(key: string): void {
  const k = normalizeKey(key);
  if (!k) return;
  db.delete(modelParamsTable).where(eq(modelParamsTable.modelKey, k)).run();
  cache.set(k, null);
}

export type ModelParamsEntry = { model: string; params: ModelParams; updatedAt: number };

/** 全部按模型参数（最近修改的在前）；解析后为空的行不列出。 */
export function listModelParams(): ModelParamsEntry[] {
  const rows = db.select().from(modelParamsTable).all();
  const out: ModelParamsEntry[] = [];
  for (const row of rows) {
    let params: ModelParams = {};
    try {
      params = sanitizeModelParams(JSON.parse(row.paramsJson));
    } catch {
      continue;
    }
    if (Object.keys(params).length === 0) continue;
    out.push({ model: row.modelKey, params, updatedAt: row.updatedAt });
  }
  return out.sort((a, b) => b.updatedAt - a.updatedAt);
}

/** 仅供测试：清掉同步读缓存（模拟进程重启后的首次读）。 */
export function __clearModelParamsCacheForTest(): void {
  cache.clear();
}
