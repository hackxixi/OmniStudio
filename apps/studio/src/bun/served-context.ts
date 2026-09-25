/**
 * 本地推理实例的「真实上下文窗口」探测 + 本地 max_tokens 的取值口径。
 *
 * 为什么不能只看设置：`SERVER_CTX_SIZE` 只是启动参数的一个输入 —— llama.cpp 走自动规划时
 * 实际 ctx 由规划器按显存算出来，`--parallel N` 还会把总窗口按 slot 均分；vLLM 的
 * `max_model_len` 可能被显存压小。按设置 / config.json 推出来的数可能**大于**真实窗口：
 * - vLLM / SGLang：`prompt + max_tokens > max_model_len` 直接 400 拒掉整个请求；
 * - llama.cpp：`n_predict` 超出剩余窗口不报错，只会在窗口用尽时以 length 收尾。
 *
 * 所以直接问正在跑的服务：
 * - llama.cpp `GET /props` → `default_generation_settings.n_ctx`（**每 slot** 窗口，单个请求能用的就是它）；
 * - vLLM / SGLang `GET /v1/models` → 对应条目的 `max_model_len`。
 * 都拿不到（mlx-lm 等不暴露窗口）才退回设置 / config.json 的旧口径。
 *
 * 结果按「base URL + 模型 id」缓存一小段时间：每次发消息都探两个 HTTP 端点太浪费，
 * 而实例重启换了 ctx 时，短 TTL + 请求失败时 `invalidateServedContext` 能很快跟上。
 */

import { estimateMessagesTokens, type MessageLike } from "../shared/token-estimate";

/** 窗口来源：决定 max_tokens 要不要给 prompt 让位（见 localMaxTokens）。 */
export type ServedContextSource = "llama-props" | "max-model-len";

export type ServedContext = { window: number; source: ServedContextSource };

/** 探测结果（含「探不到」）的缓存时长。短一点：实例重启换 ctx 后最多这么久就跟上。 */
export const SERVED_CONTEXT_TTL_MS = 30_000;
/** 单个探测请求的超时：服务已就绪，本机回环正常毫秒级返回，慢了就当探不到。 */
const PROBE_TIMEOUT_MS = 1500;

/** 本地 max_tokens 的上限（与旧实现一致：256K）与下限。 */
export const LOCAL_MAX_TOKENS_CAP = 262_144;
export const LOCAL_MAX_TOKENS_FLOOR = 1024;
/**
 * 按 max_model_len 扣掉 prompt 后至少留给输出的量。prompt 已经把窗口吃满时给多少
 * 都会被拒，这里只是不让它变成 0 / 负数。
 */
const MIN_OUTPUT_AFTER_PROMPT = 256;
/** prompt 估算的放大系数 + 固定余量：粗估（CJK 1 字 1 token、英文 4 字符 1 token）会偏小，宁可少给一点输出。 */
const PROMPT_ESTIMATE_FACTOR = 1.25;
const PROMPT_ESTIMATE_PAD = 256;

type CacheEntry = { value: ServedContext | null; expiresAt: number };
const cache = new Map<string, CacheEntry>();
/** 同一个 key 的并发探测合并成一次（连发几条消息 / 翻译 + 对话同时进来）。 */
const inflight = new Map<string, Promise<ServedContext | null>>();

function cacheKey(base: string, model: string): string {
  return `${base.replace(/\/+$/, "")}\u0000${model}`;
}

function positiveInt(v: unknown): number | null {
  const n = typeof v === "number" ? v : typeof v === "string" && v.trim() ? Number(v) : NaN;
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : null;
}

async function getJson(url: string, headers?: Record<string, string>): Promise<unknown | null> {
  try {
    const res = await fetch(url, { method: "GET", headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

/** 真正去问服务（不走缓存）。拿不到返回 null。 */
export async function probeServedContext(
  base: string,
  model: string,
  headers?: Record<string, string>,
): Promise<ServedContext | null> {
  const root = base.replace(/\/+$/, "");

  // llama.cpp：/props 的 n_ctx 是每 slot 窗口（--ctx-size 已按 --parallel 均分）。
  const props = (await getJson(`${root}/props`, headers)) as {
    default_generation_settings?: { n_ctx?: unknown } | null;
  } | null;
  const nCtx = positiveInt(props?.default_generation_settings?.n_ctx);
  if (nCtx) return { window: nCtx, source: "llama-props" };

  // vLLM / SGLang：/v1/models 条目上的 max_model_len。优先按 id 对上；只有一个条目时就是它。
  const list = (await getJson(`${root}/v1/models`, headers)) as { data?: unknown } | null;
  const data = Array.isArray(list?.data) ? (list.data as Array<Record<string, unknown>>) : [];
  const entry = data.find((m) => m?.id === model || m?.root === model) ?? (data.length === 1 ? data[0] : undefined);
  const maxLen = positiveInt(entry?.max_model_len);
  if (maxLen) return { window: maxLen, source: "max-model-len" };

  return null;
}

/** 带缓存的探测：TTL 内直接返回（包括「探不到」），并发请求共享一次探测。 */
export async function getServedContext(
  base: string,
  model: string,
  headers?: Record<string, string>,
): Promise<ServedContext | null> {
  const key = cacheKey(base, model);
  const hit = cache.get(key);
  if (hit && hit.expiresAt > Date.now()) return hit.value;

  const pending = inflight.get(key);
  if (pending) return pending;

  const p = probeServedContext(base, model, headers)
    .then((value) => {
      cache.set(key, { value, expiresAt: Date.now() + SERVED_CONTEXT_TTL_MS });
      return value;
    })
    .finally(() => inflight.delete(key));
  inflight.set(key, p);
  return p;
}

/** 只读缓存、不发请求（给同步调用方用）；没有 / 过期返回 undefined。 */
export function peekServedContext(base: string, model: string): ServedContext | null | undefined {
  const hit = cache.get(cacheKey(base, model));
  if (!hit || hit.expiresAt <= Date.now()) return undefined;
  return hit.value;
}

/**
 * 作废缓存：请求失败（可能是实例重启、换了 ctx）时调用，下一次重新探测。
 * 不传参数 = 全部清掉。
 */
export function invalidateServedContext(base?: string, model?: string): void {
  if (base === undefined) {
    cache.clear();
    return;
  }
  if (model !== undefined) {
    cache.delete(cacheKey(base, model));
    return;
  }
  const prefix = `${base.replace(/\/+$/, "")}\u0000`;
  for (const key of cache.keys()) if (key.startsWith(prefix)) cache.delete(key);
}

/**
 * 本地 max_tokens：
 * - 探到 llama.cpp 每 slot 窗口：直接给窗口 —— llama.cpp 会在窗口用尽时自己以 length 收尾、
 *   不会报错，没必要拿粗估的 prompt 去扣（扣多了反而白白截短回答）；
 * - 探到 max_model_len（vLLM / SGLang）：必须给 prompt 让位，否则 `prompt + max_tokens`
 *   超窗整请求 400；prompt 估算放大一点，结果至少留 MIN_OUTPUT_AFTER_PROMPT；
 * - 探不到：用调用方给的兜底窗口（旧口径）。
 * 最后封顶 256K；兜底口径另有 1024 下限，探到真实窗口时则**不会**超过它。
 */
export function localMaxTokens(opts: {
  served: ServedContext | null | undefined;
  fallbackWindow: number;
  promptMessages?: MessageLike[];
}): number {
  const { served, fallbackWindow } = opts;
  if (!served) {
    return Math.min(Math.max(LOCAL_MAX_TOKENS_FLOOR, fallbackWindow), LOCAL_MAX_TOKENS_CAP);
  }
  let budget = served.window;
  if (served.source === "max-model-len") {
    // 调用方没给 prompt（同步口径）时不知道要让多少：按一半窗口留给输入，宁可输出上限保守。
    const prompt = opts.promptMessages?.length
      ? Math.ceil(estimateMessagesTokens(opts.promptMessages) * PROMPT_ESTIMATE_FACTOR) + PROMPT_ESTIMATE_PAD
      : Math.ceil(served.window / 2);
    budget = Math.max(MIN_OUTPUT_AFTER_PROMPT, served.window - prompt);
  }
  // 探到真实窗口时不套 1024 下限：不能为了凑下限超出服务实际能给的量。
  return Math.min(budget, LOCAL_MAX_TOKENS_CAP);
}

/** 测试用：清空缓存与在途探测。 */
export function resetServedContextCache(): void {
  cache.clear();
  inflight.clear();
}
