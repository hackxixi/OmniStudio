/**
 * Hugging Face 仓库 id → ModelScope 等价仓库。
 *
 * 为什么要有这一层：引导页 / 推荐清单 / 控制套接字给出的多是 HF 的仓库 id
 * （`unsloth/Qwen3.5-4B-GGUF`、`Qwen/Qwen3.5-4B`），而下载源路由（net-sources）在国内
 * 会把默认平台切到 ModelScope。大部分组织在 ModelScope 上同名（Qwen / unsloth /
 * deepseek-ai …），少数换了组织名（meta-llama → LLM-Research、THUDM → ZhipuAI …）。
 * 这里先试同名、再试已知改名，查到就缓存；都没有就返回 missing，由调用方回退 HF 镜像，
 * 而不是直接报错。
 *
 * 反向（ModelScope id → HF）不做映射：只在调用方那里「同名直接试一次」。
 */
import type { SourcePlan } from "../shared/net-sources";
import { getSourcePlan, peekSourcePlan } from "./net-sources";

const MODELSCOPE_API = "https://modelscope.cn/api/v1/models";
const LOOKUP_TIMEOUT_MS = 8_000;

/**
 * HF 组织 → ModelScope 上的候选组织（同名之后按顺序再试）。
 *
 * 只收**实测过**的组织（2026-09 用 `GET modelscope.cn/api/v1/models/<id>` 逐个核对）：
 *   meta-llama/Llama-3.2-3B-Instruct      → LLM-Research/Llama-3.2-3B-Instruct
 *   THUDM/glm-4-9b-chat、zai-org/GLM-4.6   → ZhipuAI/…
 *   openai/gpt-oss-20b、openai/whisper-*   → openai-mirror/…（whisper 另有 AI-ModelScope/）
 *   nvidia/Llama-3.1-Nemotron-Nano-8B-v1   → nv-community/…
 *   google/gemma-*、mistralai/*、microsoft/Phi-* 同名就有，LLM-Research 下另有一份，作次选。
 * 同名即可的（不必进表）：Qwen、unsloth、deepseek-ai、BAAI、mlx-community、ggml-org、
 * bartowski、lmstudio-community、openbmb、moonshotai、stabilityai、black-forest-labs、hexgrad。
 */
export const HF_TO_MODELSCOPE_ORGS: Readonly<Record<string, readonly string[]>> = {
  "meta-llama": ["LLM-Research"],
  THUDM: ["ZhipuAI"],
  "zai-org": ["ZhipuAI"],
  openai: ["openai-mirror", "AI-ModelScope"],
  nvidia: ["nv-community"],
  google: ["LLM-Research", "AI-ModelScope"],
  mistralai: ["LLM-Research"],
  microsoft: ["LLM-Research"],
};

/** 查 ModelScope 的候选 id：同名优先，再按改名表。 */
export function modelScopeCandidates(repo: string): string[] {
  const idx = repo.indexOf("/");
  if (idx <= 0 || idx === repo.length - 1) return [];
  const org = repo.slice(0, idx);
  const name = repo.slice(idx + 1);
  // 组织名大小写在两边不一定一致（openbmb / OpenBMB），ModelScope 查询本身不分大小写。
  const renames = HF_TO_MODELSCOPE_ORGS[org] ?? HF_TO_MODELSCOPE_ORGS[org.toLowerCase()] ?? [];
  const out = [repo];
  for (const target of renames) {
    const id = `${target}/${name}`;
    if (!out.includes(id)) out.push(id);
  }
  return out;
}

/** 看起来像 `org/name` 仓库 id（而不是本地路径 / 带量化后缀的 llama.cpp 引用）。 */
export function looksLikeRepoId(value: string): boolean {
  return /^[A-Za-z0-9][\w.-]*\/[\w.-]+$/.test(value) && !value.startsWith(".");
}

export type ModelScopeLookup =
  /** ModelScope 上有等价仓库（repo 可能与请求的 id 不同：组织改名）。 */
  | { status: "found"; repo: string }
  /** 确认没有（各候选都 404）：调用方应改走 Hugging Face。 */
  | { status: "missing" }
  /** 查不了（断网 / 超时 / 5xx）：不缓存，调用方按原计划直接试。 */
  | { status: "unknown" };

type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

const cache = new Map<string, ModelScopeLookup>();
const inflight = new Map<string, Promise<ModelScopeLookup>>();

/** 单个 id 在 ModelScope 上是否存在：true / false / null（查不了）。 */
async function existsOnModelScope(id: string, fetchImpl: FetchLike): Promise<boolean | null> {
  try {
    const res = await fetchImpl(`${MODELSCOPE_API}/${id}`, { signal: AbortSignal.timeout(LOOKUP_TIMEOUT_MS) });
    await res.body?.cancel().catch(() => {});
    if (res.ok) return true;
    if (res.status === 404) return false;
    return null;
  } catch {
    return null;
  }
}

/**
 * 解析 HF 仓库在 ModelScope 上的等价仓库（带缓存、同一仓库并发只查一次）。
 * 结论只缓存 found / missing；unknown（网络问题）下次还会再查。
 */
export async function resolveModelScopeRepo(
  repo: string,
  opts: { fetchImpl?: FetchLike } = {},
): Promise<ModelScopeLookup> {
  const hit = cache.get(repo);
  if (hit) return hit;
  const pending = inflight.get(repo);
  if (pending) return pending;
  const fetchImpl = opts.fetchImpl ?? ((input, init) => fetch(input, init));
  const run = (async (): Promise<ModelScopeLookup> => {
    const candidates = modelScopeCandidates(repo);
    if (candidates.length === 0) return { status: "missing" };
    let sawUnknown = false;
    for (const id of candidates) {
      const exists = await existsOnModelScope(id, fetchImpl);
      if (exists === true) {
        const found = { status: "found", repo: id } as const;
        cache.set(repo, found);
        return found;
      }
      if (exists === null) sawUnknown = true;
    }
    if (sawUnknown) return { status: "unknown" };
    const missing = { status: "missing" } as const;
    cache.set(repo, missing);
    return missing;
  })();
  inflight.set(repo, run);
  try {
    return await run;
  } finally {
    inflight.delete(repo);
  }
}

/** 测试用：清掉映射缓存。 */
export function resetModelSourceMapCache(): void {
  cache.clear();
  inflight.clear();
}

/** ModelScope 仓库的 git clone 地址（实测 `git ls-remote` 可用的规范形式）。 */
export function modelScopeGitUrl(repo: string): string {
  return `https://www.modelscope.cn/${repo}.git`;
}

/**
 * 拿下载源路由，但最多等 `timeoutMs`：首次探测可能要几秒，不能拖住引擎启动 /
 * 下载开跑；超时就用同步的 peekSourcePlan（缓存或按地区猜的结论）。
 */
export async function sourcePlanWithin(timeoutMs = 3_000): Promise<SourcePlan> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      getSourcePlan(),
      new Promise<SourcePlan>((resolve) => {
        timer = setTimeout(() => resolve(peekSourcePlan()), timeoutMs);
      }),
    ]);
  } catch {
    return peekSourcePlan();
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/** HF 端点列表（去掉末尾斜杠、去重；plan 给空列表时退回官方）。 */
export function hfEndpointsOf(plan: SourcePlan): string[] {
  const out: string[] = [];
  for (const e of plan.hfEndpoints ?? []) {
    const v = e.trim().replace(/\/+$/, "");
    if (v && !out.includes(v)) out.push(v);
  }
  return out.length > 0 ? out : ["https://huggingface.co"];
}
