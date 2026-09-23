/**
 * GitHub 元数据（API / 仓库）请求的多链路版本。
 *
 * 资产下载早就走了多镜像（mirror-download.ts），但「最新版本是哪个」这类元数据查询
 * 还是只打 api.github.com —— 国内网络下它一挂，后面的镜像下载根本轮不到。这里按下载源
 * 计划的 GitHub 前缀顺序逐个试（海外模式直连在前，行为与原来一致）：
 *
 * - api.github.com：实测只有 gh-proxy.com 能代理（ghfast.top / ghproxy.net 回 403 网页），
 *   所以每条链路都要校验「真的拿到了 JSON」，拿到网页就换下一条。
 * - git 仓库（clone / ls-remote）：三个前缀镜像都能代理 smart HTTP（含 partial clone）。
 */
import type { SourcePlan } from "../shared/net-sources";
import { githubCandidates } from "./net-sources";

/** 单条链路的超时：首条沿用调用方给的（海外直连与原来一致），后面的镜像给短一点，连不上就换。 */
const MIRROR_TIMEOUT_MS = 10_000;

function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

/** 镜像失败时的简短原因（进错误信息，便于从日志判断是哪条链路、怎么挂的）。 */
function reason(e: unknown): string {
  if (e instanceof Error) return e.name === "TimeoutError" ? "超时" : e.message;
  return String(e);
}

export type GithubFetchOptions = {
  plan?: SourcePlan;
  fetchImpl?: typeof fetch;
  /** 第一条链路的超时（默认 15s）。 */
  timeoutMs?: number;
  headers?: Record<string, string>;
};

/**
 * 按链路顺序取 GitHub API 的 JSON。`validate` 不通过（代理回了错误页 / 别的 JSON）也换下一条。
 * 全部失败抛错，错误里带每条链路的原因。
 */
export async function fetchGithubJson<T>(
  url: string,
  opts: GithubFetchOptions & { validate?: (json: unknown) => boolean } = {},
): Promise<T> {
  const fetchImpl = opts.fetchImpl ?? fetch;
  const candidates = githubCandidates(url, opts.plan);
  const errors: string[] = [];
  for (let i = 0; i < candidates.length; i++) {
    const candidate = candidates[i]!;
    const via = candidate === url ? "直连" : hostOf(candidate);
    try {
      const res = await fetchImpl(candidate, {
        headers: opts.headers,
        signal: AbortSignal.timeout(i === 0 ? (opts.timeoutMs ?? 15_000) : MIRROR_TIMEOUT_MS),
      });
      if (!res.ok) {
        errors.push(`${via}：HTTP ${res.status}`);
        continue;
      }
      const text = await res.text();
      let json: unknown;
      try {
        json = JSON.parse(text);
      } catch {
        errors.push(`${via}：返回的不是 JSON`);
        continue;
      }
      if (opts.validate && !opts.validate(json)) {
        errors.push(`${via}：返回内容不对`);
        continue;
      }
      return json as T;
    } catch (e) {
      errors.push(`${via}：${reason(e)}`);
    }
  }
  throw new Error(`GitHub API 不可达（${errors.join("；")}）`);
}

/**
 * 按链路顺序取 GitHub 上的一段文本（仓库 refs、release 页片段等）。`validate` 判定内容可用。
 * 全部失败返回 null（调用方决定怎么兜底），失败原因写进 `errors`。
 */
export async function fetchGithubText(
  url: string,
  opts: GithubFetchOptions & { validate?: (text: string) => boolean; errors?: string[] } = {},
): Promise<string | null> {
  const fetchImpl = opts.fetchImpl ?? fetch;
  for (const candidate of githubCandidates(url, opts.plan)) {
    const via = candidate === url ? "直连" : hostOf(candidate);
    try {
      // 文本类请求（refs 有 1～2MB）经镜像也要几秒，每条链路用同一个预算。
      const res = await fetchImpl(candidate, {
        headers: opts.headers,
        signal: AbortSignal.timeout(opts.timeoutMs ?? 15_000),
      });
      if (!res.ok) {
        opts.errors?.push(`${via}：HTTP ${res.status}`);
        continue;
      }
      const text = await res.text();
      if (opts.validate && !opts.validate(text)) {
        opts.errors?.push(`${via}：返回内容不对`);
        continue;
      }
      return text;
    } catch (e) {
      opts.errors?.push(`${via}：${reason(e)}`);
    }
  }
  return null;
}

/**
 * git 远端的候选地址：GitHub 的 https 仓库按计划排好前缀镜像（`<prefix>https://github.com/o/r.git`，
 * 三个镜像都实测能 clone / ls-remote / partial clone）；其它远端（GitLab、ssh、自建）原样返回。
 */
export function gitRemoteCandidates(url: string, plan?: SourcePlan): string[] {
  if (!/^https:\/\/github\.com\//i.test(url)) return [url];
  return githubCandidates(url, plan);
}
