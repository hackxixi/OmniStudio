import { existsSync, mkdirSync, readdirSync, renameSync, statSync } from "fs";
import path from "path";
import {
  isModelWeightExt,
  matchFormat,
  modelDisplayName,
  modelFormats,
  safeRepoId,
  type MarketFile,
  type MarketModel,
  type MarketSearchResult,
  type SearchFormat,
} from "../shared/modelscope";
import { getDataDir } from "./paths";
import { safeJoin } from "./path-safety";
import {
  downloadWithResume,
  partialBytesFor,
  removePartialFiles,
  type DownloadOptions,
  type DownloadProgress,
} from "./downloader";
import { reportSourceFailure } from "./net-sources";
import { hfEndpointsOf, resolveModelScopeRepo, sourcePlanWithin } from "./model-source-map";
import { listRepoFiles as listHuggingFaceRepoFiles } from "./huggingface";

export { isModelWeightExt, modelDisplayName, safeRepoId, removePartialFiles };
export type { MarketFile, MarketModel, DownloadProgress };

const MODELSCOPE_BASE = "https://modelscope.cn";
const OPENAPI_BASE = `${MODELSCOPE_BASE}/openapi/v1`;

/** 搜索接口返回项（openapi/v1/models），字段与内部类型同名但为 snake_case。 */
type ModelScopeApiModel = {
  id: string;
  display_name?: string;
  description?: string;
  downloads?: number;
  likes?: number;
  license?: string;
  tasks?: string[];
  tags?: string[];
  file_size?: number;
  params?: number;
  created_at?: string;
  last_modified?: string;
};

function fileKind(name: string): MarketFile["kind"] {
  const n = name.toLowerCase();
  if (n.endsWith(".gguf") || n.endsWith(".ggml")) return "gguf";
  if (n.endsWith(".safetensors")) return "safetensors";
  return "other";
}

/**
 * Dev builds run with the process CWD inside the app bundle, which electrobun
 * regenerates on every rebuild — a CWD-relative data dir would be wiped
 * together with the downloaded models. Always use userData so data survives.
 */
export function getModelsBaseDir(): string {
  const base = getDataDir("models");
  migrateLegacyCwdDir("vllm-studio-models", base);
  migrateLegacyCwdDir("kunpengtalk-studio-models", base);
  migrateLegacyCwdDir("omni-studio-models", base);
  return base;
}

/**
 * One-time move of dev data that used to live next to the process CWD (the app
 * bundle in dev) into userData. No-op once the legacy dir is gone.
 */
function migrateLegacyCwdDir(legacyName: string, dest: string): void {
  try {
    const legacy = path.resolve(legacyName);
    if (legacy === dest || !existsSync(legacy) || existsSync(dest)) return;
    mkdirSync(path.dirname(dest), { recursive: true });
    renameSync(legacy, dest);
  } catch {
    // ignore — data inside a wiped bundle dir is already unrecoverable
  }
}

export function splitRepo(repo: string): { owner: string; name: string } {
  const idx = repo.indexOf("/");
  if (idx < 0) return { owner: repo, name: repo };
  return { owner: repo.slice(0, idx), name: repo.slice(idx + 1) };
}

/**
 * ModelScope 搜索 URL。
 *
 * 格式过滤：ModelScope 的检索接口**不支持**按库标签过滤（`filter` / `tags` /
 * `library` / `SingleCriterion` 参数实测都被忽略，返回同样的结果集），所以只能
 * 把格式关键词并进检索词（`qwen3` + gguf → `qwen3 gguf`），让平台按相关性排序，
 * 再用结果里的 `library:*` 标签二次确认（见 searchModels 的 filtered）。
 * Hugging Face 走的是真·服务端过滤（huggingface.ts），两边差异在 UI 上有说明。
 */
export function buildSearchUrl(
  query: string,
  page: number,
  pageSize: number,
  format?: SearchFormat,
): URL {
  const url = new URL(`${OPENAPI_BASE}/models`);
  url.searchParams.set("search", format ? `${query} ${format}` : query);
  url.searchParams.set("page", String(page));
  url.searchParams.set("page_size", String(pageSize));
  return url;
}

export async function searchModels(
  query: string,
  page = 1,
  pageSize = 20,
  format?: SearchFormat,
): Promise<MarketSearchResult> {
  const url = buildSearchUrl(query, page, pageSize, format);

  const res = await fetch(url, { signal: AbortSignal.timeout(15_000) });
  if (!res.ok) throw new Error(`ModelScope search failed: ${res.status}`);

  const body = (await res.json()) as {
    data?: { models?: ModelScopeApiModel[]; total_count?: number; total?: number };
  };

  const list = body.data?.models ?? [];
  const models: MarketModel[] = list.map((m) => {
    const tags = m.tags ?? [];
    return {
      id: m.id ?? "",
      name: m.display_name || m.id || "",
      description: m.description ?? "",
      downloads: m.downloads ?? 0,
      likes: m.likes ?? 0,
      license: m.license ?? "",
      tasks: m.tasks ?? [],
      tags,
      fileSize: m.file_size ?? 0,
      params: m.params ?? 0,
      createdAt: m.created_at ?? "",
      lastModified: m.last_modified ?? "",
      source: "modelscope" as const,
      formats: modelFormats(tags),
      fileCount: 0,
    };
  });

  const filtered = format ? models.filter((m) => matchFormat(m.formats, format)) : models;
  // 接口返回的是 total_count（不是 total）——之前读错字段导致分页总数永远是当前页条数。
  const total = body.data?.total_count ?? body.data?.total ?? filtered.length;
  return {
    models: filtered,
    total,
    totalExact: true,
    hasMore: page * pageSize < total && filtered.length > 0,
  };
}

/**
 * 列 ModelScope 仓库文件。仓库不在 ModelScope 上（404，常见于引导页 / 推荐清单给的
 * HF 仓库 id）时不直接报错：先按 model-source-map 找 ModelScope 上的等价仓库
 * （组织改名），再没有就改列 Hugging Face（走镜像）—— 下载管理器按同样的顺序回退，
 * 列出来的文件名与实际下载的字节来自同一个仓库。
 */
export async function listRepoFiles(repo: string): Promise<MarketFile[]> {
  const direct = await listModelScopeFiles(repo);
  if (direct) return direct;
  const mapped = await resolveModelScopeRepo(repo);
  if (mapped.status === "found" && mapped.repo !== repo) {
    const files = await listModelScopeFiles(mapped.repo);
    if (files) return files;
  }
  return listHuggingFaceRepoFiles(repo);
}

/** 直接列 ModelScope 上这个 id 的文件；仓库不存在返回 null（其它错误照常抛出）。 */
export async function listModelScopeFiles(repo: string): Promise<MarketFile[] | null> {
  const { owner, name } = splitRepo(repo);
  const url = `${MODELSCOPE_BASE}/api/v1/models/${owner}/${name}/repo/files?Revision=master&Recursive=true`;

  const res = await fetch(url, { signal: AbortSignal.timeout(20_000) });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`ModelScope files failed: ${res.status}`);

  const body = (await res.json()) as {
    Data?: { Files?: Array<{ Name: string; Path: string; Size: number; IsLFS: boolean }> };
  };

  // Return every file in the repo (not just GGUF): vLLM / SGLang models are
  // shipped as safetensors + config, TTS/ASR models as bin/pt/onnx, etc.
  // Model weights are sorted first so llama.cpp / vLLM / SGLang users can
  // pick the right file quickly; README/config/tokenizer stay downloadable.
  return (body.Data?.Files ?? [])
    .map((f) => ({
      name: f.Name,
      path: f.Path,
      size: f.Size,
      isLfs: f.IsLFS,
      kind: fileKind(f.Name),
      isWeight: isModelWeightExt(f.Name),
    }))
    .sort((a, b) => {
      const ra = a.isWeight ? 0 : 1;
      const rb = b.isWeight ? 0 : 1;
      return ra - rb || b.size - a.size;
    });
}

export function resolveFileUrl(repo: string, filePath: string): string {
  const { owner, name } = splitRepo(repo);
  return `${MODELSCOPE_BASE}/models/${owner}/${name}/resolve/master/${filePath}`;
}

export function localModelPath(repo: string, fileName: string): string {
  return path.join(getModelsBaseDir(), safeRepoId(repo), fileName);
}

/**
 * 模型下载的落盘路径（带穿越校验）。repo / fileName 会从 RPC 与控制套接字传入，
 * `fileName = "../../omni-studio.db"` 这类输入能覆盖数据目录下的任意文件，必须走这里。
 * 越界返回 null，调用方要当成参数错误处理。
 */
export function modelDestPath(repo: string, fileName: string): string | null {
  return safeJoin(path.join(getModelsBaseDir(), safeRepoId(repo)), fileName);
}

export function isModelInstalled(repo: string, fileName: string): boolean {
  return existsSync(localModelPath(repo, fileName));
}

export function installedModelSize(repo: string, fileName: string): number | null {
  const p = localModelPath(repo, fileName);
  if (!existsSync(p)) return null;
  try {
    return statSync(p).size;
  } catch {
    return null;
  }
}

/**
 * 磁盘上已经下过的字节数（分片 + 已就位部分），**不发网络请求**。
 * 市场列表用它显示「继续下载（已 1.2/4.5 GB）」；`total` 传市场给的 size 时
 * 会校验本地分片是否还属于这份文件（远端换过就返回 0）。
 */
export function downloadedBytesOf(repo: string, fileName: string, total?: number | null): number {
  const dest = modelDestPath(repo, fileName);
  if (!dest) return 0;
  return partialBytesFor(dest, total);
}

/** 该仓库目录下所有 `.part*` / `.download.json` 旁路文件的体积合计（用于「清理未完成」）。 */
export function partialBytesInRepo(repo: string): { files: number; bytes: number } {
  const dir = path.join(getModelsBaseDir(), safeRepoId(repo));
  let files = 0;
  let bytes = 0;
  try {
    for (const name of readdirSync(dir)) {
      if (!name.includes(".part") && !name.endsWith(".download.json")) continue;
      try {
        const stat = statSync(path.join(dir, name));
        if (stat.isFile()) {
          files += 1;
          bytes += stat.size;
        }
      } catch {
        // ignore
      }
    }
  } catch {
    // ignore
  }
  return { files, bytes };
}

/**
 * 下载一个仓库文件到模型目录。多路并发、断点续传、卡死重连与重试都在
 * downloader 内核里，这里只负责把 repo/fileName 映射成 URL 与落盘路径。
 */
export async function downloadFile(
  repo: string,
  fileName: string,
  options: DownloadOptions = {},
  /**
   * 实际去 ModelScope 拉字节的仓库 id（HF id 映射到 ModelScope 改名组织时与 repo 不同）；
   * 落盘目录始终按 repo，本地模型列表 / 已下载判断才认得出来。
   */
  remoteRepo: string = repo,
): Promise<{ path: string; size: number }> {
  const dir = path.join(getModelsBaseDir(), safeRepoId(repo));
  mkdirSync(dir, { recursive: true });
  const destPath = modelDestPath(repo, fileName);
  if (!destPath) throw new Error(`非法的模型文件名：${fileName}`);

  return downloadWithResume(resolveFileUrl(remoteRepo, fileName), destPath, options);
}

/** 下载内核抛出的 HTTP 错误带 status（见 downloader 的 HttpStatusError）。 */
export function httpStatusOf(e: unknown): number | null {
  const status = (e as { status?: unknown } | null)?.status;
  if (typeof status === "number") return status;
  const m = e instanceof Error ? /failed: (\d{3})\b/.exec(e.message) : null;
  return m ? Number(m[1]) : null;
}

/** 404 / 401 / 403 / 451：文件不在 / 要授权 —— 是仓库的事，不是这个源连不上。 */
function isRepoSideStatus(status: number | null): boolean {
  return status === 404 || status === 401 || status === 403 || status === 451;
}

/**
 * 下载 HuggingFace 仓库里的单个文件（audio.cpp GGUF 等不在 ModelScope 上的资源）。
 * 端点顺序来自下载源路由（net-sources：国内镜像优先或官方直连优先，官方永远兜底）；
 * 每个端点都继承断点续传与重试（换端点时保留已下载的分片，从断点接着下）。
 * 连不上的端点报给路由（reportSourceFailure）降级；404 这类仓库侧错误不算源的锅。
 */
export async function downloadHuggingFaceFile(
  repo: string,
  filePath: string,
  options: DownloadOptions = {},
): Promise<{ path: string; size: number }> {
  const destPath = modelDestPath(repo, filePath);
  if (!destPath) throw new Error(`非法的模型文件路径：${filePath}`);
  mkdirSync(path.dirname(destPath), { recursive: true });

  const endpoints = hfEndpointsOf(await sourcePlanWithin());
  let lastError: Error | null = null;
  for (const endpoint of endpoints) {
    const url = `${endpoint}/${repo}/resolve/main/${filePath}`;
    try {
      return await downloadWithResume(url, destPath, options);
    } catch (e) {
      // 被用户取消/暂停就直接抛出，不换镜像。
      if (options.signal?.aborted) throw e;
      // 第一个错误最有信息量（首选源为什么不行）；但 404 这类要保留下来给上层判断回退。
      const err = e instanceof Error ? e : new Error(String(e));
      if (!lastError || isRepoSideStatus(httpStatusOf(err))) lastError = err;
      if (!isRepoSideStatus(httpStatusOf(err))) reportSourceFailure(url);
    }
  }
  throw lastError ?? new Error("HF download failed");
}

/**
 * 通用 url + destPath 下载（PaddleOCR 引擎模型等复用）。
 * 保留旧调用形状：`(url, destPath, onProgress, signal)`。
 */
export async function downloadHttpFile(
  url: string,
  destPath: string,
  onProgress?: (progress: DownloadProgress) => void,
  signal?: AbortSignal,
): Promise<{ path: string; size: number }> {
  return downloadWithResume(url, destPath, { onProgress, signal });
}
