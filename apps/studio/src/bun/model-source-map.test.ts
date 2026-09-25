import { afterAll, beforeEach, expect, test } from "bun:test";

import type { SourcePlan } from "../shared/net-sources";
import { installFetchRouter, mockModulePartial } from "./test-mocks";

/**
 * 下载源相关的纯逻辑：HF → ModelScope 仓库映射（假 fetch，不联网）、各下载入口按路由排序
 * 候选源、以及「国内 / 海外」两种结论下顺序是否对。
 *
 * 路由（net-sources）整体换成可控的假结论：真实实现会去探测网络。
 */

function plan(over: Partial<SourcePlan> = {}): SourcePlan {
  return {
    mode: "cn",
    decidedBy: "setting",
    cnLocale: true,
    modelSource: "modelscope",
    hfEndpoints: ["https://hf-mirror.com", "https://huggingface.co"],
    pypiIndexes: [],
    githubPrefixes: ["https://gh-proxy.example/", ""],
    homebrewEnv: {},
    probes: [],
    at: 0,
    ...over,
  };
}

let currentPlan = plan();
/** getSourcePlan 的行为：默认立刻给 currentPlan；置为 "hang" 模拟首次探测很慢。 */
let getMode: "ok" | "hang" = "ok";
const reported: string[] = [];
await mockModulePartial<typeof import("./net-sources")>("./net-sources", {
  getSourcePlan: () => (getMode === "hang" ? new Promise<SourcePlan>(() => {}) : Promise.resolve(currentPlan)),
  peekSourcePlan: () => currentPlan,
  githubCandidates: (url: string, p?: SourcePlan) => (p ?? currentPlan).githubPrefixes.map((x) => `${x}${url}`),
  reportSourceFailure: (url: string) => {
    reported.push(url);
  },
});

const {
  hfEndpointsOf,
  looksLikeRepoId,
  modelScopeCandidates,
  modelScopeGitUrl,
  resetModelSourceMapCache,
  resolveModelScopeRepo,
  sourcePlanWithin,
} = await import("./model-source-map");

const originalFetch = globalThis.fetch;
const setFetch = installFetchRouter();

afterAll(() => {
  globalThis.fetch = originalFetch;
});

beforeEach(() => {
  resetModelSourceMapCache();
  currentPlan = plan();
  getMode = "ok";
  reported.length = 0;
});

/** 假 ModelScope：只有 `existing` 里的 id 存在（200），`broken` 里的返回 500，其余 404。 */
function fakeModelScope(existing: string[], broken: string[] = []) {
  const calls: string[] = [];
  const fetchImpl = async (input: string) => {
    const id = input.replace("https://modelscope.cn/api/v1/models/", "");
    calls.push(id);
    if (broken.includes(id)) return new Response("oops", { status: 500 });
    return new Response("{}", { status: existing.includes(id) ? 200 : 404 });
  };
  return { calls, fetchImpl };
}

test("候选：同名优先，再按已核实的组织改名表", () => {
  expect(modelScopeCandidates("unsloth/Qwen3.5-4B-GGUF")).toEqual(["unsloth/Qwen3.5-4B-GGUF"]);
  expect(modelScopeCandidates("meta-llama/Llama-3.2-3B-Instruct")).toEqual([
    "meta-llama/Llama-3.2-3B-Instruct",
    "LLM-Research/Llama-3.2-3B-Instruct",
  ]);
  expect(modelScopeCandidates("THUDM/glm-4-9b-chat")).toEqual(["THUDM/glm-4-9b-chat", "ZhipuAI/glm-4-9b-chat"]);
  expect(modelScopeCandidates("openai/gpt-oss-20b")).toEqual([
    "openai/gpt-oss-20b",
    "openai-mirror/gpt-oss-20b",
    "AI-ModelScope/gpt-oss-20b",
  ]);
  // 不是 org/name 的输入（本地路径、裸名）不给候选。
  expect(modelScopeCandidates("no-slash")).toEqual([]);
  expect(modelScopeCandidates("org/")).toEqual([]);
});

test("同名存在 → found 原 id，只查一次", async () => {
  const ms = fakeModelScope(["Qwen/Qwen3.5-4B"]);
  expect(await resolveModelScopeRepo("Qwen/Qwen3.5-4B", { fetchImpl: ms.fetchImpl })).toEqual({
    status: "found",
    repo: "Qwen/Qwen3.5-4B",
  });
  expect(ms.calls).toEqual(["Qwen/Qwen3.5-4B"]);
});

test("组织改名：同名 404 → 改名组织命中，结果缓存", async () => {
  const ms = fakeModelScope(["LLM-Research/Llama-3.2-3B-Instruct"]);
  const first = await resolveModelScopeRepo("meta-llama/Llama-3.2-3B-Instruct", { fetchImpl: ms.fetchImpl });
  expect(first).toEqual({ status: "found", repo: "LLM-Research/Llama-3.2-3B-Instruct" });
  expect(ms.calls).toEqual(["meta-llama/Llama-3.2-3B-Instruct", "LLM-Research/Llama-3.2-3B-Instruct"]);
  // 再查走缓存，不再发请求。
  await resolveModelScopeRepo("meta-llama/Llama-3.2-3B-Instruct", { fetchImpl: ms.fetchImpl });
  expect(ms.calls).toHaveLength(2);
});

test("各候选都 404 → missing（缓存）；网络问题 → unknown（不缓存，下回再查）", async () => {
  const gone = fakeModelScope([]);
  expect(await resolveModelScopeRepo("someone/hf-only", { fetchImpl: gone.fetchImpl })).toEqual({ status: "missing" });
  await resolveModelScopeRepo("someone/hf-only", { fetchImpl: gone.fetchImpl });
  expect(gone.calls).toEqual(["someone/hf-only"]);

  const flaky = fakeModelScope([], ["x/y"]);
  expect(await resolveModelScopeRepo("x/y", { fetchImpl: flaky.fetchImpl })).toEqual({ status: "unknown" });
  await resolveModelScopeRepo("x/y", { fetchImpl: flaky.fetchImpl });
  expect(flaky.calls).toEqual(["x/y", "x/y"]);

  const offline = async () => {
    throw new TypeError("fetch failed");
  };
  expect(await resolveModelScopeRepo("a/b", { fetchImpl: offline })).toEqual({ status: "unknown" });
});

test("同一仓库并发查询只发一轮请求", async () => {
  const ms = fakeModelScope(["unsloth/Qwen3.5-9B-GGUF"]);
  const [a, b] = await Promise.all([
    resolveModelScopeRepo("unsloth/Qwen3.5-9B-GGUF", { fetchImpl: ms.fetchImpl }),
    resolveModelScopeRepo("unsloth/Qwen3.5-9B-GGUF", { fetchImpl: ms.fetchImpl }),
  ]);
  expect(a).toEqual(b);
  expect(ms.calls).toEqual(["unsloth/Qwen3.5-9B-GGUF"]);
});

test("仓库 id 判定与 ModelScope clone 地址", () => {
  expect(looksLikeRepoId("Qwen/Qwen3.5-4B")).toBe(true);
  expect(looksLikeRepoId("/Users/me/models/qwen")).toBe(false);
  expect(looksLikeRepoId("./models/qwen")).toBe(false);
  expect(looksLikeRepoId("unsloth/Qwen3.5-4B-GGUF:Q4_K_M")).toBe(false);
  expect(modelScopeGitUrl("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")).toBe(
    "https://www.modelscope.cn/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice.git",
  );
});

test("hfEndpointsOf：去尾斜杠、去重；空列表退回官方", () => {
  expect(hfEndpointsOf(plan({ hfEndpoints: ["https://hf-mirror.com/", "https://hf-mirror.com", "https://huggingface.co"] }))).toEqual([
    "https://hf-mirror.com",
    "https://huggingface.co",
  ]);
  expect(hfEndpointsOf(plan({ hfEndpoints: [] }))).toEqual(["https://huggingface.co"]);
});

test("sourcePlanWithin：探测太慢时不等，用 peek 的结论", async () => {
  getMode = "hang";
  currentPlan = plan({ mode: "global", modelSource: "huggingface" });
  const t0 = Date.now();
  const p = await sourcePlanWithin(50);
  expect(Date.now() - t0).toBeLessThan(1_000);
  expect(p.modelSource).toBe("huggingface");
});

// ---------------------------------------------------------------------------
// 各下载入口的源顺序
// ---------------------------------------------------------------------------

test("HF 检索按路由端点顺序打；连不上的端点报给路由并换下一个", async () => {
  const { searchModels } = await import("./huggingface");
  const hosts: string[] = [];
  setFetch((async (input: RequestInfo | URL) => {
    const url = String(input);
    hosts.push(new URL(url).host);
    if (url.startsWith("https://huggingface.co")) throw new TypeError("fetch failed");
    return new Response("[]", { status: 200 });
  }) as typeof fetch);

  // 海外 / 开代理：官方优先，官方挂了换镜像。
  currentPlan = plan({ mode: "global", hfEndpoints: ["https://huggingface.co", "https://hf-mirror.com"] });
  await searchModels("qwen");
  expect(hosts).toEqual(["huggingface.co", "hf-mirror.com"]);
  expect(reported).toHaveLength(1);
  expect(reported[0]).toStartWith("https://huggingface.co/api/models?");

  // 国内：镜像优先，镜像通了就不碰官方。
  hosts.length = 0;
  currentPlan = plan();
  await searchModels("qwen");
  expect(hosts).toEqual(["hf-mirror.com"]);
});

test("ModelScope 列文件：仓库 404 → 改名组织 → 仍没有就列 Hugging Face", async () => {
  const { listRepoFiles } = await import("./modelscope");
  const seen: string[] = [];
  setFetch((async (input: RequestInfo | URL) => {
    const url = String(input);
    seen.push(url);
    // 映射查询：只有 LLM-Research 下有。
    if (url === "https://modelscope.cn/api/v1/models/LLM-Research/Llama-3.2-3B-Instruct") {
      return new Response("{}", { status: 200 });
    }
    if (url.startsWith("https://modelscope.cn/api/v1/models/LLM-Research/Llama-3.2-3B-Instruct/repo/files")) {
      return Response.json({ Data: { Files: [{ Name: "config.json", Path: "config.json", Size: 10, IsLFS: false }] } });
    }
    if (url.startsWith("https://modelscope.cn/")) return new Response("{}", { status: 404 });
    if (url.startsWith("https://hf-mirror.com/api/models/someone/hf-only/tree")) {
      return Response.json([{ type: "file", path: "w.gguf", size: 5 }]);
    }
    return new Response("[]", { status: 404 });
  }) as typeof fetch);

  const renamed = await listRepoFiles("meta-llama/Llama-3.2-3B-Instruct");
  expect(renamed.map((f) => f.name)).toEqual(["config.json"]);

  const hfOnly = await listRepoFiles("someone/hf-only");
  expect(hfOnly.map((f) => f.name)).toEqual(["w.gguf"]);
  expect(seen.some((u) => u.startsWith("https://hf-mirror.com/api/models/someone/hf-only/tree"))).toBe(true);
});

test("抠图权重：国内 HF 镜像打头，海外 GitHub 直连打头", async () => {
  const { sourcesFor, bgModelSpec } = await import("./bg-remove");
  const spec = bgModelSpec("u2netp")!;
  const cn = sourcesFor(spec, plan());
  expect(cn[0]).toBe("https://hf-mirror.com/tomjackson2023/rembg/resolve/main/u2netp.onnx");
  expect(cn).toContain("https://gh-proxy.example/https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx");
  expect(cn[cn.length - 1]).toBe("https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx");

  const global = sourcesFor(
    spec,
    plan({ mode: "global", githubPrefixes: [""], hfEndpoints: ["https://huggingface.co"] }),
  );
  expect(global).toEqual([
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
    "https://huggingface.co/tomjackson2023/rembg/resolve/main/u2netp.onnx",
  ]);
});

test("评测题库：IFEval 按 HF 端点顺序；通用题库 jsDelivr 打头、GitHub raw 走加速候选", async () => {
  const { evalMirrors } = await import("./eval");
  expect(evalMirrors("hf-ifeval", plan())).toEqual([
    "https://hf-mirror.com/datasets/google/IFEval/resolve/main/",
    "https://huggingface.co/datasets/google/IFEval/resolve/main/",
  ]);
  expect(
    evalMirrors("hf-ifeval", plan({ hfEndpoints: ["https://huggingface.co", "https://hf-mirror.com"] }))[0],
  ).toBe("https://huggingface.co/datasets/google/IFEval/resolve/main/");
  const data = evalMirrors("data", plan());
  expect(data[0]).toStartWith("https://cdn.jsdelivr.net/");
  expect(data.slice(1)).toEqual([
    "https://gh-proxy.example/https://raw.githubusercontent.com/jundot/omlx/main/omlx/eval/data/",
    "https://raw.githubusercontent.com/jundot/omlx/main/omlx/eval/data/",
  ]);
});

test("TTS git clone：国内 ModelScope（.git 规范地址）打头，海外 HF 官方打头；ModelScope 没有就不放", async () => {
  const { ttsCloneUrls, ttsCloneCommand } = await import("./tts-models");
  const repo = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice";
  expect(ttsCloneUrls(repo, plan(), repo)).toEqual([
    "https://www.modelscope.cn/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice.git",
    "https://hf-mirror.com/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
  ]);
  expect(
    ttsCloneUrls(repo, plan({ mode: "global", modelSource: "huggingface", hfEndpoints: ["https://huggingface.co"] }), repo),
  ).toEqual([
    "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "https://www.modelscope.cn/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice.git",
  ]);
  expect(ttsCloneUrls("mistralai/x", plan(), null)).toEqual([
    "https://hf-mirror.com/mistralai/x",
    "https://huggingface.co/mistralai/x",
  ]);
  // 卡死检测：低速阈值 + LFS 活动超时都要带上。
  const cmd = ttsCloneCommand("https://example/x", "/tmp/x").join(" ");
  expect(cmd).toContain("http.lowSpeedTime=");
  expect(cmd).toContain("lfs.activitytimeout=");
});
