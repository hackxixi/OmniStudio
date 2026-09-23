import { describe, expect, test } from "bun:test";

import type { SourcePlan } from "../shared/net-sources";
import {
  fetchLatestLlamaRelease,
  guessLlamaAssets,
  parseExpandedAssets,
  parseLlamaTagsFromRefs,
  planLlamaAsset,
} from "./engine-install";
import { fetchGithubJson, gitRemoteCandidates } from "./github-api";

/**
 * GitHub 元数据的多链路回退：API 经镜像、API 整个不可达时从 refs + release 页解析。
 * fetch 全部注入假实现（按 URL 路由），不联网。
 */
function planOf(mode: "cn" | "global"): SourcePlan {
  return {
    mode,
    decidedBy: "setting",
    cnLocale: mode === "cn",
    modelSource: "huggingface",
    hfEndpoints: ["https://huggingface.co"],
    pypiIndexes: ["https://pypi.org/simple"],
    githubPrefixes:
      mode === "cn"
        ? ["https://gh-proxy.com/", "https://ghfast.top/", ""]
        : ["", "https://gh-proxy.com/", "https://ghfast.top/"],
    homebrewEnv: {},
    probes: [],
    at: Date.now(),
  };
}

type Route = (url: string) => Response | Promise<Response> | null;

/** 假 fetch：按顺序问每个路由，第一个给出响应的算数；没人认领 → 网络错误。记下请求顺序。 */
function fakeFetch(...routes: Route[]): typeof fetch & { urls: string[] } {
  const urls: string[] = [];
  const fn = (async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    urls.push(url);
    for (const route of routes) {
      const res = await route(url);
      if (res) return res;
    }
    throw new TypeError("fetch failed");
  }) as unknown as typeof fetch & { urls: string[] };
  fn.urls = urls;
  return fn;
}

const API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15";
const RELEASES_JSON = [
  { tag_name: "v0.4.1", assets: [] },
  {
    tag_name: "b11124",
    assets: [
      { name: "llama-b11124-bin-macos-arm64.tar.gz", size: 12_000_000 },
      { name: "llama-b11124-bin-ubuntu-x64.tar.gz", size: 20_000_000 },
    ],
  },
];

describe("fetchGithubJson", () => {
  test("海外：直连在前，直连通了就只打一次（与原来行为一致）", async () => {
    const f = fakeFetch((url) => (url === API ? Response.json(RELEASES_JSON) : null));
    const json = await fetchGithubJson<unknown[]>(API, { plan: planOf("global"), fetchImpl: f, validate: Array.isArray });
    expect(json).toHaveLength(2);
    expect(f.urls).toEqual([API]);
  });

  test("国内：先走 gh-proxy；代理回了网页（ghfast 那种 403）就换下一条", async () => {
    const f = fakeFetch(
      (url) => (url.startsWith("https://gh-proxy.com/") ? new Response("<html>busy</html>", { status: 200 }) : null),
      (url) => (url.startsWith("https://ghfast.top/") ? new Response("Invalid input.", { status: 403 }) : null),
      (url) => (url === API ? Response.json(RELEASES_JSON) : null),
    );
    const json = await fetchGithubJson<unknown[]>(API, { plan: planOf("cn"), fetchImpl: f, validate: Array.isArray });
    expect(json).toHaveLength(2);
    expect(f.urls).toEqual([`https://gh-proxy.com/${API}`, `https://ghfast.top/${API}`, API]);
  });

  test("全部不通：错误里带每条链路的原因", async () => {
    const f = fakeFetch((url) => (url.startsWith("https://ghfast.top/") ? new Response("x", { status: 403 }) : null));
    await expect(fetchGithubJson(API, { plan: planOf("cn"), fetchImpl: f })).rejects.toThrow(/gh-proxy\.com.*ghfast\.top：HTTP 403.*直连/);
  });
});

describe("llama.cpp 最新构建：API 不可达时的兜底", () => {
  const REFS =
    "001e# service=git-upload-pack\n" +
    "003f1111 refs/tags/b11122\n003f2222 refs/tags/b11124\n003f3333 refs/tags/b11124^{}\n" +
    "003f4444 refs/tags/v0.4.1\n003f5555 refs/tags/b11123\n003f6666 refs/tags/b999-test\n";
  const EXPANDED =
    '<a href="/ggml-org/llama.cpp/releases/download/b11124/llama-b11124-bin-macos-arm64.tar.gz">x</a>' +
    '<a href="/ggml-org/llama.cpp/releases/download/b11124/cudart-llama-bin-win-cuda-12.4-x64.zip">y</a>' +
    '<a href="/ggml-org/llama.cpp/releases/download/b11124/llama-b11124-bin-macos-arm64.tar.gz">dup</a>';

  test("refs 解析：只要 b<构建号>，新到旧，去重、忽略 peeled 与杂项 tag", () => {
    expect(parseLlamaTagsFromRefs(REFS)).toEqual(["b11124", "b11123", "b11122"]);
  });

  test("资产页解析：去重，只取这个 tag 下的", () => {
    expect(parseExpandedAssets(EXPANDED, "b11124")).toEqual([
      "llama-b11124-bin-macos-arm64.tar.gz",
      "cudart-llama-bin-win-cuda-12.4-x64.zip",
    ]);
    expect(parseExpandedAssets(EXPANDED, "b1112")).toEqual([]);
  });

  test("API 正常：行为与原来一致（跳过不带二进制的 v0.x，带上 sizes）", async () => {
    const f = fakeFetch((url) => (url === API ? Response.json(RELEASES_JSON) : null));
    const release = await fetchLatestLlamaRelease(f, planOf("global"));
    expect(release?.tag).toBe("b11124");
    expect(release?.sizes["llama-b11124-bin-macos-arm64.tar.gz"]).toBe(12_000_000);
    expect(f.urls).toEqual([API]);
  });

  test("API 全挂 → refs 取最新 tag，资产页（经镜像）拿真实清单", async () => {
    const f = fakeFetch(
      (url) => (url.includes("api.github.com") ? new Response("rate limited", { status: 403 }) : null),
      (url) => (url.endsWith("llama.cpp.git/info/refs?service=git-upload-pack") ? new Response(REFS) : null),
      (url) =>
        url === "https://ghfast.top/https://github.com/ggml-org/llama.cpp/releases/expanded_assets/b11124"
          ? new Response(EXPANDED)
          : null,
    );
    const release = await fetchLatestLlamaRelease(f, planOf("cn"));
    expect(release).toEqual({
      tag: "b11124",
      assets: ["llama-b11124-bin-macos-arm64.tar.gz", "cudart-llama-bin-win-cuda-12.4-x64.zip"],
      sizes: {},
    });
    // refs 第一条链路（gh-proxy）就拿到了
    expect(f.urls).toContain("https://gh-proxy.com/https://github.com/ggml-org/llama.cpp.git/info/refs?service=git-upload-pack");
  });

  test("资产页也读不到 → 按命名规律拼出最新 tag 的基础包，装配规则认得出来", async () => {
    const f = fakeFetch(
      (url) => (url.endsWith("info/refs?service=git-upload-pack") ? new Response(REFS) : null),
      (url) => (url.includes("expanded_assets") ? new Response("nope", { status: 404 }) : null),
    );
    const release = await fetchLatestLlamaRelease(f, planOf("cn"));
    expect(release?.tag).toBe("b11124");
    expect(release?.assets).toEqual(guessLlamaAssets("b11124"));
    expect(planLlamaAsset(release!.assets, "darwin", "arm64")?.asset).toBe("llama-b11124-bin-macos-arm64.tar.gz");
    expect(planLlamaAsset(release!.assets, "linux", "x64", "amd")?.asset).toBe("llama-b11124-bin-ubuntu-vulkan-x64.tar.gz");
    expect(planLlamaAsset(release!.assets, "win32", "x64", "nvidia")?.asset).toBe("llama-b11124-bin-win-vulkan-x64.zip");
    // 只看最新两个 tag，不在读不到的资产页上反复等
    expect(f.urls.filter((u) => u.includes("expanded_assets/b11122"))).toHaveLength(0);
  });

  test("API 与 refs 都不通：抛 API 的错（界面提示检查网络）", async () => {
    const f = fakeFetch();
    await expect(fetchLatestLlamaRelease(f, planOf("global"))).rejects.toThrow(/GitHub API 不可达/);
  });
});

describe("git 远端候选", () => {
  test("GitHub https 仓库按计划带上前缀镜像；其它远端原样", () => {
    expect(gitRemoteCandidates("https://github.com/anthropics/skills.git", planOf("cn"))).toEqual([
      "https://gh-proxy.com/https://github.com/anthropics/skills.git",
      "https://ghfast.top/https://github.com/anthropics/skills.git",
      "https://github.com/anthropics/skills.git",
    ]);
    expect(gitRemoteCandidates("https://github.com/a/b.git", planOf("global"))[0]).toBe("https://github.com/a/b.git");
    expect(gitRemoteCandidates("git@github.com:a/b.git", planOf("cn"))).toEqual(["git@github.com:a/b.git"]);
    expect(gitRemoteCandidates("https://gitlab.com/a/b.git", planOf("cn"))).toEqual(["https://gitlab.com/a/b.git"]);
  });
});
