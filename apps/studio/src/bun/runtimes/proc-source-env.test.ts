import { afterEach, expect, test } from "bun:test";

import type { SourcePlan } from "../../shared/net-sources";
import { mockModulePartial } from "../test-mocks";

/**
 * 推理引擎子进程的下载源环境：环境合并顺序、HF 端点跟随路由、
 * VLLM_USE_MODELSCOPE / SGLANG_USE_MODELSCOPE 只在条件全满足时才打开、MLX 端点的「显式」判定。
 * 路由与 ModelScope 查询都换成假的（真实实现要联网）。
 */

function plan(over: Partial<SourcePlan> = {}): SourcePlan {
  return {
    mode: "cn",
    decidedBy: "setting",
    cnLocale: true,
    modelSource: "modelscope",
    hfEndpoints: ["https://hf-mirror.com", "https://huggingface.co"],
    pypiIndexes: [],
    githubPrefixes: [""],
    homebrewEnv: {},
    probes: [],
    at: 0,
    ...over,
  };
}

let currentPlan = plan();
// specifier 相对 src/bun（mockModulePartial 在 test-mocks.ts 里解析），见 test-mocks.ts 注释。
await mockModulePartial<typeof import("../net-sources")>("./net-sources", {
  getSourcePlan: async () => currentPlan,
  peekSourcePlan: () => currentPlan,
  sourceEnv: (p: SourcePlan) => ({ HF_ENDPOINT: p.hfEndpoints[0]!, MODEL_ENDPOINT: p.hfEndpoints[0]! }),
});

let onModelScope = new Set<string>(["Qwen/Qwen3.5-4B"]);
await mockModulePartial<typeof import("../model-source-map")>("./model-source-map", {
  resolveModelScopeRepo: async (repo: string) =>
    onModelScope.has(repo) ? { status: "found" as const, repo } : { status: "missing" as const },
});

const { downloadSourceEnv, mergeChildEnv, OMNI_HF_ENDPOINTS_ENV } = await import("./proc");
const { explicitMlxHfEndpoint } = await import("./mlx");

/** 「装了 modelscope 的 python」：`true` 对任何参数都退出 0；`false` 反之。 */
const PY_WITH = Bun.which("true")!;
const PY_WITHOUT = Bun.which("false")!;

const savedHf = process.env.HF_ENDPOINT;
afterEach(() => {
  currentPlan = plan();
  onModelScope = new Set(["Qwen/Qwen3.5-4B"]);
  if (savedHf === undefined) delete process.env.HF_ENDPOINT;
  else process.env.HF_ENDPOINT = savedHf;
});

test("子进程环境合并：父进程 < 代理 < 调用方；undefined 丢掉", () => {
  const env = mergeChildEnv(
    { PATH: "/bin", HF_ENDPOINT: "https://from-parent", EMPTY: undefined },
    { HTTPS_PROXY: "http://127.0.0.1:7890", HF_ENDPOINT: "https://from-proxy" },
    { HF_ENDPOINT: "https://hf-mirror.com" },
  );
  expect(env).toEqual({
    PATH: "/bin",
    HTTPS_PROXY: "http://127.0.0.1:7890",
    HF_ENDPOINT: "https://hf-mirror.com",
  });
  expect("EMPTY" in env).toBe(false);
});

test("HF 端点跟随路由；端点顺序经 OMNI_HF_ENDPOINTS 交给 python 脚本", async () => {
  delete process.env.HF_ENDPOINT;
  const cn = await downloadSourceEnv();
  expect(cn.HF_ENDPOINT).toBe("https://hf-mirror.com");
  expect(cn[OMNI_HF_ENDPOINTS_ENV]).toBe("https://hf-mirror.com,https://huggingface.co");

  currentPlan = plan({ mode: "global", modelSource: "huggingface", hfEndpoints: ["https://huggingface.co"] });
  const global = await downloadSourceEnv();
  expect(global.HF_ENDPOINT).toBe("https://huggingface.co");
});

test("用户自己在环境里设过 HF_ENDPOINT → 不覆盖", async () => {
  process.env.HF_ENDPOINT = "https://my-own-mirror";
  const env = await downloadSourceEnv();
  expect("HF_ENDPOINT" in env).toBe(false);
});

test("VLLM_USE_MODELSCOPE：国内 + 仓库原样在 ModelScope + python 装了 modelscope 才打开", async () => {
  const on = await downloadSourceEnv({ model: "Qwen/Qwen3.5-4B", python: PY_WITH, modelScopeVar: "VLLM_USE_MODELSCOPE" });
  expect(on.VLLM_USE_MODELSCOPE).toBe("True");

  // python 没装 modelscope → 照旧 HF 镜像。
  const noPkg = await downloadSourceEnv({ model: "Qwen/Qwen3.5-4B", python: PY_WITHOUT, modelScopeVar: "VLLM_USE_MODELSCOPE" });
  expect(noPkg.VLLM_USE_MODELSCOPE).toBeUndefined();

  // ModelScope 上没有（或只有改名组织的版本）→ 不打开。
  const hfOnly = await downloadSourceEnv({ model: "someone/hf-only", python: PY_WITH, modelScopeVar: "SGLANG_USE_MODELSCOPE" });
  expect(hfOnly.SGLANG_USE_MODELSCOPE).toBeUndefined();

  // 本地路径 → 不打开。
  const local = await downloadSourceEnv({ model: "/models/qwen", python: PY_WITH, modelScopeVar: "VLLM_USE_MODELSCOPE" });
  expect(local.VLLM_USE_MODELSCOPE).toBeUndefined();

  // 海外（默认平台 HF）→ 不打开。
  currentPlan = plan({ mode: "global", modelSource: "huggingface" });
  const global = await downloadSourceEnv({ model: "Qwen/Qwen3.5-4B", python: PY_WITH, modelScopeVar: "VLLM_USE_MODELSCOPE" });
  expect(global.VLLM_USE_MODELSCOPE).toBeUndefined();
});

test("MLX_HF_ENDPOINT：出厂默认值 = 跟随路由；改过的尊重用户，置空 = 官方", () => {
  expect(explicitMlxHfEndpoint("https://hf-mirror.com")).toBeNull();
  expect(explicitMlxHfEndpoint("https://hf-mirror.com/")).toBeNull();
  expect(explicitMlxHfEndpoint(undefined)).toBeNull();
  expect(explicitMlxHfEndpoint("https://my.mirror")).toBe("https://my.mirror");
  expect(explicitMlxHfEndpoint("")).toBe("https://huggingface.co");
});
