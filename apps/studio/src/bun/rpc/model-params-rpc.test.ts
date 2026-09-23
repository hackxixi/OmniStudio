import { afterEach, describe, expect, test } from "bun:test";

import { clearModelParamsForRpc, getModelParamsForRpc, setModelParamsForRpc } from "./model-params-rpc";

/** RPC 实现体：真设置库 + 真注册表（没有在跑的实例）。 */

const MODEL = "unsloth/Qwen3-8B-GGUF";

afterEach(() => {
  clearModelParamsForRpc(MODEL);
});

describe("model params RPC", () => {
  test("没存过：params = null，采样全部有值且标了来源，不需要重启", async () => {
    const res = await getModelParamsForRpc(` ${MODEL} `);
    expect(res.model).toBe(MODEL);
    expect(res.params).toBeNull();
    expect(res.needsRestart).toBe(false);
    for (const v of Object.values(res.sampling.values)) expect(Number.isFinite(v)).toBe(true);
    expect(Object.values(res.sampling.sources)).not.toContain("model-override");
  });

  test("set 校验后返回落库的那份；get 的采样把按模型的值标成 model-override；命令预览带上按模型参数", async () => {
    const set = setModelParamsForRpc(MODEL, { ctxSize: 12288, sampling: { temperature: 0.42 }, parallel: 999 });
    expect(set).toEqual({ ok: true, params: { ctxSize: 12288, sampling: { temperature: 0.42 } }, needsRestart: false });
    const res = await getModelParamsForRpc(MODEL);
    expect(res.params).toEqual({ ctxSize: 12288, sampling: { temperature: 0.42 } });
    expect(res.sampling.values.temperature).toBe(0.42);
    expect(res.sampling.sources.temperature).toBe("model-override");
    // HF 引用在默认引擎（llama.cpp）下走 -hf：命令里是按模型的窗口与温度
    expect(res.launchPreview).toContain("--ctx-size 12288");
    expect(res.launchPreview).toContain("--temp 0.42");
  });

  test("clear 后回到没存过；空 model 报错不抛", async () => {
    setModelParamsForRpc(MODEL, { parallel: 2 });
    expect(clearModelParamsForRpc(MODEL)).toEqual({ ok: true, needsRestart: false });
    expect((await getModelParamsForRpc(MODEL)).params).toBeNull();
    expect(setModelParamsForRpc("  ", { parallel: 2 }).ok).toBe(false);
  });
});
