import { describe, expect, test } from "bun:test";

import { lowerKvType, nextDegradeStep, type DegradeStepKind, type MemoryKnobs } from "./llama-degrade";

/** 一份中性旋钮：32K、f16、全部层在 GPU（999）、认 --fit、48 层。 */
function knobs(over: Partial<MemoryKnobs> = {}): MemoryKnobs {
  return {
    ctx: 32768,
    ctxPinned: false,
    cacheTypeK: "f16",
    cacheTypeV: "f16",
    gpuLayers: 999,
    gpuLayersPinned: false,
    nCpuMoe: null,
    fit: "off",
    blockCount: 48,
    fitSupported: true,
    flashAttnOff: false,
    minCtx: 4096,
    ...over,
  };
}

describe("lowerKvType", () => {
  test("f16 / 未设 → q8_0 → q4_0 → 不再降", () => {
    expect(lowerKvType("f16")).toBe("q8_0");
    expect(lowerKvType(null)).toBe("q8_0");
    expect(lowerKvType("q8_0")).toBe("q4_0");
    expect(lowerKvType("q5_1")).toBe("q4_0");
    expect(lowerKvType("q4_0")).toBeNull();
  });
});

describe("nextDegradeStep", () => {
  test("顺序：先上下文减半 → 再 KV 降一档 → 再交给 --fit on", () => {
    const used = new Set<DegradeStepKind>();
    const s1 = nextDegradeStep(knobs(), {}, used)!;
    expect(s1.kind).toBe("ctx");
    expect(s1.adjust).toEqual({ ctxCap: 16384 });
    expect(s1.summary).toContain("32768 → 16384");
    used.add(s1.kind);

    const s2 = nextDegradeStep(knobs({ ctx: 16384 }), s1.adjust, used)!;
    expect(s2.kind).toBe("kv");
    expect(s2.adjust).toEqual({ ctxCap: 16384, cacheTypeK: "q8_0", cacheTypeV: "q8_0" });
    used.add(s2.kind);

    const s3 = nextDegradeStep(knobs({ ctx: 16384, cacheTypeK: "q8_0", cacheTypeV: "q8_0" }), s2.adjust, used)!;
    expect(s3.kind).toBe("gpu");
    expect(s3.adjust.fit).toBe("on");
    expect(s3.adjust.ctxCap).toBe(16384); // 前面的调整保留（累加）
    expect(s3.summary).toContain("--fit on");
  });

  test("三步都用过了还能退 → 从头再挑（上下文再减半）", () => {
    const used = new Set<DegradeStepKind>(["ctx", "kv", "gpu"]);
    const s = nextDegradeStep(knobs({ ctx: 16384 }), { ctxCap: 16384 }, used)!;
    expect(s.kind).toBe("ctx");
    expect(s.adjust.ctxCap).toBe(8192);
  });

  test("上下文不低于下限；已到下限就跳过这一步", () => {
    expect(nextDegradeStep(knobs({ ctx: 6000 }), {}, new Set())!.adjust.ctxCap).toBe(4096);
    const s = nextDegradeStep(knobs({ ctx: 4096 }), {}, new Set())!;
    expect(s.kind).toBe("kv");
  });

  test("用户固定了上下文：照样临时调小，但说明里写清楚不改设置", () => {
    const s = nextDegradeStep(knobs({ ctxPinned: true }), {}, new Set())!;
    expect(s.kind).toBe("ctx");
    expect(s.summary).toContain("固定了上下文");
  });

  test("FA 关着：只降 K，不碰 V（llama.cpp 拒绝没有 FA 的量化 V）", () => {
    const s = nextDegradeStep(knobs({ ctx: 4096, flashAttnOff: true }), {}, new Set())!;
    expect(s.kind).toBe("kv");
    expect(s.adjust.cacheTypeK).toBe("q8_0");
    expect(s.adjust.cacheTypeV).toBeUndefined();
  });

  test("不认 --fit → GPU 层数砍到 75%（没发层数时按模型层数算）", () => {
    const k = knobs({ ctx: 4096, cacheTypeK: "q4_0", cacheTypeV: "q4_0", fitSupported: false, fit: null, gpuLayers: null });
    const s = nextDegradeStep(k, {}, new Set())!;
    expect(s.kind).toBe("gpu");
    expect(s.adjust.gpuLayers).toBe(36); // 48 × 0.75
    const s2 = nextDegradeStep({ ...k, gpuLayers: 36 }, s.adjust, new Set(["gpu"]))!;
    expect(s2.adjust.gpuLayers).toBe(27);
  });

  test("用户固定了层数：不交给 --fit（那会推翻用户的层数），按 75% 临时调小并说明", () => {
    const k = knobs({ ctx: 4096, cacheTypeK: "q4_0", cacheTypeV: "q4_0", gpuLayers: 40, gpuLayersPinned: true, fit: null });
    const s = nextDegradeStep(k, {}, new Set())!;
    expect(s.adjust.fit).toBeUndefined();
    expect(s.adjust.gpuLayers).toBe(30);
    expect(s.summary).toContain("固定");
  });

  test("MoE 专家下放模式：GPU 这一步是多放几层专家，而不是砍层数", () => {
    const k = knobs({ ctx: 4096, cacheTypeK: "q4_0", cacheTypeV: "q4_0", nCpuMoe: 10 });
    const s = nextDegradeStep(k, { nCpuMoe: 10 }, new Set())!;
    expect(s.kind).toBe("gpu");
    expect(s.adjust.nCpuMoe).toBe(22); // + ceil(48/4)
    expect(s.adjust.gpuLayers).toBeUndefined();
  });

  test("一步都退不了 → null（调用方按原错误报失败）", () => {
    const k = knobs({
      ctx: 4096,
      cacheTypeK: "q4_0",
      cacheTypeV: "q4_0",
      fit: "on",
      gpuLayers: null,
      blockCount: null,
    });
    expect(nextDegradeStep(k, { fit: "on" }, new Set())).toBeNull();
  });
});
