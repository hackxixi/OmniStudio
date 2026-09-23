import { describe, expect, test } from "bun:test";
import { SAMPLING_FIELDS } from "./model-params";
import { SAMPLING_PRESETS, matchSamplingPreset, matchSamplingPresetByArch, presetVariant } from "./sampling-presets";

const id = (name: string) => matchSamplingPreset(name)?.id ?? null;

describe("matchSamplingPreset", () => {
  test("最长匹配胜：Coder / 3.5 / 2507 不被 qwen3 吃掉", () => {
    expect(id("Qwen3-8B-Q4_K_M")).toBe("qwen3");
    expect(id("Qwen3-Coder-30B-A3B-Instruct")).toBe("qwen3-coder");
    expect(id("Qwen3.5-4B")).toBe("qwen3.5");
    expect(id("Qwen3.6-35B-A3B-UD-Q4_K_XL")).toBe("qwen3.6");
    expect(id("Qwen3-30B-A3B-Instruct-2507")).toBe("qwen3-instruct-2507");
    expect(id("Qwen3-4B-Thinking-2507")).toBe("qwen3-thinking-2507");
    expect(id("Qwen3-VL-8B-Thinking")).toBe("qwen3-vl-thinking");
    expect(id("Qwen3-VL-8B-Instruct")).toBe("qwen3-vl");
  });

  test("大小写 / 下划线 / 空格不敏感", () => {
    expect(id("QWEN3_CODER_30B")).toBe("qwen3-coder");
    expect(id("gemma 3 27b it")).toBe("gemma");
    expect(id("gemma-4-26B-A4B-it")).toBe("gemma");
  });

  test("常见家族都能认出来", () => {
    expect(id("DeepSeek-R1-Distill-Qwen-7B")).toBe("deepseek-r1");
    expect(id("DeepSeek-R1-0528-Qwen3-8B")).toBe("deepseek-r1");
    expect(id("DeepSeek-V3-0324")).toBe("deepseek-v3-0324");
    expect(id("DeepSeek-V3.1-Terminus")).toBe("deepseek-v3");
    expect(id("Meta-Llama-3.1-8B-Instruct")).toBe("llama-3");
    expect(id("Llama-4-Scout-17B-16E")).toBe("llama-4");
    expect(id("Mistral-Small-3.2-24B-Instruct-2506")).toBe("mistral-small");
    expect(id("Devstral-Small-2507")).toBe("devstral");
    expect(id("Magistral-Small-2509")).toBe("magistral");
    expect(id("phi-4")).toBe("phi-4");
    expect(id("Phi-4-mini-reasoning")).toBe("phi-4-reasoning");
    expect(id("GLM-4.6")).toBe("glm");
    expect(id("gpt-oss-20b-MXFP4")).toBe("gpt-oss");
    expect(id("Kimi-K2-Instruct")).toBe("kimi");
    expect(id("Kimi-K2-Thinking")).toBe("kimi-k2-thinking");
    expect(id("SmolLM3-3B")).toBe("smollm3");
    expect(id("MiniMax-M2")).toBe("minimax");
    expect(id("granite-4.0-h-small")).toBe("granite-4");
    expect(id("QwQ-32B")).toBe("qwq");
    expect(id("Qwen2.5-7B-Instruct")).toBe("qwen2.5");
  });

  test("OCR 档案保留原值", () => {
    for (const name of ["LightOnOCR-2-1B-bbox-soup", "chandra-ocr-2", "GLM-OCR"]) {
      const m = matchSamplingPreset(name);
      expect(m).not.toBeNull();
      expect(presetVariant(m!.preset)).toEqual({ temperature: 0.2, topP: 0.9, topK: 40, repeatPenalty: 1.12 });
    }
    expect(id("GLM-OCR")).toBe("glmocr");
  });

  test("认不出 → null", () => {
    expect(matchSamplingPreset("my-finetune")).toBeNull();
    expect(matchSamplingPreset("")).toBeNull();
  });
});

describe("presetVariant", () => {
  test("Qwen3 按思考模式取不同档，auto = 思考档", () => {
    const qwen3 = SAMPLING_PRESETS.qwen3!;
    expect(presetVariant(qwen3, "on")).toMatchObject({ temperature: 0.6, topP: 0.95, topK: 20, minP: 0 });
    expect(presetVariant(qwen3, "off")).toMatchObject({ temperature: 0.7, topP: 0.8, topK: 20, minP: 0 });
    expect(presetVariant(qwen3, "auto")).toEqual(presetVariant(qwen3, "on"));
  });

  test("Qwen3.5 非思考档带 presence_penalty 1.5", () => {
    expect(presetVariant(SAMPLING_PRESETS["qwen3.5"]!, "off").presencePenalty).toBe(1.5);
  });

  test("单模式模型：on / off 都回落到 default", () => {
    const r1 = SAMPLING_PRESETS["deepseek-r1"]!;
    expect(presetVariant(r1, "off")).toEqual(presetVariant(r1, "on"));
    expect(presetVariant(r1, "auto").temperature).toBe(0.6);
  });

  test("表里每条预设至少有一个变体，字段都是合法字段名", () => {
    for (const [pid, preset] of Object.entries(SAMPLING_PRESETS)) {
      const variants = [preset.default, preset.thinking, preset.nonThinking].filter(Boolean);
      expect(variants.length, pid).toBeGreaterThan(0);
      for (const v of variants) {
        for (const k of Object.keys(v!)) expect(SAMPLING_FIELDS as readonly string[]).toContain(k);
      }
      for (const p of preset.patterns) {
        for (const s of typeof p === "string" ? [p] : p) expect(s).toBe(s.toLowerCase());
      }
    }
  });
});

describe("matchSamplingPresetByArch", () => {
  test("微调改了名也能按架构认出底座（Qwopus3.5 → qwen35 → Qwen3.5）", () => {
    expect(matchSamplingPresetByArch("qwen35")?.id).toBe("qwen3.5");
    expect(matchSamplingPresetByArch("Gemma3")?.id).toBe("gemma");
  });
  test("多家族共用的架构不猜", () => {
    expect(matchSamplingPresetByArch("llama")).toBeNull();
    expect(matchSamplingPresetByArch("deepseek2")).toBeNull();
    expect(matchSamplingPresetByArch(null)).toBeNull();
  });
});
