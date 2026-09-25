import { afterAll, beforeEach, describe, expect, mock, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, utimesSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

// 设置表：展开真实模块再覆盖（mock-hygiene.test.ts 规矩）
const SETTINGS: Record<string, string> = {};
const realSettings = await import("./db/settings");
mock.module("./db/settings", () => ({
  ...realSettings,
  getSetting: (key: string) => SETTINGS[key] ?? "",
}));

const {
  clearSamplingCache,
  parseGenerationConfig,
  refreshSamplingMetadata,
  resolveSampling,
  samplingModelName,
} = await import("./model-sampling");

const root = mkdtempSync(join(tmpdir(), "model-sampling-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));

// ---------- 合成 GGUF ----------
const u32 = (v: number) => {
  const b = new Uint8Array(4);
  new DataView(b.buffer).setUint32(0, v, true);
  return b;
};
const u64 = (v: number) => new Uint8Array([...u32(v), ...u32(0)]);
const f32 = (v: number) => {
  const b = new Uint8Array(4);
  new DataView(b.buffer).setFloat32(0, v, true);
  return b;
};
const str = (s: string) => {
  const b = new TextEncoder().encode(s);
  return new Uint8Array([...u64(b.length), ...b]);
};
function gguf(entries: [string, number, Uint8Array][]): Uint8Array {
  const body = entries.flatMap(([k, t, v]) => [...str(k), ...u32(t), ...v]);
  return new Uint8Array([...new TextEncoder().encode("GGUF"), ...u32(3), ...u64(0), ...u64(entries.length), ...body]);
}

function writeDir(name: string, files: Record<string, string | Uint8Array>): string {
  const dir = join(root, name);
  mkdirSync(dir, { recursive: true });
  for (const [f, content] of Object.entries(files)) writeFileSync(join(dir, f), content);
  return dir;
}

beforeEach(() => {
  for (const k of Object.keys(SETTINGS)) delete SETTINGS[k];
  clearSamplingCache();
});

describe("resolveSampling", () => {
  test("认不出的模型：全局设置；设置缺失 / 写坏落回中性默认", () => {
    SETTINGS.SERVER_TEMP = "0.3";
    SETTINGS.SERVER_TOP_K = "abc";
    const r = resolveSampling("someone/my-finetune");
    expect(r.preset).toBeNull();
    expect(r.values).toEqual({
      temperature: 0.3,
      topP: 0.9,
      topK: 40,
      minP: 0.05,
      presencePenalty: 0,
      repeatPenalty: 1.0,
    });
    expect(new Set(Object.values(r.sources))).toEqual(new Set(["global"]));
  });

  test("家族表压过全局；表里没写的字段落回全局", () => {
    SETTINGS.SERVER_TEMP = "0.2";
    SETTINGS.SERVER_MIN_P = "0.1";
    const r = resolveSampling("/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf");
    expect(r.preset).toBe("llama-3");
    expect(r.values.temperature).toBe(0.6);
    expect(r.sources.temperature).toBe("family-preset");
    expect(r.values.minP).toBe(0.1);
    expect(r.sources.minP).toBe("global");
  });

  test("思考模式挑变体：Qwen3 off → 0.7 / 0.8，on / auto → 0.6 / 0.95", () => {
    const off = resolveSampling("unsloth/Qwen3-8B-GGUF:Q4_K_M", { thinking: "off" });
    expect(off.preset).toBe("qwen3");
    expect([off.values.temperature, off.values.topP, off.values.topK, off.values.minP]).toEqual([0.7, 0.8, 20, 0]);
    const auto = resolveSampling("unsloth/Qwen3-8B-GGUF:Q4_K_M");
    expect([auto.values.temperature, auto.values.topP]).toEqual([0.6, 0.95]);
  });

  test("generation_config.json 压过家族表；按模型设置压过一切", () => {
    const dir = writeDir("Qwen__Qwen2.5-7B-Instruct", {
      "config.json": "{}",
      "generation_config.json": JSON.stringify({ temperature: 0.5, top_p: 0.7, repetition_penalty: 1.1, do_sample: true }),
    });
    const r = resolveSampling(dir);
    expect(r.preset).toBe("qwen2.5");
    expect(r.values.temperature).toBe(0.5);
    expect(r.sources.temperature).toBe("model-metadata");
    expect(r.values.topK).toBe(20); // 表里有、config 没写
    expect(r.sources.topK).toBe("family-preset");
    expect(r.values.repeatPenalty).toBe(1.1);

    const o = resolveSampling(dir, { override: { temperature: 0.9 } });
    expect(o.values.temperature).toBe(0.9);
    expect(o.sources.temperature).toBe("model-override");
    expect(o.sources.topP).toBe("model-metadata");
  });

  test("显式思考模式且表里单列了该变体时，变体压过模型自带的默认档", () => {
    // Qwen3 的 generation_config 写的是思考档
    const dir = writeDir("Qwen3-8B", {
      "generation_config.json": JSON.stringify({ temperature: 0.6, top_p: 0.95, top_k: 20 }),
      "model.safetensors": "",
    });
    const off = resolveSampling(dir, { thinking: "off" });
    expect(off.values.temperature).toBe(0.7);
    expect(off.sources.temperature).toBe("family-preset");
    const auto = resolveSampling(dir);
    expect(auto.sources.temperature).toBe("model-metadata");
  });

  test("generation_config 按 mtime 缓存：文件改了会重新读", () => {
    const dir = writeDir("gen-cache-model", { "generation_config.json": JSON.stringify({ temperature: 0.4 }) });
    expect(resolveSampling(dir).values.temperature).toBe(0.4);
    writeFileSync(join(dir, "generation_config.json"), JSON.stringify({ temperature: 0.45 }));
    const later = new Date(Date.now() + 5000);
    utimesSync(join(dir, "generation_config.json"), later, later);
    expect(resolveSampling(dir).values.temperature).toBe(0.45);
  });

  test("GGUF general.sampling.* 预热后生效，且比同目录 generation_config 更优先", async () => {
    const file = join(
      writeDir("gguf-repo", {
        "generation_config.json": JSON.stringify({ temperature: 0.5, top_p: 0.5 }),
        "weights-Q4_K_M.gguf": gguf([
          ["general.architecture", 8, str("llama")],
          ["general.name", 8, str("Gemma 3 4B It")],
          ["general.sampling.temp", 6, f32(0.25)],
        ]),
      }),
      "weights-Q4_K_M.gguf",
    );
    await refreshSamplingMetadata(file);
    const r = resolveSampling(file);
    expect(r.values.temperature).toBeCloseTo(0.25, 5);
    expect(r.sources.temperature).toBe("model-metadata");
    expect(r.values.topP).toBe(0.5); // 来自 generation_config
    // 文件名认不出，靠 GGUF 里的 general.name 认成 Gemma
    expect(r.preset).toBe("gemma");
    expect(r.values.topK).toBe(64);
  });

  test("HF 缓存 snapshot 目录：用 models--org--repo 认家族", () => {
    const dir = join(root, "hub", "models--google--gemma-3-4b-it", "snapshots", "abc123");
    mkdirSync(dir, { recursive: true });
    expect(resolveSampling(dir).preset).toBe("gemma");
  });
});

describe("parseGenerationConfig / samplingModelName", () => {
  test("字段映射与越界过滤（top_k -1 = 不截断 → 0）", () => {
    expect(
      parseGenerationConfig({
        temperature: 0.6,
        top_p: 0.95,
        top_k: -1,
        min_p: 0,
        presence_penalty: 1.5,
        repetition_penalty: 1.05,
        max_new_tokens: 100,
      }),
    ).toEqual({ temperature: 0.6, topP: 0.95, topK: 0, minP: 0, presencePenalty: 1.5, repeatPenalty: 1.05 });
    expect(parseGenerationConfig({ temperature: -1, top_p: 2 })).toEqual({});
    expect(parseGenerationConfig(null)).toEqual({});
  });

  test("展示名去掉扩展名 / 量化后缀 / 分片号", () => {
    expect(samplingModelName("/m/Qwen3-8B-Q4_K_M.gguf")).toBe("Qwen3-8B");
    expect(samplingModelName("/m/GLM-4.6-UD-Q3_K_M-00001-of-00004.gguf")).toBe("GLM-4.6");
    expect(samplingModelName("unsloth/gpt-oss-20b-GGUF:F16")).toBe("gpt-oss-20b-GGUF");
    expect(samplingModelName("/models/Qwen__Qwen3.5-4B")).toBe("Qwen3.5-4B");
  });
});
