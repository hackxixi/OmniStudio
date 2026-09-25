import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";

import {
  cachedLaunchPlan,
  clearLaunchPlanCache,
  pairedMmprojPath,
  refreshLaunchPlan,
  type LaunchPlanKey,
} from "./launch-plan";

// ---------- 测试辅助：合成 GGUF 字节流（复制自 gguf-meta.test.ts，不从 .test.ts import） ----------

const u32 = (v: number) => new Uint8Array([v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >>> 24) & 0xff]);
const u64 = (v: number | bigint) => {
  const n = typeof v === "bigint" ? v : BigInt(v);
  const lo = Number(n & 0xffffffffn);
  const hi = Number((n >> 32n) & 0xffffffffn);
  return new Uint8Array([...u32(lo), ...u32(hi)]);
};
const str = (s: string) => {
  const b = new TextEncoder().encode(s);
  return new Uint8Array([...u64(b.length), ...b]);
};

type KV = { key: string; type: number; value: Uint8Array };

function buildGguf(kvEntries: KV[]): Uint8Array {
  const magic = new TextEncoder().encode("GGUF");
  const body = kvEntries.flatMap((kv) => [str(kv.key), u32(kv.type), kv.value]);
  return new Uint8Array([
    ...magic,
    ...u32(3),
    ...u64(0),
    ...u64(kvEntries.length),
    ...body.flatMap((b) => Array.from(b)),
  ]);
}

function kvStr(key: string, value: string): KV {
  return { key, type: 8, value: str(value) };
}

function kvU32(key: string, value: number): KV {
  return { key, type: 4, value: u32(value) };
}

// ---------- ----------

let dir: string;

beforeAll(async () => {
  dir = await mkdtemp(join(tmpdir(), "launch-plan-test-"));
  clearLaunchPlanCache();
});

afterAll(async () => {
  await rm(dir, { recursive: true, force: true });
  clearLaunchPlanCache();
});

async function writeGguf(name: string): Promise<string> {
  const bytes = buildGguf([
    kvStr("general.architecture", "llama"),
    kvU32("llama.block_count", 32),
    kvU32("llama.attention.head_count", 32),
    kvU32("llama.attention.head_count_kv", 8),
    kvU32("llama.embedding_length", 4096),
    kvU32("llama.context_length", 8192),
  ]);
  const p = join(dir, name);
  await writeFile(p, bytes);
  return p;
}

function baseKey(modelPath: string): LaunchPlanKey {
  return {
    modelPath,
    parallel: 1,
    ubatch: null,
    batch: null,
    cacheTypeK: null,
    cacheTypeV: null,
    ctxOverride: null,
    flashAttn: null,
    supportsKvUnified: true,
    mmprojBytes: null,
  };
}

describe("launch-plan", () => {
  test("refreshLaunchPlan 算出计划、cachedLaunchPlan 同 key 命中、改 key 未命中", async () => {
    const p = await writeGguf("a.gguf");
    const key = baseKey(p);

    const plan = await refreshLaunchPlan(key);
    expect(plan).not.toBeNull();
    if (plan === null) return;
    // 元数据里 context_length=8192，自动路径
    expect(plan.ctxTokens).toBe(8192);

    // 同 key 命中缓存
    const cached = cachedLaunchPlan(key);
    expect(cached).not.toBeNull();
    if (cached === null) return;
    expect(cached.ctxTokens).toBe(plan.ctxTokens);
    expect(cached.parallel).toBe(plan.parallel);

    // 改 key（改 ctxOverride）→ 未命中
    const other = { ...key, ctxOverride: 16384 };
    expect(cachedLaunchPlan(other)).toBeNull();
  });

  test("非 GGUF 文件 → 返回 null，不抛异常", async () => {
    const p = join(dir, "notgguf.gguf");
    await writeFile(p, new Uint8Array(32).fill(7));
    const plan = await refreshLaunchPlan(baseKey(p));
    expect(plan).toBeNull();
  });

  test("缓存超过 4 条时最旧的被淘汰（LRU）", async () => {
    const p1 = await writeGguf("lru1.gguf");
    const p2 = await writeGguf("lru2.gguf");
    const p3 = await writeGguf("lru3.gguf");
    const p4 = await writeGguf("lru4.gguf");
    const p5 = await writeGguf("lru5.gguf");

    const k1 = baseKey(p1);
    const k2 = baseKey(p2);
    const k3 = baseKey(p3);
    const k4 = baseKey(p4);
    const k5 = baseKey(p5);

    // 灌 5 条，k1 最先入队 → 被 k5 挤出去
    await refreshLaunchPlan(k1);
    await refreshLaunchPlan(k2);
    await refreshLaunchPlan(k3);
    await refreshLaunchPlan(k4);
    await refreshLaunchPlan(k5);

    expect(cachedLaunchPlan(k1)).toBeNull(); // 最旧被淘汰
    expect(cachedLaunchPlan(k2)).not.toBeNull();
    expect(cachedLaunchPlan(k3)).not.toBeNull();
    expect(cachedLaunchPlan(k4)).not.toBeNull();
    expect(cachedLaunchPlan(k5)).not.toBeNull();
  });

  test("supportsKvUnified=false + parallel>1 → 计划按 slot 均分窗口，且与支持时是两条缓存", async () => {
    const p = await writeGguf("kvu.gguf");
    const unified = await refreshLaunchPlan({ ...baseKey(p), parallel: 2, supportsKvUnified: true });
    const split = await refreshLaunchPlan({ ...baseKey(p), parallel: 2, supportsKvUnified: false });
    expect(unified?.kvUnified).toBe(true);
    expect(unified?.ctxPerSlot).toBe(unified?.ctxTokens);
    expect(split?.kvUnified).toBe(false);
    expect(split?.ctxPerSlot).toBe(Math.floor((split?.ctxTokens ?? 0) / 2));
    expect(split?.reasons.map((r) => r.code)).toContain("kv.split-per-slot");
    expect(cachedLaunchPlan({ ...baseKey(p), parallel: 2, supportsKvUnified: true })?.kvUnified).toBe(true);
  });

  test("mmprojBytes 计入预算（totalBytes 变大），且进缓存 key", async () => {
    const p = await writeGguf("mm.gguf");
    const plain = await refreshLaunchPlan(baseKey(p));
    const withMm = await refreshLaunchPlan({ ...baseKey(p), mmprojBytes: 512 * 1024 * 1024 });
    expect(plain).not.toBeNull();
    expect(withMm).not.toBeNull();
    expect(withMm!.estimates.totalBytes).toBeGreaterThan(plain!.estimates.totalBytes);
    expect(cachedLaunchPlan(baseKey(p))?.estimates.totalBytes).toBe(plain!.estimates.totalBytes);
  });

  test("pairedMmprojPath：优先 f16（bf16 不误中）、无 f16 取字典序首个、无投影 null、目录入参也行", async () => {
    const d = join(dir, "pair");
    await mkdir(d, { recursive: true });
    await writeFile(join(d, "model.gguf"), "gguf");
    expect(pairedMmprojPath(join(d, "model.gguf"))).toBeNull();
    await writeFile(join(d, "mmproj-bf16.gguf"), "gguf");
    expect(pairedMmprojPath(join(d, "model.gguf"))).toBe(join(d, "mmproj-bf16.gguf"));
    await writeFile(join(d, "mmproj-F16.gguf"), "gguf");
    expect(pairedMmprojPath(join(d, "model.gguf"))).toBe(join(d, "mmproj-F16.gguf"));
    expect(pairedMmprojPath(d)).toBe(join(d, "mmproj-F16.gguf"));
    expect(pairedMmprojPath(join(dir, "no-such-dir", "x.gguf"))).toBeNull();
  });

  test("pairedMmprojPath：平铺目录里有不相干的模型时不配对，同一模型的多个量化照配", async () => {
    const flat = join(dir, "flat");
    await mkdir(flat, { recursive: true });
    await writeFile(join(flat, "Qwen3-VL-8B-Instruct-Q4_K_M.gguf"), "gguf");
    await writeFile(join(flat, "mmproj-F16.gguf"), "gguf");
    await writeFile(join(flat, "Llama-3.2-3B-Instruct-Q8_0.gguf"), "gguf");
    expect(pairedMmprojPath(join(flat, "Llama-3.2-3B-Instruct-Q8_0.gguf"))).toBeNull();

    const repo = join(dir, "repo");
    await mkdir(repo, { recursive: true });
    await writeFile(join(repo, "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"), "gguf");
    await writeFile(join(repo, "Qwen3-VL-8B-Instruct-Q8_0.gguf"), "gguf");
    await writeFile(join(repo, "Qwen3-VL-8B-Instruct-BF16-00001-of-00002.gguf"), "gguf");
    await writeFile(join(repo, "Qwen3-VL-8B-Instruct-BF16-00002-of-00002.gguf"), "gguf");
    await writeFile(join(repo, "mmproj-F16.gguf"), "gguf");
    expect(pairedMmprojPath(join(repo, "Qwen3-VL-8B-Instruct-Q8_0.gguf"))).toBe(join(repo, "mmproj-F16.gguf"));
  });

  test("HF 缓存式符号链接（snapshots/x.gguf → blobs/…）算完就能命中，不被当成「文件换过」", async () => {
    const blob = await writeGguf("blob-abc");
    const link = join(dir, "snapshot-model.gguf");
    await symlink(blob, link);
    const key = baseKey(link);
    await refreshLaunchPlan(key);
    expect(cachedLaunchPlan(key)).not.toBeNull();
  });

  test("clearLaunchPlanCache 清空缓存", async () => {
    const p = await writeGguf("clr.gguf");
    const key = baseKey(p);
    await refreshLaunchPlan(key);
    expect(cachedLaunchPlan(key)).not.toBeNull();
    clearLaunchPlanCache();
    expect(cachedLaunchPlan(key)).toBeNull();
  });
});
