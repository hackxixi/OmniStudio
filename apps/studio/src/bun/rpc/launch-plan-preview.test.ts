import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { clearLaunchPlanCache } from "../launch-plan";
import { launchPlanPreviewForRpc } from "./launch-plan-preview";

/**
 * 启动计划预览的路径解析：LOCAL_MODEL_PATH 可能是「主模型 + mmproj」的仓库目录，
 * 真正启动时 llama.ts 用 llamaLoadablePath 挑出主 GGUF；预览必须走同一条解析，
 * 否则预览报「读不到」而实际能起来。
 */

// ---------- 合成 GGUF 字节流（复制自 launch-plan.test.ts，不从 .test.ts import） ----------
const u32 = (v: number) => new Uint8Array([v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >>> 24) & 0xff]);
const u64 = (v: number) => new Uint8Array([...u32(v >>> 0), ...u32(Math.floor(v / 2 ** 32))]);
const str = (s: string) => {
  const b = new TextEncoder().encode(s);
  return new Uint8Array([...u64(b.length), ...b]);
};
type KV = { key: string; type: number; value: Uint8Array };
const kvStr = (key: string, value: string): KV => ({ key, type: 8, value: str(value) });
const kvU32 = (key: string, value: number): KV => ({ key, type: 4, value: u32(value) });
function buildGguf(kv: KV[]): Uint8Array {
  const body = kv.flatMap((e) => [str(e.key), u32(e.type), e.value]);
  return new Uint8Array([
    ...new TextEncoder().encode("GGUF"),
    ...u32(3),
    ...u64(0),
    ...u64(kv.length),
    ...body.flatMap((b) => Array.from(b)),
  ]);
}
const modelGguf = () =>
  buildGguf([
    kvStr("general.architecture", "llama"),
    kvU32("llama.block_count", 32),
    kvU32("llama.attention.head_count", 32),
    kvU32("llama.attention.head_count_kv", 8),
    kvU32("llama.embedding_length", 4096),
    kvU32("llama.context_length", 8192),
  ]);
// mmproj 只有 clip 架构，没有 llama.* —— 若被误选，计划必然算不出来。
const mmprojGguf = () => buildGguf([kvStr("general.architecture", "clip")]);

let dir: string;
beforeAll(async () => {
  dir = await mkdtemp(join(tmpdir(), "launch-plan-preview-"));
  clearLaunchPlanCache();
});
afterAll(async () => {
  await rm(dir, { recursive: true, force: true });
  clearLaunchPlanCache();
});

describe("launchPlanPreviewForRpc", () => {
  test("目录模型（主模型 + mmproj）：解析到主 GGUF，预览与文件路径一致", async () => {
    const repo = join(dir, "Some-4B-GGUF");
    await mkdir(repo);
    const main = join(repo, "Some-4B-Q4_K_M.gguf");
    await writeFile(main, modelGguf());
    // mmproj 故意更大：按大小挑会挑错，必须按「不是投影文件」挑。
    await writeFile(join(repo, "mmproj-F32.gguf"), new Uint8Array([...mmprojGguf(), ...new Uint8Array(4096)]));

    const viaDir = await launchPlanPreviewForRpc(repo);
    expect(viaDir.ok).toBe(true);
    const viaFile = await launchPlanPreviewForRpc(main);
    expect(viaFile.ok).toBe(true);
    if (!viaDir.ok || !viaFile.ok) return;
    expect(viaDir.plan.ctxTokens).toBe(viaFile.plan.ctxTokens);
    expect(viaDir.plan.parallel).toBe(viaFile.plan.parallel);
  });

  test("空路径 → not-found；不存在的路径原样交给读取，带回失败原因", async () => {
    expect(await launchPlanPreviewForRpc("   ")).toEqual({ ok: false, error: "no model path", reason: "not-found" });
    const missing = await launchPlanPreviewForRpc(join(dir, "nope.gguf"));
    expect(missing.ok).toBe(false);
  });
});
