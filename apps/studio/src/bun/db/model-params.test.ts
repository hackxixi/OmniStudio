import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { sqliteClient } from "./index";
import {
  __clearModelParamsCacheForTest,
  clearModelParams,
  getModelParams,
  listModelParams,
  modelParamsKey,
  sanitizeModelParams,
  setModelParams,
} from "./model-params";

/**
 * 按模型参数的存储：真库（test-preload 指向临时数据目录，import ./index 即全量迁移）。
 */

const KEYS = ["repo/a", "repo/b", "repo/bad", "repo/empty"];
afterEach(() => {
  for (const k of KEYS) clearModelParams(k);
  __clearModelParamsCacheForTest();
});

describe("迁移", () => {
  test("0041 建出 model_params 表（主键 model_key）且登记在 journal 里", () => {
    const cols = sqliteClient.query("PRAGMA table_info(model_params)").all() as { name: string; pk: number }[];
    expect(cols.map((c) => c.name).sort()).toEqual(["model_key", "params_json", "updated_at"]);
    expect(cols.find((c) => c.name === "model_key")?.pk).toBe(1);
    const journal = JSON.parse(
      readFileSync(join(import.meta.dir, "migrations", "meta", "_journal.json"), "utf8"),
    ) as { entries: { tag: string; when: number }[] };
    const idx = journal.entries.findIndex((e) => e.tag === "0041_model_params");
    expect(idx).toBeGreaterThan(0);
    // when 必须高于前一条，否则老库升级会跳过它（见 db/index.ts 的时间戳自愈说明）
    expect(journal.entries[idx]!.when).toBeGreaterThan(journal.entries[idx - 1]!.when);
  });
});

describe("sanitizeModelParams", () => {
  test("合法值原样保留", () => {
    const p = {
      ctxSize: 32768,
      parallel: 4,
      gpuLayers: -1,
      cacheTypeK: "bf16",
      cacheTypeV: "iq4_nl",
      flashAttn: "on" as const,
      thinking: "off" as const,
      sampling: { temperature: 0.6, topP: 0.95, topK: 20, minP: 0, presencePenalty: -2, repeatPenalty: 0.5 },
      extraArgs: "--seed 1",
    };
    expect(sanitizeModelParams(p)).toEqual(p);
  });

  test("越界 / 非整数 / 类型不对 / 白名单之外：逐字段丢弃（不钳位）", () => {
    expect(
      sanitizeModelParams({
        ctxSize: 100,
        parallel: 65,
        gpuLayers: 1.5,
        cacheTypeK: "q2_k",
        cacheTypeV: "--evil",
        flashAttn: "yes",
        thinking: true,
        sampling: { temperature: 6, topP: 1.1, topK: 1001, minP: -0.1, presencePenalty: 3, repeatPenalty: 0.4 },
        extraArgs: 42,
        unknown: "x",
      }),
    ).toEqual({});
    expect(sanitizeModelParams({ ctxSize: 1_048_577 })).toEqual({});
    expect(sanitizeModelParams({ sampling: { topK: 2.5 } })).toEqual({});
  });

  test("边界值收；数字串收成数字；空串丢", () => {
    expect(sanitizeModelParams({ ctxSize: 256, parallel: "64", gpuLayers: 999, sampling: { temperature: "0" } })).toEqual({
      ctxSize: 256,
      parallel: 64,
      gpuLayers: 999,
      sampling: { temperature: 0 },
    });
    expect(sanitizeModelParams({ ctxSize: "", sampling: { topP: "" } })).toEqual({});
  });

  test("extraArgs：换行当空白、控制字符丢掉、空白串丢、超长丢", () => {
    expect(sanitizeModelParams({ extraArgs: " --a\n--b\x07 " })).toEqual({ extraArgs: "--a --b" });
    expect(sanitizeModelParams({ extraArgs: "   " })).toEqual({});
    expect(sanitizeModelParams({ extraArgs: "x".repeat(4097) })).toEqual({});
  });

  test("非对象输入 → {}", () => {
    for (const v of [null, undefined, 1, "x", []]) expect(sanitizeModelParams(v)).toEqual({});
  });
});

describe("读写", () => {
  test("set → get 往返（跨缓存清空仍读得到 = 真落库）；整份替换不合并", () => {
    expect(getModelParams("repo/a")).toBeNull();
    setModelParams("repo/a", { ctxSize: 8192, sampling: { temperature: 0.7 } });
    expect(getModelParams("repo/a")).toEqual({ ctxSize: 8192, sampling: { temperature: 0.7 } });
    __clearModelParamsCacheForTest();
    expect(getModelParams("repo/a")).toEqual({ ctxSize: 8192, sampling: { temperature: 0.7 } });
    setModelParams("repo/a", { parallel: 2 });
    expect(getModelParams("repo/a")).toEqual({ parallel: 2 });
  });

  test("写入即校验：非法字段不落库，返回实际落库的那份", () => {
    const saved = setModelParams("repo/bad", { ctxSize: 1, parallel: 3, cacheTypeK: "nope" } as never);
    expect(saved).toEqual({ parallel: 3 });
    __clearModelParamsCacheForTest();
    expect(getModelParams("repo/bad")).toEqual({ parallel: 3 });
  });

  test("空对象（或全被丢光）→ 删除这一行", () => {
    setModelParams("repo/empty", { parallel: 2 });
    expect(setModelParams("repo/empty", { ctxSize: 1 })).toBeNull();
    __clearModelParamsCacheForTest();
    expect(getModelParams("repo/empty")).toBeNull();
    const row = sqliteClient.query("select 1 from model_params where model_key = ?").get("repo/empty");
    expect(row).toBeNull();
  });

  test("库被手改成非法值：读侧再收一遍", () => {
    sqliteClient.run("insert into model_params (model_key, params_json, updated_at) values (?, ?, ?)", [
      "repo/bad",
      JSON.stringify({ ctxSize: 4096, cacheTypeK: "--evil", extraArgs: 1 }),
      Date.now(),
    ]);
    expect(getModelParams("repo/bad")).toEqual({ ctxSize: 4096 });
    sqliteClient.run("update model_params set params_json = 'not json' where model_key = 'repo/bad'");
    __clearModelParamsCacheForTest();
    expect(getModelParams("repo/bad")).toBeNull();
  });

  test("clear 后读不到；list 按修改时间倒序、不列空行", async () => {
    setModelParams("repo/a", { parallel: 2 });
    await Bun.sleep(5);
    setModelParams("repo/b", { parallel: 3 });
    const list = listModelParams().filter((e) => e.model.startsWith("repo/"));
    expect(list.map((e) => e.model)).toEqual(["repo/b", "repo/a"]);
    clearModelParams("repo/a");
    expect(getModelParams("repo/a")).toBeNull();
    expect(listModelParams().some((e) => e.model === "repo/a")).toBe(false);
  });

  test("空 key：get → null，set 抛错", () => {
    expect(getModelParams("  ")).toBeNull();
    expect(() => setModelParams("", { parallel: 2 })).toThrow();
  });
});

describe("modelParamsKey：与已服务模型注册表同一归一规则", () => {
  test("HF repo id / 不存在的路径原样（去首尾空白）", () => {
    expect(modelParamsKey(" Qwen/Qwen3-8B ")).toBe("Qwen/Qwen3-8B");
  });

  test("仓库目录里的权重文件 → 仓库目录；分批 GGUF 的中间片 → 第一片（界面与运行时落同一 key）", () => {
    const dir = mkdtempSync(join(tmpdir(), "model-params-key-"));
    const repo = join(dir, "repo");
    mkdirSync(repo);
    writeFileSync(join(repo, "config.json"), "{}");
    writeFileSync(join(repo, "model.safetensors"), "x");
    expect(modelParamsKey(join(repo, "model.safetensors"))).toBe(repo);

    const split = join(dir, "split");
    mkdirSync(split);
    const first = join(split, "M-Q4_K_M-00001-of-00002.gguf");
    const second = join(split, "M-Q4_K_M-00002-of-00002.gguf");
    writeFileSync(first, "gguf");
    writeFileSync(second, "gguf");
    expect(modelParamsKey(second)).toBe(first);

    setModelParams(second, { ctxSize: 4096 });
    expect(getModelParams(first)).toEqual({ ctxSize: 4096 });
    clearModelParams(first);
  });
});
