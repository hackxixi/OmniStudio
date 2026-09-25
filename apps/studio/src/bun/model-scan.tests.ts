import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { isMmprojFile, scanPlainDir, specialModelFormat } from "./model-scan";

/**
 * model-scan 的 mmproj 排除测试（真测试体）。
 *
 * 为什么套一层子进程：与 model-store.embedding.test.ts 同一先例 —— bun 的 mock
 * 注册表在同一批次里跨文件共享，别的测试文件会桩掉 fs（download-manager.test.ts
 * 把整个 fs 模块桩掉了，leftover 的 existing 集合会往每个 readdir 结果里塞假
 * 文件名），而本套测试需要真实 readdir/stat 真实临时目录。子进程里零 mock，
 * 结果确定。文件名不带 .test.，批次不会自动发现；入口是 model-scan.test.ts
 * （子进程包装）。
 */

const tmp = mkdtempSync(join(tmpdir(), "model-scan-mmproj-"));

function put(dir: string, name: string) {
  writeFileSync(join(dir, name), "gguf");
}

// 平面目录：模型 + 两个 mmproj 变体（排除后只应剩模型一条）
mkdirSync(join(tmp, "flat"));
put(join(tmp, "flat"), "gme-2b-f16.gguf");
put(join(tmp, "flat"), "mmproj-f16.gguf");
put(join(tmp, "flat"), "mmproj-bf16.gguf");

// 子目录里的平面布局（扫描任意深度都排除）
mkdirSync(join(tmp, "nested"));
put(join(tmp, "nested"), "wemm-9b.gguf");
put(join(tmp, "nested"), "mmproj-model-f16.gguf");

// 仓库布局目录（config.json + safetensors → walkWeights 聚合，不走排除点）
mkdirSync(join(tmp, "repo"));
writeFileSync(join(tmp, "repo", "config.json"), "{}");
writeFileSync(join(tmp, "repo", "model.safetensors"), "x");
put(join(tmp, "repo"), "mmproj-f16.gguf");

// laya-mlx 判定模型目录（mlx_config.json 自声明 format）
mkdirSync(join(tmp, "laya"), { recursive: true });
writeFileSync(
  join(tmp, "laya", "mlx_config.json"),
  JSON.stringify({ format: "laya-mlx", repository: "aac6fef/laya-multilingual-mlx" }),
);
writeFileSync(join(tmp, "laya", "model.safetensors"), "x");

// 同名格式但 repo 不在 catalog（别的东西 / 用户自己转换的 checkpoint）：
// 只看自声明的 format，照样标记
mkdirSync(join(tmp, "notlaya"), { recursive: true });
writeFileSync(
  join(tmp, "notlaya", "mlx_config.json"),
  JSON.stringify({ format: "laya-mlx", repository: "someone/else-mlx" }),
);
writeFileSync(join(tmp, "notlaya", "model.safetensors"), "x");

// mlx_config.json 损坏：不标记
mkdirSync(join(tmp, "badcfg"), { recursive: true });
writeFileSync(join(tmp, "badcfg", "mlx_config.json"), "{ not json");
writeFileSync(join(tmp, "badcfg", "model.safetensors"), "x");

const models = scanPlainDir(tmp, "external");

describe("model-scan / mmproj 排除", () => {
  test("isMmprojFile：mmproj-*.gguf 命中，普通权重不命中", () => {
    expect(isMmprojFile("mmproj-f16.gguf")).toBe(true);
    expect(isMmprojFile("mmproj-bf16.gguf")).toBe(true);
    expect(isMmprojFile("mmproj-model-f16.gguf")).toBe(true);
    expect(isMmprojFile("model.gguf")).toBe(false);
    expect(isMmprojFile("mmproj-f16.safetensors")).toBe(false);
    expect(isMmprojFile("prefix-mmproj-f16.gguf")).toBe(false);
  });

  test("平面目录（含子目录）里的 mmproj-*.gguf 不出现在扫描结果", () => {
    const fileNames = models.filter((m) => !m.isDir).map((m) => `${m.repo}/${m.fileName}`);
    expect(fileNames).toContain("flat/gme-2b-f16.gguf");
    expect(fileNames).toContain("nested/wemm-9b.gguf");
    for (const name of fileNames) {
      expect(name.includes("mmproj")).toBe(false);
    }
  });

  test("仓库目录的 files[] 聚合不受损：mmproj 仍在（已知可接受残留，风险表在案）", () => {
    const repoEntry = models.find((m) => m.isDir && m.repo === "repo");
    expect(repoEntry).toBeDefined();
    expect([...(repoEntry?.files ?? [])].sort()).toEqual(["mmproj-f16.gguf", "model.safetensors"]);
  });
});

/**
 * laya-mlx 判定模型的识别：这些目录（worker 下到 HF 缓存的 checkpoint）会被扫描器
 * 当成普通 safetensors 仓库列进「已安装」，列表 / 后端靠 specialModelFormat 认出
 * 它们 —— 列表据此隐藏启动入口，后端据此拒绝启动推理服务器（两者共用同一判定）。
 */
describe("model-scan / specialModelFormat（laya-mlx）", () => {
  test("目录里的 mlx_config.json 声明 format=laya-mlx → 标记", () => {
    expect(specialModelFormat(join(tmp, "laya"))).toBe("laya-mlx");
  });

  test("同格式但 repo 不在 catalog（别的东西 / 自己转换的）→ 照样标记（只认自声明格式）", () => {
    expect(specialModelFormat(join(tmp, "notlaya"))).toBe("laya-mlx");
  });

  test("mlx_config.json 损坏 → 不标记（宁可当普通目录，不把能加载的模型藏掉）", () => {
    expect(specialModelFormat(join(tmp, "badcfg"))).toBeNull();
  });

  test("普通仓库 / 无 mlx_config.json 的目录 → 不标记", () => {
    expect(specialModelFormat(join(tmp, "repo"))).toBeNull();
    expect(specialModelFormat(join(tmp, "flat"))).toBeNull();
  });

  test("扫描结果：laya 目录的文件条目带 special 标记，非 laya 的目录不带", () => {
    // laya 目录（mlx_config.json + model.safetensors，无 config.json）不是标准仓库布局，
    // 扫描器按文件条目列出；special 标记来自目录里的 mlx_config.json。
    const laya = models.find((m) => m.repo === "laya");
    expect(laya?.special).toBe("laya-mlx");
    for (const m of models) {
      if (m.repo === "repo" || m.repo === "badcfg" || m.repo === "flat" || m.repo === "nested") {
        expect(m.special).toBeUndefined();
      }
    }
    const notLaya = models.find((m) => m.repo === "notlaya");
    expect(notLaya?.special).toBe("laya-mlx");
  });
});
