import { describe, expect, test } from "bun:test";

import { effectiveFromArgv } from "./launch-effective";

/**
 * 从启动命令还原「实际生效值」：这是模型参数抽屉 placeholder 括号里那串数字的来源，
 * 解析错一位用户就会以为「跟着全局给的是 8192」而实际是别的值。
 */
describe("effectiveFromArgv", () => {
  test("llama 完整命令：ctx / parallel / gpu / KV 类型 / FA 都还原", () => {
    const argv = [
      "/opt/homebrew/bin/llama-server",
      "-m",
      "/data/models/my model/model-00001-of-00002.gguf",
      "--host",
      "127.0.0.1",
      "--port",
      "18080",
      "--ctx-size",
      "81920",
      "--parallel",
      "4",
      "--n-gpu-layers",
      "28",
      "--cache-type-k",
      "q8_0",
      "--cache-type-v",
      "q8_0",
      "--flash-attn",
      "on",
      "--temp",
      "0.42",
    ];
    expect(effectiveFromArgv(argv)).toEqual({
      ctxSize: 81920,
      parallel: 4,
      gpuLayers: 28,
      cacheTypeK: "q8_0",
      cacheTypeV: "q8_0",
      flashAttn: "on",
    });
  });

  test("vLLM 的 = 形式：--max-model-len / --max-num-seqs", () => {
    const argv = [
      "python3",
      "-m",
      "vllm.entrypoints.openai.api_server",
      "serve",
      "/data/models/Qwen3-8B",
      "--host",
      "127.0.0.1",
      "--port",
      "18081",
      "--max-model-len=32768",
      "--max-num-seqs=16",
      "--dtype",
      "auto",
    ];
    expect(effectiveFromArgv(argv)).toEqual({ ctxSize: 32768, parallel: 16 });
  });

  test("同一参数出现多次：取最后一次（追加参数拼在最后，就是实际生效的那次）", () => {
    const argv = [
      "llama-server",
      "-m",
      "x.gguf",
      "--ctx-size",
      "4096",
      "--parallel",
      "1",
      "--ctx-size=65536",
      "-np",
      "8",
    ];
    expect(effectiveFromArgv(argv)).toEqual({ ctxSize: 65536, parallel: 8 });
  });

  test("-fa 不带值是布尔开关：视为 on", () => {
    const argv = ["llama-server", "-m", "x.gguf", "-fa", "--temp", "0.8"];
    expect(effectiveFromArgv(argv).flashAttn).toBe("on");
  });

  // shared 不能 import bun 侧（那边会连带初始化数据库）：真实路径里的空格 / 引号
  // 由 bun 侧的引号感知切分还原，这里直接给出切分后的 argv 验证识别不受影响。
  test("带引号的路径（切分后的 argv）不影响参数识别", () => {
    const argv = [
      "C:\\Program Files\\llama.cpp\\llama-server",
      "-m",
      "/data/my model/qwen-8b.gguf",
      "-c",
      "16384",
      "-ngl",
      "999",
    ];
    expect(effectiveFromArgv(argv)).toEqual({ ctxSize: 16384, gpuLayers: 999 });
  });

  test("空命令：什么也认不出", () => {
    expect(effectiveFromArgv([])).toEqual({});
  });

  test("数字解析失败就不填该字段", () => {
    expect(effectiveFromArgv(["llama-server", "-c", "auto"])).toEqual({});
  });
});
