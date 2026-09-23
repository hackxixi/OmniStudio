import { expect, test } from "bun:test";

import { classifyStartupError } from "../../shared/engine-errors";
import { extractDeadWorkerError, extractStartupError, isOutOfMemoryLog } from "./errors";

/** mlx_lm.server 遇到不认识的架构时真实打出来的那段（背景见 errors.ts）。 */
const UNSUPPORTED_MODEL_LOG = [
  "Exception in thread Thread-1 (_generate):",
  "Traceback (most recent call last):",
  "  File \"/site-packages/mlx_lm/server.py\", line 695, in _generate",
  "    self.model_provider.load_default()",
  "  File \"/site-packages/mlx_lm/utils.py\", line 255, in _get_classes",
  "    raise ValueError(msg)",
  "ValueError: Model type deepseek_v41 not supported.",
  "/site-packages/mlx_lm/server.py:1723: UserWarning: mlx_lm.server is not recommended for production",
  "Starting httpd at 127.0.0.1 on port 18010...",
].join("\n");

test("认出「服务起来了但模型加载线程已死」，并把 model_type 说出来", () => {
  const message = extractDeadWorkerError(UNSUPPORTED_MODEL_LOG);
  expect(message).toContain("deepseek_v41");
  expect(message).toContain("不会出图");
});

test("线程死在别的原因上：仍然按失败处理，带上异常行", () => {
  const logs = [
    "Exception in thread Thread-1 (_generate):",
    "OSError: [Errno 28] No space left on device",
  ].join("\n");
  const message = extractDeadWorkerError(logs);
  expect(message).toContain("No space left on device");
});

test("正常启动的日志不误报", () => {
  const logs = [
    "Fetching 12 files: 100%|##########| 12/12",
    "Starting httpd at 127.0.0.1 on port 18010...",
  ].join("\n");
  expect(extractDeadWorkerError(logs)).toBeNull();
});

/** llama-server 加载不了模型时真实打出来的尾巴（实测 2026-09-18：架构不认识的 GGUF）。 */
const LLAMA_LOAD_FAILURE_LOG = [
  "0.00.051.535 I srv    load_model: loading model '/tmp/Bonsai-27B-dspark-bf16.gguf'",
  "0.00.052.115 E llama_model_load: error loading model: unknown model architecture: 'dspark'",
  "0.00.052.290 E llama_model_load_from_file_impl: failed to load model",
  "0.00.052.310 E common_fit_params: encountered an error while trying to fit params to free device memory: failed to load model",
  "0.00.052.675 E llama_model_load: error loading model: unknown model architecture: 'dspark'",
  "0.00.052.678 E llama_model_load_from_file_impl: failed to load model",
  "0.00.052.679 E cmn  common_init_: failed to load model '/tmp/Bonsai-27B-dspark-bf16.gguf'",
  "0.00.052.681 I srv  operator(): operator(): cleaning up before exit...",
  "0.00.053.106 E srv  llama_server: exiting due to model loading error",
].join("\n");

test("级联错误：挑出根因行，而不是最后那句总结", () => {
  const message = extractStartupError(LLAMA_LOAD_FAILURE_LOG, "fallback");
  expect(message).toContain("unknown model architecture");
  expect(message).toContain("dspark");
  // 根因行要能被分到 model-format，界面才给得出「架构不支持」的建议
  expect(classifyStartupError(message)).toBe("model-format");
});

test("extractStartupError 的既有行为不变（它只在进程退出 / 超时那条路上被调用）", () => {
  // 这段日志里最后一条错误行就是那条 ValueError —— 问题从来不在于它挑不出来，
  // 而在于服务没退出时根本没人去调它（新的 extractDeadWorkerError 补的就是这一步）。
  expect(extractStartupError(UNSUPPORTED_MODEL_LOG, "fallback")).toBe(
    "ValueError: Model type deepseek_v41 not supported.",
  );
  expect(extractStartupError("Error: unknown model architecture: 'spark2_5'", "fallback")).toContain(
    "unknown model architecture",
  );
});

test("llama.cpp 读不到 GGUF 魔数时，报根因那一行而不是收尾的「模型加载错误」", () => {
  // 实测日志（把 GGUF 仓库目录当成 -m 传进去）：根因在前，收尾句在最后。
  const log = [
    "0.00.119.883 I srv    load_model: loading model '/models/Qwopus3.5-4B-Coder-MTP-GGUF'",
    "0.00.120.730 E gguf_init_from_reader: failed to read magic",
    "0.00.121.070 E llama_model_load: error loading model: llama_model_loader: failed to load model from /models/Qwopus3.5-4B-Coder-MTP-GGUF",
    "0.00.121.142 E llama_model_load_from_file_impl: failed to load model",
    "0.00.145.643 E srv  llama_server: exiting due to model loading error",
  ].join("\n");
  const message = extractStartupError(log, "fallback");
  expect(message).toContain("failed to read magic");
  expect(message).not.toContain("exiting due to model loading error");
});

test("isOutOfMemoryLog：CUDA / Metal / Vulkan 的分配失败措辞都认（降级重试的触发条件）", () => {
  expect(
    isOutOfMemoryLog(
      "llama_model_load: loading\n" +
        "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9216.00 MiB on device 0: cudaMalloc failed: out of memory\n" +
        "llama_init_from_model: failed to initialize the context\n",
    ),
  ).toBe(true);
  expect(isOutOfMemoryLog("ggml_metal_buffer_init: error: failed to allocate buffer, size = 8192.00 MiB\n")).toBe(true);
  expect(
    isOutOfMemoryLog("ggml_metal: error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)\n"),
  ).toBe(true);
  expect(isOutOfMemoryLog("ggml_vulkan: Device memory allocation failed: ErrorOutOfDeviceMemory\n")).toBe(true);
});

test("isOutOfMemoryLog：锁页警告 / 我们自己的注解 / 架构不认识都不算（重试也没用）", () => {
  expect(isOutOfMemoryLog("warning: failed to mlock 1234-byte buffer: Cannot allocate memory\n")).toBe(false);
  expect(isOutOfMemoryLog("[omni] 显存不足，第 1 次降级重试：上下文 32768 → 16384\n")).toBe(false);
  expect(isOutOfMemoryLog("llama_model_load: error loading model: unknown model architecture: 'foo'\n")).toBe(false);
  expect(isOutOfMemoryLog("CUDA error: invalid device function\n")).toBe(false);
});

test("extractStartupError：超窗那句进根因表，排在派生的收尾句前面", () => {
  const log = [
    "srv  send_error: task id = 3, error: the request exceeds the available context size, try increasing it",
    "srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 400",
    "failed to process request",
  ].join("\n");
  expect(extractStartupError(log, "fallback")).toContain("exceeds the available context size");
});
