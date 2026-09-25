/**
 * RPC `getLaunchPlanPreview` 的实现体（rpc/index.ts 只转发）。
 *
 * 单独成文件是为了能测：rpc/index.ts 在 import 期就初始化 electrobun 运行时，测试里
 * import 不了；这里只依赖 db/settings + launch-plan + llama 的纯函数，测试可直接调用。
 */
import { getSetting, type SettingsKey } from "../db/settings";
import { readGgufMeta, type GgufReadFailure } from "../gguf-meta";
import { getModelParams } from "../db/model-params";
import { refreshLaunchPlan, type LaunchPlan } from "../launch-plan";
import { launchPlanKeyForModel, llamaLoadablePath, persistedEffectiveFlashAttn } from "../runtimes/llama";

export type LaunchPlanPreviewResult =
  | { ok: true; plan: LaunchPlan }
  | { ok: false; error: string; reason: string };

export async function launchPlanPreviewForRpc(path: string): Promise<LaunchPlanPreviewResult> {
  const rawPath = path.trim();
  if (rawPath === "") {
    return { ok: false, error: "no model path", reason: "not-found" };
  }
  // 路径解析与真正启动同源：LOCAL_MODEL_PATH 对「带 mmproj 的 GGUF 仓库」存的是目录，
  // llama.ts 启动（无论走设置里的 LOCAL_MODEL_PATH 还是 served-model 的 overrides.model）
  // 都先过 llamaLoadablePath 挑出主权重文件；预览若直接拿目录去读 GGUF，只会报「读不到」，
  // 而实际能启动 —— 预览与启动对不上。文件路径 / 不存在的路径原样返回，行为不变。
  const modelPath = llamaLoadablePath(rawPath);
  // 与 llama.ts 启动时完全同源的 key：同一函数（launchPlanKeyForModel）、同一设置读法、
  // 同一份按模型参数（key 是原始路径，与启动时的 target 同一归一规则）、同一个
  // 「设置 + 上次实测」的 FA 折算（预览端没有 Runtime 实例，实测值读设置里回写的
  // SERVER_FLASH_ATTN_EFFECTIVE —— 启动过之后两者必然相等）。
  const key = launchPlanKeyForModel(
    modelPath,
    getModelParams(rawPath),
    (k) => getSetting(k as SettingsKey),
    persistedEffectiveFlashAttn(getSetting("SERVER_FLASH_ATTN_EFFECTIVE")),
  );
  const plan = await refreshLaunchPlan(key);
  if (plan !== null) return { ok: true, plan };

  // refreshLaunchPlan 对「读不到 GGUF」静默返回 null，这里补一次读取只为拿到
  // 失败原因码（该读取自身有 mtime 缓存，成本可忽略）。
  const read = await readGgufMeta(modelPath);
  if (read.ok) {
    // GGUF 读得到但计划算不出来：元数据不足以估算 KV cache。
    return { ok: false, error: read.data.filePath, reason: "no-metadata" };
  }
  const reason: GgufReadFailure = read.reason;
  return { ok: false, error: read.error, reason };
}
