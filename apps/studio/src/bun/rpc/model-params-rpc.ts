/**
 * RPC getModelParams / setModelParams / clearModelParams 的实现体（rpc/index.ts 只转发）。
 * 单独成文件同 launch-plan-preview.ts：rpc/index.ts 在 import 期初始化 electrobun，测试 import 不了。
 */
import {
  clearModelParams,
  getModelParams,
  modelParamsKey,
  setModelParams,
} from "../db/model-params";
import { refreshSamplingMetadata, resolveSampling } from "../model-sampling";
import * as Served from "../model-servers";
import * as ServerManager from "../server-manager";
import type { ModelParams, ResolvedSampling } from "../../shared/model-params";

export type GetModelParamsResult = {
  /** 归一后的模型身份（界面之后的 set / clear 用它或原值都行，服务端会再归一一次）。 */
  model: string;
  /** 该模型保存的参数；没存过 = null。 */
  params: ModelParams | null;
  /** 最终采样值 + 逐项来源（按模型 > 模型自带 > 家族推荐 > 全局）。 */
  sampling: ResolvedSampling;
  /** 按现在的参数会发出去的启动命令（跑着的实例用它自己的端口 / 服务名；算不出 = 缺省）。 */
  launchPreview?: string;
  /** 这个模型正在跑，且按现在的参数重启后 argv 会变（参数改了还没生效）。 */
  needsRestart: boolean;
};

export async function getModelParamsForRpc(model: string): Promise<GetModelParamsResult> {
  const key = modelParamsKey(model);
  const params = key ? getModelParams(key) : null;
  // GGUF 头里的推荐采样只能异步读：先预热，下面同步解析（及命令预览）才看得到「模型自带」那一级。
  if (key) await refreshSamplingMetadata(key).catch(() => {});
  const sampling = resolveSampling(key, { override: params?.sampling, thinking: params?.thinking });
  let launchPreview: string | undefined;
  if (key) {
    try {
      launchPreview = Served.servedLaunchPreview(key) ?? ServerManager.getLaunchCommand(key).command;
    } catch {
      launchPreview = undefined;
    }
  }
  let needsRestart = false;
  try {
    needsRestart = key ? Served.servedModelNeedsRestart(key) : false;
  } catch {
    needsRestart = false;
  }
  return { model: key, params, sampling, launchPreview, needsRestart };
}

export function setModelParamsForRpc(
  model: string,
  params: ModelParams,
): { ok: boolean; params: ModelParams | null; needsRestart: boolean; error?: string } {
  const key = modelParamsKey(model);
  if (!key) return { ok: false, params: null, needsRestart: false, error: "model is required" };
  try {
    const saved = setModelParams(key, params);
    return { ok: true, params: saved, needsRestart: Served.servedModelNeedsRestart(key) };
  } catch (e) {
    return { ok: false, params: null, needsRestart: false, error: e instanceof Error ? e.message : String(e) };
  }
}

export function clearModelParamsForRpc(model: string): { ok: boolean; needsRestart: boolean } {
  const key = modelParamsKey(model);
  if (!key) return { ok: false, needsRestart: false };
  clearModelParams(key);
  return { ok: true, needsRestart: Served.servedModelNeedsRestart(key) };
}
