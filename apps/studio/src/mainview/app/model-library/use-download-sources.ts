import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";

import { rpcClient } from "@lib/rpc";
import { useMarketStore } from "@stores/market";
import type { SourcePlan } from "@/shared/net-sources";

/** 与设置页「下载源」共用同一个缓存键：那边点「重新检测」后这里跟着更新。 */
export const DOWNLOAD_SOURCES_QUERY_KEY = ["download-sources"] as const;

/**
 * 下载源路由结论（主进程 net-sources 探测 / 用户强制）。拿不到（老主进程、RPC 报错）
 * 时为 undefined，调用方按原来的 ModelScope 默认处理。
 */
export function useDownloadSourcePlan(): SourcePlan | undefined {
  return useDownloadSourcePlanQuery().data;
}

/** 同上，但把查询对象整个给出来（需要区分「还在查」和「查不到」的调用方用）。 */
export function useDownloadSourcePlanQuery() {
  return useQuery({
    queryKey: DOWNLOAD_SOURCES_QUERY_KEY,
    // 不在这里吞错误：缓存键与设置页共用，那边按 SourcePlan 读 data，塞个 null 进去会炸。
    queryFn: () => rpcClient.getDownloadSources({}),
    staleTime: 60_000,
    retry: false,
  });
}

/**
 * 模型市场 / 详情页的默认平台跟随下载源路由；用户手动选过平台就不动（见 stores/market）。
 * 返回路由结论，方便调用方顺手展示实际端点。
 */
export function useMarketSourceFromPlan(): SourcePlan | undefined {
  const plan = useDownloadSourcePlan();
  const applyDefaultSource = useMarketStore((s) => s.applyDefaultSource);
  const modelSource = plan?.modelSource;
  useEffect(() => {
    if (modelSource) applyDefaultSource(modelSource);
  }, [modelSource, applyDefaultSource]);
  return plan;
}

/** Hugging Face 这一路实际先打的域名（镜像或官方），给平台切换按钮显示用。 */
export function hfHostOf(plan: SourcePlan | undefined): string | null {
  const first = plan?.hfEndpoints?.[0];
  if (!first) return null;
  try {
    return new URL(first).host;
  } catch {
    return null;
  }
}
