import { create } from "zustand";
import type { ModelSource, SearchFormat } from "../../shared/modelscope";

/** 格式筛选：跟随当前引擎（默认）/ 全部 / 指定格式。 */
export type MarketFormatFilter = "auto" | "all" | SearchFormat;

type MarketState = {
  /** 检索平台：打到 ModelScope 还是 Hugging Face。 */
  source: ModelSource;
  /**
   * 用户是否手动选过平台。选过就一直用用户的（跨重启记在 localStorage）；
   * 没选过就跟随下载源路由的默认平台（国内 ModelScope / 海外 Hugging Face）。
   */
  sourceChosen: boolean;
  format: MarketFormatFilter;
  /** 用户手动切换平台：记住选择，之后不再被路由默认值覆盖。 */
  setSource: (source: ModelSource) => void;
  /** 路由给出的默认平台：只在用户没手动选过时生效，不落盘。 */
  applyDefaultSource: (source: ModelSource) => void;
  setFormat: (format: MarketFormatFilter) => void;
};

export const MARKET_SOURCE_KEY = "omni.market.source";

/** 读写偏好：隐私模式 / 无 localStorage 时会抛错，当次会话仍然生效（与 stores/music-player 同策略）。 */
function readPref(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writePref(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* 无 localStorage：只用当次会话 */
  }
}

function storedSource(): ModelSource | null {
  const raw = readPref(MARKET_SOURCE_KEY);
  return raw === "modelscope" || raw === "huggingface" ? raw : null;
}

/**
 * 在线模型市场的检索偏好。放在 store 里而不是组件 state：
 * 打开某个模型的详情页再返回时，平台/格式/关键词都还在。
 */
export const useMarketStore = create<MarketState>((set, get) => {
  const stored = storedSource();
  return {
    // 路由结论到手前先用 ModelScope 占位（与改造前一致），拿到后 applyDefaultSource 覆盖。
    source: stored ?? "modelscope",
    sourceChosen: stored != null,
    format: "auto",
    setSource: (source) => {
      writePref(MARKET_SOURCE_KEY, source);
      set({ source, sourceChosen: true });
    },
    applyDefaultSource: (source) => {
      if (get().sourceChosen || get().source === source) return;
      set({ source });
    },
    setFormat: (format) => set({ format }),
  };
});
