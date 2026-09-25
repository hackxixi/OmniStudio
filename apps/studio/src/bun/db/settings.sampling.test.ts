import { expect, test } from "bun:test";
import { getSetting, updateSettings } from "./settings";

// 全局采样只是最后一级兜底：默认给通用对话的中性值（以前是 OCR 口味的 0.2 / 1.12）。
// 预加载脚本已把数据库指到临时目录，这里读到的就是 DEFAULTS。
test("全局采样默认值：中性档 + llama.cpp 的 min_p 默认", () => {
  expect(getSetting("SERVER_TEMP")).toBe("0.7");
  expect(getSetting("SERVER_TOP_P")).toBe("0.9");
  expect(getSetting("SERVER_TOP_K")).toBe("40");
  expect(getSetting("SERVER_REPEAT_PENALTY")).toBe("1.0");
  expect(getSetting("SERVER_MIN_P")).toBe("0.05");
  expect(getSetting("SERVER_PRESENCE_PENALTY")).toBe("0");
});

test("默认值只是读侧回落：显式存过的值原样保留", () => {
  updateSettings({ SERVER_TEMP: "0.2", SERVER_MIN_P: "0.1" });
  expect(getSetting("SERVER_TEMP")).toBe("0.2");
  expect(getSetting("SERVER_MIN_P")).toBe("0.1");
});
