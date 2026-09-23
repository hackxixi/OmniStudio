import { beforeEach, expect, test } from "bun:test";
import { inArray } from "drizzle-orm";
import { db } from "./index";
import { settings as settingsTable } from "./schema";
import { getAllSettings, getSetting, invalidateSettingsCache, updateSettings } from "./settings";

// 自动调参默认开，但手调过启动参数的老用户保持关（见 settings.ts 的 defaultFor）。
const KEYS = ["SERVER_AUTO_TUNE", "SERVER_CTX_SIZE", "SERVER_BATCH_SIZE", "SERVER_UBATCH_SIZE", "SERVER_GPU_LAYERS"];

beforeEach(() => {
  db.delete(settingsTable).where(inArray(settingsTable.key, KEYS)).run();
  invalidateSettingsCache();
});

test("什么都没存过：默认开", () => {
  expect(getSetting("SERVER_AUTO_TUNE")).toBe("1");
  expect(getAllSettings().SERVER_AUTO_TUNE).toBe("1");
});

test("手存过上下文长度：默认关，getSetting 与 getAllSettings 一致", () => {
  updateSettings({ SERVER_CTX_SIZE: "81920" });
  expect(getSetting("SERVER_AUTO_TUNE")).toBe("0");
  expect(getAllSettings().SERVER_AUTO_TUNE).toBe("0");
});

test("先读过一次默认（开），再存手动参数：缓存立刻作废", () => {
  expect(getSetting("SERVER_AUTO_TUNE")).toBe("1");
  updateSettings({ SERVER_BATCH_SIZE: "512" });
  expect(getSetting("SERVER_AUTO_TUNE")).toBe("0");
});

test("显式选过自动：以他为准，哪怕也存过手动参数", () => {
  updateSettings({ SERVER_CTX_SIZE: "81920", SERVER_AUTO_TUNE: "1" });
  expect(getSetting("SERVER_AUTO_TUNE")).toBe("1");
});
