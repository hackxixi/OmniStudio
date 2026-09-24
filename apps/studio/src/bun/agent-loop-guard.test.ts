import { describe, expect, test } from "bun:test";
import { isEmptyResult, TurnLoopGuard } from "./agent-loop-guard";

describe("isEmptyResult：识别「没找到」", () => {
  test("各检索工具的空结果文案", () => {
    expect(isEmptyResult("Nothing found in the knowledge base, notes or memory.")).toBe(true);
    expect(isEmptyResult("[file contents]\n(no matches)。没搜到不代表不存在")).toBe(true);
    expect(isEmptyResult("Error: page not found.")).toBe(true);
    expect(isEmptyResult("")).toBe(true);
  });
  test("有内容的结果不算空（长结果里偶然出现 not found 也不算）", () => {
    expect(isEmptyResult("[notes]\n笔记《向量数据库选型》：结论：选 Qdrant。")).toBe(false);
    expect(isEmptyResult(`${"正文".repeat(400)} not found`)).toBe(false);
  });
});

describe("搜索打转", () => {
  test("连续两次没结果就在结果后追加提示，连续三次后拦下搜索", () => {
    const g = new TurnLoopGuard();
    expect(g.record("recall", "Nothing found", false)).toBeNull();
    expect(g.record("web_search", "No relevant results.", false)).toContain("不要再换关键词");
    expect(g.check("find")).toBeNull();
    g.record("find", "No matches.", false);
    expect(g.check("note_read")).toContain("不再继续搜索");
    // 非搜索工具不受影响
    expect(g.check("write_file")).toBeNull();
  });
  test("有结果会清零连续空结果计数", () => {
    const g = new TurnLoopGuard();
    g.record("recall", "Nothing found", false);
    g.record("recall", "[notes]\n有内容", false);
    expect(g.record("recall", "Nothing found", false)).toBeNull();
    expect(g.check("recall")).toBeNull();
  });
  test("总次数上限：搜到东西也不能无限搜", () => {
    const g = new TurnLoopGuard();
    for (let i = 0; i < 4; i++) expect(g.record("web_search", `结果 ${i}`, false)).toBeNull();
    expect(g.record("web_fetch", "页面内容", false)).toContain("本轮已搜索 5 次");
    for (let i = 0; i < 3; i++) g.record("web_fetch", "页面内容", false);
    expect(g.check("web_search")).toContain("不再继续搜索");
  });
});

describe("生成失败后不重试、不绕路", () => {
  test("失败后同一生成工具与绕路工具被拦下，其他工具照常", () => {
    const g = new TurnLoopGuard();
    expect(g.record("generate_speech", "请先在设置里配置语音合成后端", true)).toContain("不要重试");
    expect(g.check("generate_speech")).toContain("已经失败过");
    expect(g.check("bash")).toContain("不要改用 bash 绕路");
    expect(g.check("load_tools")).not.toBeNull();
    expect(g.check("generate_image")).toBeNull();
    expect(g.check("read_file")).toBeNull();
  });
  test("生成成功不留记录", () => {
    const g = new TurnLoopGuard();
    expect(g.record("generate_image", "Image saved as img_1", false)).toBeNull();
    expect(g.check("generate_image")).toBeNull();
  });
});
