import { describe, expect, test } from "bun:test";
import { DEV_INTENT, isEmptyResult, TurnLoopGuard } from "./agent-loop-guard";

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

describe("dev 组只给开发类请求", () => {
  test("非开发请求加载 dev 组被拦，其他组照常", () => {
    const g = new TurnLoopGuard({ devAllowed: false });
    expect(g.check("load_tools", { group: "dev" })).toContain("开发任务");
    expect(g.check("load_tools", { group: "creation" })).toBeNull();
  });
  test("开发请求放行；默认放行（经典调用方不传参数时行为不变）", () => {
    expect(new TurnLoopGuard({ devAllowed: true }).check("load_tools", { group: "dev" })).toBeNull();
    expect(new TurnLoopGuard().check("load_tools", { group: "dev" })).toBeNull();
  });
  test("开发类说法识别", () => {
    expect(DEV_INTENT.test("帮我跑一下单测看看报错")).toBe(true);
    expect(DEV_INTENT.test("git 提交一下")).toBe(true);
    expect(DEV_INTENT.test("把这句话翻译成英文")).toBe(false);
    expect(DEV_INTENT.test("给王总发封邮件")).toBe(false);
  });
});

describe("ask_user 没人应答", () => {
  test("之后本轮不再追问，也不许转向 bash / load_tools", () => {
    const g = new TurnLoopGuard();
    expect(g.record("ask_user", "Asking the user is not available in this mode.", false)).toContain("没有人能回答");
    expect(g.check("ask_user")).toContain("不要再追问");
    expect(g.check("bash")).not.toBeNull();
    expect(g.check("load_tools", { group: "dev" })).not.toBeNull();
    expect(g.check("write_file")).toBeNull();
  });
  test("之后也不许自己编内容去生成（问的就是画什么）", () => {
    const g = new TurnLoopGuard();
    g.record("ask_user", "Asking the user is not available in this mode.", true);
    expect(g.check("generate_image")).toContain("没有人能回答");
    expect(g.check("generate_speech")).not.toBeNull();
  });
  test("用户关掉提问或超时也算没人应答", () => {
    const g = new TurnLoopGuard();
    const dismissed = "The user did not answer (dismissed or timed out). Continue sensibly and say what you assumed.";
    expect(g.record("ask_user", dismissed, false)).toContain("没有人能回答");
    expect(g.check("ask_user")).toContain("不要再追问");
  });
  test("用户正常回答时不受影响", () => {
    const g = new TurnLoopGuard();
    expect(g.record("ask_user", "用户回答：发给 wang@example.com", false)).toBeNull();
    expect(g.check("ask_user")).toBeNull();
  });
});
