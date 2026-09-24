import { describe, expect, test } from "bun:test";
import { groupQuestionCriteria, groupsFromProbabilities, REMEMBER_INTENT, routedArgProblem, routedTurnNote } from "./agent-routing";

describe("groupsFromProbabilities：JEV 概率 → 要加载的工具组", () => {
  test("第一名是 none 就只用核心工具", () => {
    expect(groupsFromProbabilities({ none: 0.7, creation: 0.2, dev: 0.1 }, ["creation", "dev"])).toEqual([]);
  });
  test("第一名加载；第二名过阈值也加载，结果按固定组顺序排", () => {
    expect(groupsFromProbabilities({ dev: 0.5, creation: 0.3, none: 0.2 }, ["creation", "dev"])).toEqual(["creation", "dev"]);
  });
  test("第二名不过阈值不加载", () => {
    expect(groupsFromProbabilities({ creation: 0.8, dev: 0.1, none: 0.1 }, ["creation", "dev"])).toEqual(["creation"]);
  });
  test("none 第一但第二名很高时仍加载第二名", () => {
    expect(groupsFromProbabilities({ none: 0.5, creation: 0.4 }, ["creation"])).toEqual(["creation"]);
  });
  test("不在可用列表里的组忽略", () => {
    expect(groupsFromProbabilities({ mcp: 0.9, none: 0.1 }, ["creation"])).toEqual([]);
  });
});

describe("选组问题与说明", () => {
  test("选项只含 none 与可用组", () => {
    expect(Object.keys(groupQuestionCriteria(["creation"]))).toEqual(["none", "creation"]);
  });
  test("全部加载后不再附说明", () => {
    expect(routedTurnNote([])).toBeNull();
    expect(routedTurnNote(["dev"])).toContain("load_tools");
  });
});

describe("routedArgProblem：占位参数兜底", () => {
  test("占位值被拦下，并提示先 ask_user", () => {
    expect(routedArgProblem("generate_speech", { text: "<要朗读的文字>" })).toContain("ask_user");
    expect(routedArgProblem("recall", { query: "unknown" })).not.toBeNull();
    expect(routedArgProblem("mcp__mail__send", { to: "收件人" })).not.toBeNull();
  });
  test("正常值放行；自由文本类工具不查", () => {
    expect(routedArgProblem("web_search", { query: "Bun 最新版本" })).toBeNull();
    expect(routedArgProblem("generate_image", { prompt: "雪山日出", count: 1, negative_prompt: "" })).toBeNull();
    expect(routedArgProblem("web_search", { query: "NA 联赛赛程" })).toBeNull();
    expect(routedArgProblem("write_file", { path: "a.md", content: "TODO" })).toBeNull();
    expect(routedArgProblem("bash", { command: "" })).toBeNull();
  });
});

describe("REMEMBER_INTENT：「记住…」类请求", () => {
  test("中英文说法都认", () => {
    expect(REMEMBER_INTENT.test("记住：我对花生过敏")).toBe(true);
    expect(REMEMBER_INTENT.test("Please remember that my manager's name is Priya.")).toBe(true);
    expect(REMEMBER_INTENT.test("以后推荐菜谱的时候注意")).toBe(true);
  });
  test("普通请求不误判", () => {
    expect(REMEMBER_INTENT.test("我之前跟你说过我喜欢什么音乐来着？")).toBe(false);
    expect(REMEMBER_INTENT.test("画一只猫")).toBe(false);
  });
});
