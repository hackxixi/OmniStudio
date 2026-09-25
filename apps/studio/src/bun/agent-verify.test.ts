import { describe, expect, test } from "bun:test";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { buildVerifyState, escalationPrompt, lookupsAllEmpty, parseVerifyMode, parseVerifyThreshold } from "./agent-verify";

describe("设置解析", () => {
  test("验收模式：未知值一律 off", () => {
    expect(parseVerifyMode("report")).toBe("report");
    expect(parseVerifyMode("escalate")).toBe("escalate");
    expect(parseVerifyMode("")).toBe("off");
    expect(parseVerifyMode("yes")).toBe("off");
  });
  test("阈值：只接受 (0,1)，否则默认 0.7", () => {
    expect(parseVerifyThreshold("0.8")).toBe(0.8);
    expect(parseVerifyThreshold("")).toBe(0.7);
    expect(parseVerifyThreshold("2")).toBe(0.7);
    expect(parseVerifyThreshold("abc")).toBe(0.7);
  });
});

describe("buildVerifyState：把一轮整理成验收状态", () => {
  const turn = [
    { role: "assistant", content: [{ type: "toolCall", id: "1", name: "read_file", arguments: { path: "notes/todo.md" } }] },
    { role: "toolResult", toolCallId: "1", toolName: "read_file", content: [{ type: "text", text: "- [ ] 报销\n- [ ] 预约牙医" }] },
    { role: "assistant", content: [{ type: "toolCall", id: "2", name: "generate_speech", arguments: { text: "报销，预约牙医" } }] },
    { role: "toolResult", toolCallId: "2", toolName: "generate_speech", content: [{ type: "text", text: "Audio saved as aud_1" }] },
    { role: "assistant", content: [{ type: "text", text: "已生成语音 aud_1。" }] },
  ] as unknown as AgentMessage[];

  test("动作按顺序带参数与结果，最终回复单独给出", () => {
    const { context, reply } = buildVerifyState("把待办读给我听", turn);
    expect(context).toContain("User request: 把待办读给我听");
    expect(context).toContain('1. read_file({"path":"notes/todo.md"}) → - [ ] 报销 - [ ] 预约牙医');
    expect(context).toContain("2. generate_speech(");
    expect(context).toContain("→ Audio saved as aud_1");
    expect(reply).toBe("已生成语音 aud_1。");
  });
  test("没有动作时写 (none)", () => {
    const only = [{ role: "assistant", content: [{ type: "text", text: "75" }] }] as unknown as AgentMessage[];
    expect(buildVerifyState("1200/16", only).context).toContain("(none)");
  });
  test("没人应答的 ask_user 写成「已问、等回答」，不当失败", () => {
    const asked = [
      { role: "assistant", content: [{ type: "toolCall", id: "1", name: "ask_user", arguments: { questions: [{ question: "发给谁？" }] } }] },
      {
        role: "toolResult",
        toolCallId: "1",
        toolName: "ask_user",
        content: [{ type: "text", text: "Asking the user is not available in this mode." }],
        details: { error: "Asking the user is not available in this mode." },
      },
      { role: "assistant", content: [{ type: "text", text: "请告诉我收件人。" }] },
    ] as unknown as AgentMessage[];
    const { context } = buildVerifyState("给他发封邮件", asked);
    expect(context).toContain("waiting for their answer");
    expect(context).not.toContain("not available");
  });
  test("用户答了的 ask_user 原样保留问答", () => {
    const answered = [
      { role: "assistant", content: [{ type: "toolCall", id: "1", name: "ask_user", arguments: {} }] },
      { role: "toolResult", toolCallId: "1", toolName: "ask_user", content: [{ type: "text", text: "Q: 发给谁？\nA: 张伟" }] },
    ] as unknown as AgentMessage[];
    expect(buildVerifyState("给他发封邮件", answered).context).toContain("A: 张伟");
  });
  test("循环守卫追加的【提示】不进验收状态", () => {
    const hinted = [
      { role: "assistant", content: [{ type: "toolCall", id: "1", name: "recall", arguments: { query: "差旅" } }] },
      {
        role: "toolResult",
        toolCallId: "1",
        toolName: "recall",
        content: [
          { type: "text", text: "Nothing found." },
          { type: "text", text: "\n\n【提示】本轮已搜索 2 次。" },
        ],
      },
    ] as unknown as AgentMessage[];
    expect(buildVerifyState("q", hinted).context).not.toContain("【提示】");
  });
  test("工具结果截断，不会把大段输出塞给 JEV", () => {
    const big = [
      { role: "assistant", content: [{ type: "toolCall", id: "1", name: "web_fetch", arguments: { url: "x" } }] },
      { role: "toolResult", toolCallId: "1", toolName: "web_fetch", content: [{ type: "text", text: "字".repeat(5000) }] },
    ] as unknown as AgentMessage[];
    expect(buildVerifyState("q", big).context.length).toBeLessThan(600);
  });
});

test("升级要求带上判定值，并要求不重复已成功的步骤", () => {
  const text = escalationPrompt(0.42);
  expect(text).toContain("0.42");
  expect(text).toContain("不要重复");
});

describe("lookupsAllEmpty：只查找且全没找到时不升级", () => {
  const call = (id: string, name: string) => ({ role: "assistant", content: [{ type: "toolCall", id, name, arguments: {} }] });
  const result = (id: string, name: string, text: string, extra: Record<string, unknown> = {}) => ({
    role: "toolResult",
    toolCallId: id,
    toolName: name,
    content: [{ type: "text", text }],
    ...extra,
  });
  const turn = (...m: unknown[]) => m as AgentMessage[];

  test("知识库为空 + 网页打不开 → 是", () => {
    expect(
      lookupsAllEmpty(
        turn(
          call("1", "recall"),
          result("1", "recall", "Nothing found."),
          call("2", "web_fetch"),
          result("2", "web_fetch", "fetch failed", { details: { error: "fetch failed" } }),
          { role: "assistant", content: [{ type: "text", text: "没找到相关内容。" }] },
        ),
      ),
    ).toBe(true);
  });
  test("空结果后面追加了守卫提示，仍算没找到", () => {
    const hinted = {
      role: "toolResult",
      toolCallId: "1",
      toolName: "recall",
      content: [
        { type: "text", text: "" },
        { type: "text", text: "\n\n【提示】本轮已搜索 2 次，连续 2 次没有结果。" },
      ],
    };
    expect(lookupsAllEmpty(turn(call("1", "recall"), hinted))).toBe(true);
  });
  test("有一次查到了 → 否", () => {
    expect(
      lookupsAllEmpty(turn(call("1", "recall"), result("1", "recall", "Nothing found."), call("2", "web_search"), result("2", "web_search", "Bun 1.5.0 released"))),
    ).toBe(false);
  });
  test("调了查找以外的工具 → 否（写文件、生成这类漏做 / 做错是升级能补的）", () => {
    expect(lookupsAllEmpty(turn(call("1", "recall"), result("1", "recall", "Nothing found."), call("2", "write_file"), result("2", "write_file", "ok")))).toBe(false);
  });
  test("一个工具都没调 → 否", () => {
    expect(lookupsAllEmpty(turn({ role: "assistant", content: [{ type: "text", text: "75" }] }))).toBe(false);
  });
});
