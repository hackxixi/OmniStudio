import { describe, expect, test } from "bun:test";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { buildVerifyState, escalationPrompt, parseVerifyMode, parseVerifyThreshold } from "./agent-verify";

describe("设置解析", () => {
  test("验收模式：未知值一律 off", () => {
    expect(parseVerifyMode("report")).toBe("report");
    expect(parseVerifyMode("escalate")).toBe("escalate");
    expect(parseVerifyMode("")).toBe("off");
    expect(parseVerifyMode("yes")).toBe("off");
  });
  test("阈值：只接受 (0,1)，否则 0.9", () => {
    expect(parseVerifyThreshold("0.8")).toBe(0.8);
    expect(parseVerifyThreshold("")).toBe(0.9);
    expect(parseVerifyThreshold("2")).toBe(0.9);
    expect(parseVerifyThreshold("abc")).toBe(0.9);
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
