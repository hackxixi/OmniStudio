/**
 * 自愈逻辑与**内核契约**的对接测试。
 *
 * 前面那些用例只验证了判据与文案；这里用假模型流真跑一遍 `Agent`，
 * 钉住两件我们依赖的内核行为：
 *
 * 1. `shouldStopAfterTurn` 返回 false + `agent.followUp()` → 循环会真的再跑一轮
 *    （空回合自愈的实现基础）；
 * 2. 摘掉末尾那条失败的空壳助手消息之后，`agent.continue()` 能正常发起下一次请求
 *    （失败重发的实现基础）。
 *
 * 内核升级时这两条如果变了，这里会先炸 —— 而不是等用户看到"停止没用 / 白跑一轮"。
 */
import { describe, expect, test } from "bun:test";
import { Agent } from "@earendil-works/pi-agent-core";
import {
  createAssistantMessageEventStream,
  type AssistantMessage,
  type AssistantMessageEvent,
  type AssistantMessageEventStream,
  type Model,
} from "@earendil-works/pi-ai";

import { attachTurnRecovery, dropTrailingFailures, isRetryableTurnFailure, stepLimitWrapUpText } from "./agent-retry";

const MODEL = {
  id: "fake",
  name: "fake",
  api: "openai-completions",
  provider: "omni-studio",
  baseUrl: "http://127.0.0.1:1",
  reasoning: false,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 8192,
  maxTokens: 1024,
} as unknown as Model<string>;

/** 一轮的剧本：答一段正文 / 以某个错误结束 / 被输出上限钳断（length 且只生成 1 token） / 只调一个工具。 */
type Script = { text: string } | { error: string } | { clamped: true } | { tool: boolean };

function message(input: Script, text: string): AssistantMessage {
  const failed = "error" in input;
  const clamped = "clamped" in input;
  return {
    role: "assistant",
    content: failed || clamped ? [] : "tool" in input ? [{ type: "toolCall", id: "c1", name: "bash", arguments: {} }] : [{ type: "text", text }],
    api: "openai-completions",
    provider: "omni-studio",
    model: "fake",
    usage: { input: 1, output: clamped ? 1 : 1, cacheRead: 0, cacheWrite: 0, total: 2 } as never,
    stopReason: failed ? "error" : clamped ? "length" : "stop",
    errorMessage: failed ? input.error : undefined,
    timestamp: Date.now(),
  } as AssistantMessage;
}

/** 假 streamFn：按剧本依次返回，调用次数记在 calls 里。 */
function fakeStream(
  scripts: Script[],
  calls: { count: number },
  sink: AssistantMessage[],
): unknown {
  return () => {
    const script = scripts[Math.min(calls.count, scripts.length - 1)]!;
    calls.count += 1;
    const stream = createAssistantMessageEventStream();
    const final = message(script, "text" in script ? script.text : "");
    sink.push(final);
    const done: AssistantMessageEvent =
      "error" in script
        ? { type: "error", reason: "error", error: final }
        : { type: "done", reason: "clamped" in script ? "length" : "stop", message: final };
    queueMicrotask(() => {
      stream.push({ type: "start", partial: final });
      stream.push(done);
    });
    return stream as never;
  };
}

const buildAgent = (scripts: Script[], calls: { count: number }, sink: AssistantMessage[]) =>
  new Agent({
    streamFn: fakeStream(scripts, calls, sink) as never,
    initialState: { model: MODEL, systemPrompt: "你是测试用助手", tools: [], messages: [] },
  });

describe("空回合自愈（内核契约）", () => {
  test("空回合会注入提醒并继续，直到模型真的给出结论", async () => {
    const calls = { count: 0 };
    const sink: AssistantMessage[] = [];
    const agent = buildAgent([{ text: "" }, { text: "这次答了" }], calls, sink);
    const nudges: number[] = [];
    attachTurnRecovery(agent, {
      steps: () => 0,
      maxSteps: 10,
      budget: 2,
      onNudge: (attempt) => nudges.push(attempt),
    });

    await agent.prompt("做点事");

    expect(calls.count).toBe(2);
    expect(nudges).toEqual([1]);
    // 提醒是作为 harness 消息进上下文的（模型下一轮才看得见），并且标注了不是用户发言。
    const nudge = agent.state.messages.find(
      (item) =>
        item.role === "user" && JSON.stringify(item.content).includes("系统消息，不是用户发言"),
    );
    expect(nudge).toBeDefined();
  });

  test("一路空到底：提醒用完就放手（不会无限要它说话）", async () => {
    const calls = { count: 0 };
    const agent = buildAgent([{ text: "" }], calls, []);
    const nudges: number[] = [];
    attachTurnRecovery(agent, {
      steps: () => 0,
      maxSteps: 10,
      budget: 2,
      onNudge: (attempt) => nudges.push(attempt),
    });

    await agent.prompt("做点事");

    expect(nudges).toEqual([1, 2]);
    expect(calls.count).toBe(3); // 首轮 + 两次提醒
  });

  test("长度钳制型空回合不提醒：重发一次 max_tokens 还是会被钳到 1，直接交出去", async () => {
    const calls = { count: 0 };
    const agent = buildAgent([{ clamped: true }], calls, []);
    const nudges: number[] = [];
    attachTurnRecovery(agent, {
      steps: () => 0,
      maxSteps: 10,
      budget: 2,
      onNudge: (attempt) => nudges.push(attempt),
    });

    await agent.prompt("做点事");

    // 只跑了首轮：钳制是请求侧算术，提醒模型没有任何意义。
    expect(calls.count).toBe(1);
    expect(nudges).toEqual([]);
  });

  test("预算为 0（用户关掉自愈）时一次都不提醒", async () => {
    const calls = { count: 0 };
    const agent = buildAgent([{ text: "" }], calls, []);
    attachTurnRecovery(agent, { steps: () => 0, maxSteps: 10, budget: 0, onNudge: () => {} });

    await agent.prompt("做点事");

    expect(calls.count).toBe(1);
  });

  test("步数到顶 + 空回合（非钳制）：先给收尾回合，不再提醒（收尾优先级高于空回合提醒）", async () => {
    const calls = { count: 0 };
    const agent = buildAgent([{ text: "" }, { text: "收尾后说了" }], calls, []);
    const nudges: number[] = [];
    const wrapUps: number[] = [];
    attachTurnRecovery(agent, {
      steps: () => 5,
      maxSteps: 5,
      budget: 2,
      onNudge: (attempt) => nudges.push(attempt),
      onWrapUp: () => wrapUps.push(1),
    });

    await agent.prompt("做点事");

    // 空回合本身不算到顶后的"该停"：它进的是收尾分支（空回合是"没正文"的子集），
    // 所以 nudges 为 0、wrapUps 为 1。
    expect(calls.count).toBe(2);
    expect(nudges).toEqual([]);
    expect(wrapUps).toEqual([1]);
  });

  test("步数到顶 + 长度钳制型空回合：直接停（收尾重发也是同一截）", async () => {
    const calls = { count: 0 };
    const agent = buildAgent([{ clamped: true }], calls, []);
    const wrapUps: number[] = [];
    attachTurnRecovery(agent, {
      steps: () => 5,
      maxSteps: 5,
      budget: 2,
      onNudge: () => {},
      onWrapUp: () => wrapUps.push(1),
    });

    await agent.prompt("做点事");

    expect(calls.count).toBe(1);
    expect(wrapUps).toEqual([]);
  });
});

describe("步数上限收尾（内核契约）", () => {
  /**
   * 钉住收尾回合用到的内核行为：`shouldStopAfterTurn` 拿到的是下一轮真正要用的
   * currentContext（可变，改它的 `tools` 就是改下一轮请求的工具列表）；返回 false
   * 后内核会 drain steering 队列接着跑。fakeStream 把每一轮收到的 tools 记进
   * seenTools，用例据此断言"那一轮真的没有工具"。
   */
  function buildAgentWithToolProbe(
    scripts: Script[],
    calls: { count: number },
    sink: AssistantMessage[],
    seenTools: (number | undefined)[],
  ) {
    // 外层包一层记下每次 provider 请求收到的 tools 长度（内核用 currentContext.tools
    // 组装 llmContext，见 agent-loop 的 streamAssistantResponse），内层就是原假模型流。
    const baseStream = (
      _model: never,
      context: { tools?: unknown[] },
      _options: never,
    ): AssistantMessageEventStream => {
      seenTools.push(context.tools?.length);
      const script = scripts[Math.min(calls.count, scripts.length - 1)]!;
      calls.count += 1;
      const stream = createAssistantMessageEventStream();
      const final = message(script, "text" in script ? script.text : "");
      sink.push(final);
      const done: AssistantMessageEvent =
        "error" in script
          ? { type: "error", reason: "error", error: final }
          : { type: "done", reason: "clamped" in script ? "length" : "stop", message: final };
      queueMicrotask(() => {
        stream.push({ type: "start", partial: final });
        stream.push(done);
      });
      return stream;
    };
    return new Agent({
      streamFn: baseStream as never,
      initialState: {
        model: MODEL,
        systemPrompt: "你是测试用助手",
        tools: [{ name: "bash", description: "d", parameters: {}, execute: async () => ({ content: [], details: {} }) } as never],
        messages: [],
      },
    });
  }

  test("到上限且最后一轮是工具调用：多跑一轮、那一轮 tools 为空、收到 wrap-up、最后有正文", async () => {
    const calls = { count: 0 };
    const sink: AssistantMessage[] = [];
    const seenTools: (number | undefined)[] = [];
    const agent = buildAgentWithToolProbe([{ tool: true }, { text: "结论：…" }], calls, sink, seenTools);
    const wrapUps: number[] = [];
    attachTurnRecovery(agent, {
      steps: () => 5,
      maxSteps: 5,
      budget: 2,
      onNudge: () => {},
      onWrapUp: () => wrapUps.push(1),
    });

    await agent.prompt("做点事");

    expect(calls.count).toBe(2); // 首轮（到顶） + 收尾回合，不再多
    expect(wrapUps).toEqual([1]);
    // 首轮带工具，收尾回合不带（置空的清单原样进了下一轮请求）。
    expect(seenTools[0]).toBe(1);
    expect(seenTools[1]).toBe(0);
    // 收尾消息进了上下文，且标明是 harness 消息、禁止再调工具。
    const wrapUp = agent.state.messages.find(
      (item) =>
        item.role === "user" &&
        JSON.stringify(item.content).includes(stepLimitWrapUpText(5).split("\n")[0]!),
    );
    expect(wrapUp).toBeDefined();
    const last = agent.state.messages[agent.state.messages.length - 1] as { content?: unknown };
    expect(JSON.stringify(last.content)).toContain("结论：");
  });

  test("到上限但最后一轮已经有正文：直接停，不给收尾回合", async () => {
    const calls = { count: 0 };
    const seenTools: (number | undefined)[] = [];
    const agent = buildAgentWithToolProbe([{ text: "它已经说完了" }], calls, [], seenTools);
    const wrapUps: number[] = [];
    attachTurnRecovery(agent, {
      steps: () => 5,
      maxSteps: 5,
      budget: 2,
      onNudge: () => {},
      onWrapUp: () => wrapUps.push(1),
    });

    await agent.prompt("做点事");

    expect(calls.count).toBe(1);
    expect(wrapUps).toEqual([]);
  });

  test("收尾只发生一次：第二轮到顶（收尾又没正文）就是真停", async () => {
    const calls = { count: 0 };
    const seenTools: (number | undefined)[] = [];
    // 剧本：首轮调工具（触发收尾）→ 收尾又调工具（此时收尾已给过，应直接停）。
    const agent = buildAgentWithToolProbe([{ tool: true }, { tool: true }], calls, [], seenTools);
    const wrapUps: number[] = [];
    attachTurnRecovery(agent, {
      steps: () => 5,
      maxSteps: 5,
      budget: 2,
      onNudge: () => {},
      onWrapUp: () => wrapUps.push(1),
    });

    await agent.prompt("做点事");

    expect(wrapUps).toEqual([1]);
    expect(calls.count).toBe(2);
    // 两轮的 tools 都空了（收尾置空之后不再恢复）。
    expect(seenTools).toEqual([1, 0]);
  });
});

describe("失败重发（内核契约）", () => {
  test("错误回合之后：摘掉空壳 + continue() 能接着跑出结论", async () => {
    const calls = { count: 0 };
    const sink: AssistantMessage[] = [];
    const agent = buildAgent(
      [{ error: "503 Service Unavailable" }, { text: "重试之后答上了" }],
      calls,
      sink,
    );

    await agent.prompt("做点事");
    const failed = agent.state.messages[agent.state.messages.length - 1]!;
    expect(isRetryableTurnFailure(failed as never)).toBe(true);

    // 这就是 runAgentTurn 里做的两步：摘掉失败消息 → continue()。
    agent.state.messages = dropTrailingFailures(agent.state.messages as never[]) as never;
    expect(agent.state.messages.some((item) => item.role === "assistant")).toBe(false);
    await agent.continue();

    expect(calls.count).toBe(2);
    const last = agent.state.messages[agent.state.messages.length - 1] as { content?: unknown };
    expect(JSON.stringify(last.content)).toContain("重试之后答上了");
  });

  test("不摘就直接 continue()：内核会拒绝（说明这一步不是多余的）", async () => {
    const calls = { count: 0 };
    const agent = buildAgent([{ error: "503 Service Unavailable" }], calls, []);
    await agent.prompt("做点事");
    await expect(agent.continue()).rejects.toThrow(/assistant/i);
  });
});
