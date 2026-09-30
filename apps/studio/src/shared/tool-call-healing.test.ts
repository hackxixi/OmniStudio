import { describe, expect, test } from "bun:test";
import {
  healToolCalls,
  parseInlineToolCalls,
  stripInlineToolCalls,
  hasInlineToolCallMarker,
  MAX_HEAL_INPUT_CHARS,
} from "./tool-call-healing";

const FC = "</" + "function>";

describe("tool-call-healing", () => {
    test("1a. hermes format", () => {
    const text =
      'Sure! <tool_call>{"name":"read_file","arguments":{"path":"a.txt"}}</tool_call> done';
    const calls = parseInlineToolCalls(text);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.name).toBe("read_file");
    expect(calls[0]!.arguments).toBe('{"path":"a.txt"}');
    expect(calls[0]!.format).toBe("hermes");
    const { text: healed } = healToolCalls(text, ["read_file"]);
    expect(healed).toBe("Sure!  done");
  });

  test("1b. function-tag well-formed", () => {
    const text =
      'before <function=grep>{"path":"b.txt"}' + FC + ' after';
    const calls = parseInlineToolCalls(text);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.name).toBe("grep");
    expect(calls[0]!.arguments).toBe('{"path":"b.txt"}');
    expect(calls[0]!.format).toBe("function-tag");
    const { text: healed } = healToolCalls(text, ["grep"]);
    expect(healed).toBe("before  after");
  });

  test("1c. function-tag missing '>'", () => {
    const text = 'before <function=grep{"path":"c.txt"}' + FC + ' after';
    const calls = parseInlineToolCalls(text);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.name).toBe("grep");
    expect(calls[0]!.arguments).toBe('{"path":"c.txt"}');
  });

  test("1d. bracket-tool-calls", () => {
    const text =
      'x [TOOL_CALLS][{"name":"read_file","arguments":{"path":"a.txt"}},{"name":"ls","arguments":{}}] y';
    const calls = parseInlineToolCalls(text);
    expect(calls).toHaveLength(2);
    expect(calls[0]!.name).toBe("read_file");
    expect(calls[0]!.format).toBe("bracket-tool-calls");
    expect(calls[1]!.name).toBe("ls");
    const { text: healed } = healToolCalls(text, ["read_file", "ls"]);
    expect(healed).toBe("x  y");
  });

  test("1e. gemma shape A", () => {
    const text =
      'g <|tool_call|>{"name":"calc","arguments":{"x":1}}<|/tool_call|> h';
    const calls = parseInlineToolCalls(text);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.name).toBe("calc");
    expect(calls[0]!.format).toBe("gemma");
  });

  test("1f. gemma shape B (fenced)", () => {
    const text =
      'g ```tool_code\n{"name":"calc","arguments":{"x":2}}\n``` h';
    const calls = parseInlineToolCalls(text);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.name).toBe("calc");
    expect(calls[0]!.format).toBe("gemma");
  });

  test("2. two formats + prose mixed", () => {
    const text =
      "Let me look.\n" +
      '<tool_call>{"name":"read_file","arguments":{"path":"a.txt"}}</tool_call>' +
      "\nthen grep.\n" +
      "<function=grep>{\"q\":\"foo\"}" + FC +
      "\nfinished.";
    const calls = parseInlineToolCalls(text);
    expect(calls).toHaveLength(2);
    expect(calls[0]!.name).toBe("read_file");
    expect(calls[1]!.name).toBe("grep");
    const { text: healed } = healToolCalls(text, ["read_file", "grep"]);
    expect(healed).toBe("Let me look.\n\nthen grep.\n\nfinished.");
  });

  test("3. missing close tag (hermes truncated)", () => {
    const text = 'start <tool_call>{"name":"read_file","arguments":{"path":"a.txt"}}';
    const calls = parseInlineToolCalls(text);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.name).toBe("read_file");
    const { text: healed } = healToolCalls(text, ["read_file"]);
    expect(healed).toBe("start ");
  });

  test("4. allowedNames filter: disallowed call stays in text", () => {
    const text =
      '<tool_call>{"name":"read_file","arguments":{"path":"a.txt"}}</tool_call>' +
      '<tool_call>{"name":"rm_rf","arguments":{"dir":"x"}}</tool_call>';
    const res = healToolCalls(text, ["read_file"]);
    expect(res.calls).toHaveLength(1);
    expect(res.calls[0]!.name).toBe("read_file");
    expect(res.text).toBe(
      '<tool_call>{"name":"rm_rf","arguments":{"dir":"x"}}</tool_call>',
    );
  });

  test("5. broken JSON -> no throw, empty calls, text unchanged", () => {
    const text =
      'before <tool_call>{"name":"read_file","arguments":{BAD}}</tool_call> after';
    const res = healToolCalls(text, ["read_file"]);
    expect(res.calls).toHaveLength(0);
    expect(res.text).toBe(text);
  });

  test("6. plain text with no calls -> strict equality (toBe)", () => {
    const text = "just a normal reply, nothing fancy here.";
    const res = healToolCalls(text, ["read_file"]);
    expect(res.calls).toHaveLength(0);
    expect(res.text).toBe(text);
  });

  test("7. arguments as string form", () => {
    const text =
      '<tool_call>{"name":"run","arguments":"{\\"a\\":1}"}</tool_call>';
    const calls = parseInlineToolCalls(text);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.arguments).toBe('{"a":1}');
  });

  test("8. over MAX_HEAL_INPUT_CHARS -> returned verbatim, no calls", () => {
    const text = "a".repeat(MAX_HEAL_INPUT_CHARS + 1);
    const res = healToolCalls(text, ["read_file"]);
    expect(res.calls).toHaveLength(0);
    expect(res.text).toBe(text);
  });

  test("9. multiple calls: spans non-overlapping, in order", () => {
    const text =
      '<tool_call>{"name":"a1","arguments":{}}</tool_call>' +
      " middle " +
      "<function=b2>{\"x\":1}" + FC;
    const calls = parseInlineToolCalls(text);
    expect(calls).toHaveLength(2);
    for (let i = 1; i < calls.length; i++) {
      expect(calls[i]!.start).toBeGreaterThanOrEqual(calls[i - 1]!.end);
    }
    expect(calls[0]!.start).toBeLessThan(calls[1]!.start);
    expect(calls[0]!.name).toBe("a1");
    expect(calls[1]!.name).toBe("b2");
  });

  test("invariant: heal never alters non-promoted bytes (hermes + prose)", () => {
    const prose = "The model says: ";
    const call = '<tool_call>{"name":"read_file","arguments":{"path":"a.txt"}}</tool_call>';
    const tail = " and that is it.";
    const text = prose + call + tail;
    const res = healToolCalls(text, ["read_file"]);
    expect(res.text).toBe(prose + tail);
  });
});

/* ---------------- stripInlineToolCalls（展示侧清场） ---------------- */

/**
 * 测试里的标签用拼接构造：开/闭标签都是单行形态（逐段拼只是为了源码可读），
 * function 开标签与参数标签同样逐段拼 —— 字面量与真实模型输出逐字节一致。
 */
const TOPEN = "<tool_call" + ">";
const TCLOSE = "<" + "/tool_call>";
const ft = (name: string, params: [string, string][], close = true) =>
  "<" + "function=" + name + ">" +
  params.map(([k, v]) => "<" + "parameter=" + k + ">" + v + "</" + "parameter>").join("") +
  (close ? "</" + "function>" : "");

describe("stripInlineToolCalls（展示侧清场：对话页 / 历史重放）", () => {
  test("1. 真实形态：hermes 外壳包两个参数式调用（mimo 蒸馏模型）→ 全部移除，空外壳不残留", () => {
    const text =
      TOPEN +
      "\n" + ft("Bash", [["command", "curl -L --max-time 20 -sS 'https://api.github.com/search/repositories?q=toolcall-15' | head -c 4000"]]) +
      "\n" + ft("Bash", [["command", "git ls-remote https://github.com/toolcall-15/toolcall-15.git HEAD 2>&1 | head -5"]]) +
      TCLOSE;
    const { text: out, removedNames } = stripInlineToolCalls(text);
    expect(out).toBe("");
    expect(removedNames).toEqual(["Bash"]);
  });

  test("2. 完整的 hermes 纯 JSON 调用同样移除", () => {
    const text = TOPEN + '{"name":"Bash","arguments":{"command":"ls"}}' + TCLOSE;
    const { text: out, removedNames } = stripInlineToolCalls(text);
    expect(out).toBe("");
    expect(removedNames).toEqual(["Bash"]);
  });

  test("3. 正文 + 调用 + 收尾：只移除调用，正文原样保留", () => {
    const text =
      "先搜一下。" + TOPEN +
      "\n" + ft("Read", [["path", "a.txt"]]) + TCLOSE +
      "\n然后看结果。";
    const { text: out, removedNames } = stripInlineToolCalls(text);
    expect(out).toBe("先搜一下。\n然后看结果。");
    expect(removedNames).toEqual(["Read"]);
  });

  test("4. 末尾未闭合（流式在调用写到一半时中断）：从开标签处截断，名字找回", () => {
    const text =
      "我来搜一下。" + TOPEN +
      "\n" + ft("Bash", [["command", "curl https://example.com/api?q=很长很长被中断的查询串"]], false);
    const { text: out, removedNames } = stripInlineToolCalls(text);
    expect(out).toBe("我来搜一下。");
    expect(removedNames).toEqual(["Bash"]);
  });

  test("5. 正文里顺嘴提到未闭合标签不当中断截（开标签后是空格 + 普通词）", () => {
    const text = "这就是 " + TOPEN + " 的样子，对吧？";
    const { text: out, removedNames } = stripInlineToolCalls(text);
    expect(out).toBe(text);
    expect(removedNames).toEqual([]);
  });

  test("6. 坏 JSON 的跨度不认成调用，原样保留（没见过就是没见过）", () => {
    const text = TOPEN + '{"name":"Bash","arguments":{BROKEN}}' + TCLOSE + ' done';
    const { text: out, removedNames } = stripInlineToolCalls(text);
    expect(out).toBe(text);
    expect(removedNames).toEqual([]);
  });

  test("7. 未闭合的参数式正文（参数没写完）从开标签处截断", () => {
    const text = "跑一下这个。" + "<" + "function=Bash><" + "parameter=command>curl https://example.com/x";
    const { text: out, removedNames } = stripInlineToolCalls(text);
    expect(out).toBe("跑一下这个。");
    expect(removedNames).toEqual(["Bash"]);
  });

  test("8. 超长：原样返回", () => {
    const text = "a".repeat(MAX_HEAL_INPUT_CHARS + 1) + TOPEN;
    const { text: out } = stripInlineToolCalls(text);
    expect(out).toBe(text);
  });

  test("9. hasInlineToolCallMarker：廉价探测只看开标签", () => {
    expect(hasInlineToolCallMarker(TOPEN)).toBe(true);
    expect(hasInlineToolCallMarker("<" + "function=Bash>")).toBe(true);
    expect(hasInlineToolCallMarker("[TOOL_CALLS]")).toBe(true);
    expect(hasInlineToolCallMarker(TCLOSE)).toBe(false);
    expect(hasInlineToolCallMarker("普通回答")).toBe(false);
  });

  test("10. healToolCalls 也能提升参数式调用（agent 执行侧受益）", () => {
    const text = TOPEN + "\n" + ft("Bash", [["command", "ls"]]) + TCLOSE;
    const { calls, text: healed } = healToolCalls(text, ["Bash"]);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.name).toBe("Bash");
    expect(calls[0]!.arguments).toBe('{"command":"ls"}');
    expect(healed).not.toContain("<" + "parameter=");
  });
});
