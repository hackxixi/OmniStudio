/**
 * tool-call-healing — parses tool calls that small local models emit as
 * plain text instead of structured tool_calls.
 *
 * Pure functions only: no IO, no node:* / bun / electrobun imports.
 *
 * Invariants:
 *   1. Only functions whose name is in allowedNames are promoted.
 *   2. Only the promoted spans are removed from the text.
 *   3. Any parse failure means "never saw it": skip, never throw.
 */

export type InlineToolCall = {
  name: string;
  arguments: string;
  start: number;
  end: number;
  format: "hermes" | "function-tag" | "function-xml" | "bracket-tool-calls" | "gemma";
};

export type HealResult = {
  text: string;
  calls: InlineToolCall[];
};

export const MAX_HEAL_INPUT_CHARS = 256 * 1024;

/* ---------------- JSON helpers (unknown narrowing, no any) -------- */

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function parseJsonObject(s: string): Record<string, unknown> | null {
  let v: unknown;
  try {
    v = JSON.parse(s);
  } catch {
    return null;
  }
  return isRecord(v) ? v : null;
}

function parseJsonArray(s: string): unknown[] | null {
  let v: unknown;
  try {
    v = JSON.parse(s);
  } catch {
    return null;
  }
  return Array.isArray(v) ? v : null;
}

type RawCall = { name: string; arguments: string };

function rawCallFromObject(obj: Record<string, unknown>): RawCall | null {
  if (typeof obj.name !== "string" || obj.name.length === 0) return null;
  const args = obj.arguments;
  let argumentsStr: string;
  if (typeof args === "string") {
    argumentsStr = args;
  } else if (args === undefined) {
    argumentsStr = "";
  } else if (isRecord(args) || Array.isArray(args)) {
    argumentsStr = JSON.stringify(args);
  } else {
    return null;
  }
  return { name: obj.name, arguments: argumentsStr };
}

/* ---------------- Format 1: hermes ------------------------------- */

const HERMES_OPEN = "<tool_call>";
const HERMES_CLOSE = "</tool_call>";

function parseHermes(text: string): InlineToolCall[] {
  const out: InlineToolCall[] = [];
  let from = 0;
  while (from < text.length) {
    const idx = text.indexOf(HERMES_OPEN, from);
    if (idx === -1) break;
    const bodyStart = idx + HERMES_OPEN.length;
    const closeIdx = text.indexOf(HERMES_CLOSE, bodyStart);
    const nextOpen = text.indexOf(HERMES_OPEN, bodyStart);
    // end = where this call ends in the source (spans include the close tag);
    // bodyEnd = where the JSON body ends (excludes the close tag).
    let end: number;
    let bodyEnd: number;
    if (closeIdx !== -1 && (nextOpen === -1 || closeIdx < nextOpen)) {
      bodyEnd = closeIdx;
      end = closeIdx + HERMES_CLOSE.length;
    } else {
      bodyEnd = nextOpen === -1 ? text.length : nextOpen;
      end = bodyEnd;
    }
    const raw = text.slice(bodyStart, bodyEnd).trim();
    const obj = raw.startsWith("{") ? parseJsonObject(raw) : null;
    from = end; // always advance, even when parsing fails
    if (obj) {
      const call = rawCallFromObject(obj);
      if (call) out.push({ ...call, start: idx, end, format: "hermes" });
    }
  }
  return out;
}

/* ---------------- Format 2: function-tag ------------------------- */

const FUNC_OPEN_RE = /<function=([A-Za-z0-9_.-]+)(?:>|\s|(?=\{))/g;
const FUNC_CLOSE = "</" + "function>";
const FUNC_OPEN_MARKER = "<function=";

function parseFunctionTag(text: string): InlineToolCall[] {
  const out: InlineToolCall[] = [];
  for (let m = FUNC_OPEN_RE.exec(text); m !== null; m = FUNC_OPEN_RE.exec(text)) {
    const start = m.index;
    const tagName = m[1]!;
    const bodyStart = m.index + m[0].length;
    const closeIdx = text.indexOf(FUNC_CLOSE, bodyStart);
    let nextOpenIdx = -1;
    let probe = bodyStart;
    for (;;) {
      const t = text.indexOf(FUNC_OPEN_MARKER, probe);
      if (t === -1) break;
      const c = text[t + FUNC_OPEN_MARKER.length];
      if (c !== undefined && /^[A-Za-z0-9_.-]/.test(c)) { nextOpenIdx = t; break; }
      probe = t + 1;
    }
    // end = where this call ends in the source (spans include the close tag);
    // bodyEnd = where the JSON body ends (excludes the close tag).
    let end: number;
    let bodyEnd: number;
    if (closeIdx !== -1 && (nextOpenIdx === -1 || closeIdx < nextOpenIdx)) {
      bodyEnd = closeIdx;
      end = closeIdx + FUNC_CLOSE.length;
    } else {
      bodyEnd = nextOpenIdx === -1 ? text.length : nextOpenIdx;
      end = bodyEnd;
    }
    const raw = text.slice(bodyStart, bodyEnd).trim();
    const obj = raw.startsWith("{") ? parseJsonObject(raw) : null;
    // In function-tag the braces ARE the arguments: the name is already in
    // the tag, so the whole object is the call's arguments JSON. If a model
    // does wrap it with an "arguments" field, use that instead.
    let argumentsStr: string | undefined;
    if (obj) {
      if ("arguments" in obj) {
        const args = obj.arguments;
        if (typeof args === "string") {
          argumentsStr = args;
        } else if (isRecord(args) || Array.isArray(args)) {
          argumentsStr = JSON.stringify(args);
        }
      } else {
        argumentsStr = JSON.stringify(obj);
      }
    }
    if (argumentsStr != undefined) {
      out.push({ name: tagName, arguments: argumentsStr, start, end, format: "function-tag" });
    }
    // advance past this tag: on a zero-length match the regex cursor does
    // not move past m.index, which would spin forever on a malformed tag
    FUNC_OPEN_RE.lastIndex = Math.max(FUNC_OPEN_RE.lastIndex, start + FUNC_OPEN_MARKER.length + 1);
  }
  return out;
}

/* ---------------- Format 3: bracket-tool-calls -------------------- */

const BRACKET_MARKER = "[TOOL_CALLS]";

function parseBracketToolCalls(text: string): InlineToolCall[] {
  const out: InlineToolCall[] = [];
  let from = 0;
  while (from < text.length) {
    const idx = text.indexOf(BRACKET_MARKER, from);
    if (idx === -1) break;
    const bodyStart = idx + BRACKET_MARKER.length;
    let depth = 0;
    let i = bodyStart;
    let found = false;
    for (; i < text.length; i++) {
      const c = text[i];
      if (c === "[") depth++;
      else if (c === "]") {
        depth--;
        if (depth === 0) { found = true; break; }
      } else if (depth === 0) {
        if (c !== " " && c !== "\t" && c !== "\n" && c !== "\r") break;
      }
    }
    if (!found) {
      // unterminated: the body extends to the end of the text
      from = idx + BRACKET_MARKER.length;
      break;
    }
    const end = i + 1;
    const raw = text.slice(bodyStart, end).trim();
    const arr = parseJsonArray(raw);
    from = end; // always advance, even if the array fails to parse
    for (const entry of arr ?? []) {
      if (!isRecord(entry)) continue;
      const call = rawCallFromObject(entry);
      if (!call) continue;
      out.push({ ...call, start: idx, end, format: "bracket-tool-calls" });
    }
  }
  return out;
}

/* ---------------- Format 4: gemma -------------------------------- */

const GEMMA_OPEN = "<|tool_call|>";
const GEMMA_CLOSE = "<|/tool_call|>";
const GEMMA_FENCE_OPEN = "```tool_code";
const GEMMA_FENCE_CLOSE = "```";

function parseGemma(text: string): InlineToolCall[] {
  const out: InlineToolCall[] = [];

  // Shape A: <|tool_call|> ... (close may be missing)
  let from = 0;
  while (from < text.length) {
    const idx = text.indexOf(GEMMA_OPEN, from);
    if (idx === -1) break;
    const bodyStart = idx + GEMMA_OPEN.length;
    const closeIdx = text.indexOf(GEMMA_CLOSE, bodyStart);
    // end = 这段调用在原文里的结束位置（含闭合标签）；
    // bodyEnd = JSON 正文的结束位置（不含闭合标签）—— 与 parseHermes 同一套区分。
    let end: number;
    let bodyEnd: number;
    if (closeIdx !== -1) {
      bodyEnd = closeIdx;
      end = closeIdx + GEMMA_CLOSE.length;
    } else {
      const fence = text.indexOf(GEMMA_FENCE_OPEN, bodyStart);
      bodyEnd = fence !== -1 ? fence : text.length;
      end = bodyEnd;
    }
    const raw = text.slice(bodyStart, bodyEnd).trim();
    const obj = raw.startsWith("{") ? parseJsonObject(raw) : null;
    if (obj) {
      const call = rawCallFromObject(obj);
      if (call) out.push({ ...call, start: idx, end, format: "gemma" });
    }
    // an open tag can only be followed by another open tag when there is
    // no close/fence in between (handled above), so end >= bodyStart here
    from = Math.max(end, bodyStart + 1);
  }

  // Shape B: fenced tool_code blocks
  let f = 0;
  while (f < text.length) {
    const o = text.indexOf(GEMMA_FENCE_OPEN, f);
    if (o === -1) break;
    const bodyStart = text.indexOf("\n", o);
    const realStart = bodyStart === -1 ? o + GEMMA_FENCE_OPEN.length : bodyStart + 1;
    const closeFence = text.indexOf(GEMMA_FENCE_CLOSE, realStart);
    const end = closeFence === -1 ? text.length : closeFence + GEMMA_FENCE_CLOSE.length;
    const raw = text.slice(realStart, closeFence === -1 ? text.length : closeFence).trim();
    const obj = raw.startsWith("{") ? parseJsonObject(raw) : null;
    if (obj) {
      const call = rawCallFromObject(obj);
      if (call) out.push({ ...call, start: o, end, format: "gemma" });
    }
    // never let a zero-width / zero-length fence stop the loop
    f = Math.max(end, o + 1);
  }

  return out;
}

/* ---------------- Top-level API ---------------------------------- */

export function parseInlineToolCalls(text: string): InlineToolCall[] {
  if (text.length > MAX_HEAL_INPUT_CHARS) return [];
  const calls: InlineToolCall[] = [
    ...parseHermes(text),
    ...parseFunctionTag(text),
    ...parseFunctionXml(text),
    ...parseBracketToolCalls(text),
    ...parseGemma(text),
  ];
  calls.sort((a, b) => a.start - b.start || a.end - b.end);
  return calls;
}

export function healToolCalls(
  text: string,
  allowedNames: readonly string[],
): HealResult {
  if (text.length > MAX_HEAL_INPUT_CHARS) {
    return { text, calls: [] };
  }
  const allowed = new Set(allowedNames);
  const all = parseInlineToolCalls(text);
  const promoted = all.filter((c) => allowed.has(c.name));

  if (promoted.length === 0) {
    return { text, calls: [] };
  }

  let out = "";
  let cursor = 0;
  for (const c of promoted) {
    if (c.start < cursor) continue;
    out += text.slice(cursor, c.start);
    cursor = c.end;
  }
  out += text.slice(cursor);

  return { text: out, calls: promoted };
}

/* ---------------- Format 5: function-xml（parameter 参数） -------------- */

/**
 * pi 风格的文本协议：`<function=名字` + 终止符（`>` 等，同格式 2）之后是一组
 * `<parameter=键>值</parameter` 参数块，最后以 `</function>` 收尾。
 *
 * 会输出这种形态的模型一般是 agent 轨迹蒸馏出来的（如 mimo-v2.6-distill）：
 * 没有结构化 tool_calls 能力，只能把调用写成文本。与格式 2 的区别是正文不是
 * JSON 而是 XML 参数组；两者共用同一个开标签形态，按正文内容分流。
 */
const XML_PARAM_MARKER = "<" + "parameter=";
const XML_PARAM_RE = new RegExp("<" + "parameter=([A-Za-z0-9_.-]+)>([\\s\\S]*?)</" + "parameter>", "g");

/** 标准 XML 实体；顺序有讲究：&amp; 最后展开，避免把 &amp;&lt; 变成 &lt;。 */
function unescapeXml(s: string): string {
  return s
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&#(\d+);/g, (_m, d: string) => String.fromCharCode(Number(d)))
    .replace(/&amp;/g, "&");
}

function parseFunctionXml(text: string): InlineToolCall[] {
  const out: InlineToolCall[] = [];
  // 与格式 2 同一个开标签形态（复用那份正则）；解析器函数会把 lastIndex 留在非 0 处，先归零。
  FUNC_OPEN_RE.lastIndex = 0;
  for (const m of text.matchAll(FUNC_OPEN_RE)) {
    const tagName = m[1] ?? "";
    if (tagName === "") continue;
    const start = m.index;
    const bodyStart = m.index + m[0].length;
    const closeIdx = text.indexOf(FUNC_CLOSE, bodyStart);
    let nextOpenIdx = -1;
    let probe = bodyStart;
    for (;;) {
      const t = text.indexOf(FUNC_OPEN_MARKER, probe);
      if (t === -1) break;
      const c = text[t + FUNC_OPEN_MARKER.length];
      if (c !== undefined && /^[A-Za-z0-9_.-]/.test(c)) { nextOpenIdx = t; break; }
      probe = t + 1;
    }
    let end: number;
    let bodyEnd: number;
    if (closeIdx !== -1 && (nextOpenIdx === -1 || closeIdx < nextOpenIdx)) {
      bodyEnd = closeIdx;
      end = closeIdx + FUNC_CLOSE.length;
    } else {
      bodyEnd = nextOpenIdx === -1 ? text.length : nextOpenIdx;
      end = bodyEnd;
    }
    const raw = text.slice(bodyStart, bodyEnd);
    // 正文里有参数元素才认这种格式（JSON 正文是格式 2 的事）。
    if (!raw.includes(XML_PARAM_MARKER)) continue;
    XML_PARAM_RE.lastIndex = 0;
    const args: Record<string, string> = {};
    for (const pm of raw.matchAll(XML_PARAM_RE)) {
      const k = pm[1] ?? "";
      if (k === "") continue;
      args[k] = unescapeXml(pm[2] ?? "");
    }
    // 一个完整参数都没有 → 当普通文本（半截参数交给末尾截断那条规则）。
    if (Object.keys(args).length === 0) continue;
    out.push({
      name: tagName,
      arguments: JSON.stringify(args),
      start,
      end,
      format: "function-xml",
    });
  }
  return out;
}

/* ---------------- Display-side stripping (no promotion) -------------- */

/**
 * 文本里是否出现任何一种内联工具调用的开标签（廉价 indexOf 探测）。
 *
 * 给流式侧做前置判断用：先跑这个便宜的检查，命中才值得跑完整的
 * `parseInlineToolCalls`（那是多次全串扫描 + JSON 解析）。
 */
export function hasInlineToolCallMarker(text: string): boolean {
  return (
    text.includes(HERMES_OPEN) ||
    text.includes(FUNC_OPEN_MARKER) ||
    text.includes(BRACKET_MARKER) ||
    text.includes(GEMMA_OPEN) ||
    text.includes(GEMMA_FENCE_OPEN)
  );
}

export type StripResult = {
  /** 去掉工具调用跨度后的文本（其余字符一个不动，不做空白归一）。 */
  text: string;
  /** 被移除的调用名（去重、按出现顺序）；末尾未闭合的调用能找回名字时也并入。 */
  removedNames: string[];
};

/**
 * 展示侧清除 —— 与 `healToolCalls` 相对：
 *
 * - heal 是「提升」：只提升允许名单里的名字，其余保留为正文（拿去执行）；
 * - strip 是「清场」：调用方**一个工具都没有**（对话页 / 历史重放），于是任何
 *   成形的内联工具调用都只能是模型的产物而不是内容，不论工具名一律移除。
 *
 * 移除三层东西（后两层是真实泄漏形态逼出来的：mimo 蒸馏模型把 hermes 开标签
 * —— 单行开标签，里面包 function-xml 调用 —— 当正文吐：内层调用被解析器移除后
 * 只剩一个空 hermes 外壳，而 hermes 解析器按「正文是 JSON」认调用、认不出空壳）：
 *   1. 所有成形跨度（五个格式，复用解析器）；
 *   2. ① 之后剩下的**空开闭对**（空外壳）；
 *   3. **末尾未闭合**的开标签（流式被中断 / 输出截断，调用写到一半就没了）：
 *      从开标签处截断。
 *
 * 不变量与库内其余函数一致：任何解析失败都按「没见过」处理，绝不抛错；
 * 只删工具调用跨度（及未闭合尾巴），其余字符原样保留。
 */
export function stripInlineToolCalls(text: string): StripResult {
  if (text.length > MAX_HEAL_INPUT_CHARS) return { text, removedNames: [] };

  const removedNames: string[] = [];
  let out = "";
  let cursor = 0;
  for (const c of parseInlineToolCalls(text)) {
    if (c.start < cursor) continue; // 与已移除跨度重叠（内层标签），跳过
    out += text.slice(cursor, c.start);
    cursor = c.end;
    if (!removedNames.includes(c.name)) removedNames.push(c.name);
  }
  out += text.slice(cursor);

  out = removeEmptyWrappers(out);

  const tail = findUnterminatedTail(out);
  if (tail !== null) {
    out = out.slice(0, tail.at);
    if (typeof tail.name === "string" && tail.name.length > 0 && !removedNames.includes(tail.name)) removedNames.push(tail.name);
  }

  return { text: out, removedNames };
}

/**
 * 移除「空」开闭对：开闭标签之间没有任何内容（或全空白）。
 *
 * 发生在复合形态（hermes 外壳包 function 调用）的内层被上一遍移除之后：
 * 外壳还在、身体空了。hermes / gemma 解析器都按「正文是 JSON」认调用，
 * 空正文认不出，不补这条规则空外壳就会原样留在会话里。
 *
 * 只处理 hermes 与 gemma（bracket 的空体由末尾截断那条规则覆盖）。
 * 每移除一个就从原位继续扫（数量极少，文本 ≤ MAX_HEAL_INPUT_CHARS，代价可忽略）。
 */
function removeEmptyWrappers(text: string): string {
  let out = text;
  for (const [open, close] of [
    [HERMES_OPEN, HERMES_CLOSE],
    [GEMMA_OPEN, GEMMA_CLOSE],
  ] as const) {
    let i = out.indexOf(open);
    while (i !== -1) {
      const bodyStart = i + open.length;
      const closeIdx = out.indexOf(close, bodyStart);
      if (closeIdx !== -1 && out.slice(bodyStart, closeIdx).trim() === "") {
        out = out.slice(0, i) + out.slice(closeIdx + close.length);
        i = out.indexOf(open, Math.max(0, i + close.length));
      } else {
        i = out.indexOf(open, bodyStart);
      }
    }
  }
  return out;
}

type TailCut = { at: number; name?: string };

/**
 * 找「开了头没写完」的末尾工具调用（流式被中断 / 输出截断时留下）。
 *
 * 找到时返回截断位置（开标签本身：它之前是真正文，从它起是没写完的调用）
 * 与能找回的工具名（给「模型想调用 X」那条说明用）。
 *
 * 只认「像真中断」的形态，避免把正文里顺嘴提到的未闭合标签截掉：
 *   - hermes / gemma 开标签：开标签后的正文为空（内层已被移除）、或以 `{`
 *     （JSON 开头）/ function 开标签（复合形态）起头，且其后没有对应闭合标签
 *     —— 纯文字讨论通常以空格 + 普通词开头，不会被截；
 *   - function 开标签：其后没有闭合标签，且正文以 `{` 起头或含参数元素；
 *   - `[TOOL_CALLS]` / tool_code 围栏：其后没有对应的 `]` / 闭合围栏。
 */
function findUnterminatedTail(text: string): TailCut | null {
  let best: TailCut | null = null;
  const consider = (at: number, name?: string) => {
    if (best === null || at < best.at) best = { at, name };
  };

  for (const [open, close] of [
    [HERMES_OPEN, HERMES_CLOSE],
    [GEMMA_OPEN, GEMMA_CLOSE],
  ] as const) {
    const i = text.lastIndexOf(open);
    if (i === -1) continue;
    const bodyStart = i + open.length;
    if (text.indexOf(close, bodyStart) !== -1) continue; // 有闭合 → 不是未闭合尾
    const body = text.slice(bodyStart).trimStart();
    if (body !== "" && !body.startsWith("{") && !body.startsWith(FUNC_OPEN_MARKER)) continue; // 不像真调用
    consider(i, extractToolNameFromBody(text.slice(bodyStart)) ?? undefined);
  }

  // function 开标签：取最后一个符合解析器形态的。
  let lastFt: { at: number; name: string; bodyStart: number } | null = null;
  // 解析器函数会把 lastIndex 留在非 0 处，matchAll 会继承它 —— 先归零，
  // 否则从上次位置之后才开始找，漏掉真正在前的开标签。
  FUNC_OPEN_RE.lastIndex = 0;
  for (const m of text.matchAll(FUNC_OPEN_RE)) {
    lastFt = { at: m.index, name: m[1] ?? "", bodyStart: m.index + m[0].length };
  }
  if (lastFt !== null && lastFt.name !== "" && text.indexOf(FUNC_CLOSE, lastFt.bodyStart) === -1) {
    const after = text.slice(lastFt.bodyStart).trimStart();
    if (after.startsWith("{") || after.includes(XML_PARAM_MARKER)) {
      consider(lastFt.at, lastFt.name);
    }
  }

  const bracket = text.lastIndexOf(BRACKET_MARKER);
  if (bracket !== -1 && !text.slice(bracket + BRACKET_MARKER.length).includes("]")) {
    consider(bracket);
  }

  const fence = text.lastIndexOf(GEMMA_FENCE_OPEN);
  if (fence !== -1 && text.indexOf(GEMMA_FENCE_CLOSE, fence + GEMMA_FENCE_OPEN.length) === -1) {
    consider(fence);
  }

  return best;
}

/** 从（可能已损坏的）调用正文里找回工具名：先找内层 function 开标签，再找 `"name":"…"`。 */
function extractToolNameFromBody(body: string): string | null {
  // 内层形态与解析器一致，直接复用同一份正则源（非全局拷贝，不动共享的 lastIndex）。
  const inner = new RegExp(FUNC_OPEN_RE.source).exec(body);
  if (inner) return inner[1] ?? null;
  const named = /"name"\s*:\s*"([^"\n]{1,128})"/.exec(body);
  return named ? named[1] ?? null : null;
}
