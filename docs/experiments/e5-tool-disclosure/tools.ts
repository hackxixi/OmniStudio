/**
 * E5 的工具：真实 Agent 的 34 个工具（real-tools.json 快照）+ 路线图里要补的感知 / 日程通讯工具，
 * 以及精简方案新增的合并工具（recall / remember / find）和按需加载的元工具 load_tools。
 *
 * 各组对比用的工具清单都在 variantTools() 里拼出来；桩 runStub() 同时认真实工具名与合并后的工具名。
 */
import realTools from "./real-tools.json";

export type ToolDef = { name: string; description: string; parameters: Record<string, unknown> };

const real = new Map((realTools as (ToolDef & { group: string })[]).map((t) => [t.name, { name: t.name, description: t.description, parameters: t.parameters }]));
const r = (name: string): ToolDef => {
  const t = real.get(name);
  if (!t) throw new Error(`no real tool ${name}`);
  return t;
};
const str = (description: string) => ({ type: "string", description });
const obj = (properties: Record<string, unknown>, required: string[]) => ({ type: "object", properties, required });

/** 真实 Agent 现有的 34 个工具，原样。 */
export const REAL_ALL: ToolDef[] = [...real.values()];

/** 路线图里还没接进 Agent 的能力（B1 感知侧、以及 MCP 常见的日程通讯），代表「不久之后的完整工具集」。 */
export const PERCEPTION: ToolDef[] = [
  { name: "transcribe_audio", description: "Transcribe an attached audio or video recording into text.", parameters: obj({ file: str("File name of the attached recording") }, ["file"]) },
  { name: "ocr_document", description: "Extract the text from an attached scanned document, photo of a page or PDF.", parameters: obj({ file: str("File name of the attached document") }, ["file"]) },
];
export const SCHEDULE: ToolDef[] = [
  { name: "set_reminder", description: "Create a one-off reminder notification at a given time.", parameters: obj({ time: str("When, ISO 8601 local time, e.g. 2026-09-25T09:00"), message: str("What to remind about") }, ["time", "message"]) },
  {
    name: "create_calendar_event",
    description: "Add an event (meeting, appointment, deadline) to the user's calendar.",
    parameters: obj({ title: str("Event title"), start: str("Start, ISO 8601 local time"), end: str("End, ISO 8601 local time (optional)") }, ["title", "start"]),
  },
  { name: "contacts_lookup", description: "Look up a person in the user's contacts to get their email address.", parameters: obj({ name: str("Person's name") }, ["name"]) },
  { name: "send_email", description: "Send an email.", parameters: obj({ to: str("Recipient email address"), subject: str("Subject line"), body: str("Email body") }, ["to", "subject", "body"]) },
];

/** 精简方案：知识库 / 笔记 / 记忆三组近义检索合成一个，文件查找三件套合成一个。 */
export const RECALL: ToolDef = {
  name: "recall",
  description:
    "Search what the user already has: the company knowledge base (policies, contracts), the user's own notes, and facts the user told you before. " +
    "Results are labeled with where they came from. Use source to narrow it down when the user says where to look.",
  parameters: obj(
    {
      query: str("What to look for"),
      source: { type: "string", enum: ["any", "knowledge_base", "notes", "memory"], description: "Where to look; default any" },
    },
    ["query"],
  ),
};
export const REMEMBER: ToolDef = {
  name: "remember",
  description: "Save a lasting fact about the user (a preference, a personal detail) so it is available in future conversations. One sentence.",
  parameters: obj({ content: str("The fact, as one sentence") }, ["content"]),
};
export const FIND: ToolDef = {
  name: "find",
  description: "Find files in the user's workspace by file name pattern (e.g. *budget*) or by text inside files. Returns matching paths.",
  parameters: obj({ pattern: str("File name glob or text to search for"), path: str("Folder to search in; default the whole workspace") }, ["pattern"]),
};

/** 工具组（渐进式披露的第 1 层）。summary 是系统提示里常驻的一行说明。 */
export const GROUPS: Record<string, { summary: string; tools: ToolDef[] }> = {
  creation: {
    summary: "creation — generate images, speech (text-to-speech) and videos; find and reuse media generated earlier",
    tools: ["generate_image", "generate_speech", "generate_video", "media_search", "media_export"].map(r),
  },
  perception: {
    summary: "perception — transcribe attached recordings, read (OCR) attached scanned documents and PDFs",
    tools: PERCEPTION,
  },
  schedule: {
    summary: "schedule — reminders, calendar events, contacts and sending email",
    tools: SCHEDULE,
  },
  dev: {
    summary: "dev — run shell commands and apply multi-file code patches",
    tools: ["bash", "apply_patch"].map(r),
  },
};

export const LOAD_TOOLS: ToolDef = {
  name: "load_tools",
  description:
    "Load an extra group of tools when none of your current tools can do what the user asked. Groups: " +
    Object.values(GROUPS)
      .map((g) => g.summary)
      .join("; ") +
    ". After loading, the group's tools are described in the result and you can call them.",
  parameters: obj({ group: { type: "string", enum: Object.keys(GROUPS), description: "Which group to load" } }, ["group"]),
};

/** 精简后常驻的核心工具（第 0 层）。 */
export const CORE: ToolDef[] = [r("web_search"), r("web_fetch"), RECALL, REMEMBER, r("read_file"), r("write_file"), r("edit_file"), FIND, r("ask_user")];

export type Variant = "A" | "B" | "C" | "D" | "E";

/** 各组一开始放进工具列表（即提示前缀）的工具。C / D 再加上选中的组。 */
export function variantTools(v: Variant, groups: string[] = []): ToolDef[] {
  switch (v) {
    case "A":
      return [...REAL_ALL, ...PERCEPTION, ...SCHEDULE];
    case "B":
      return [...CORE, ...Object.values(GROUPS).flatMap((g) => g.tools)];
    default:
      return [...CORE, ...groups.flatMap((g) => GROUPS[g]!.tools), LOAD_TOOLS];
  }
}

// ---------------------------------------------------------------------------
// 桩
// ---------------------------------------------------------------------------

const FILES: Record<string, string> = {
  "notes/todo.md": "- [x] 提交周报\n- [ ] 报销差旅发票\n- [ ] 预约牙医\n- [x] 续费域名",
  "docs/plan.md": "# 上线计划\n\n- 功能冻结：9 月 30 日\n- 上线日期：10 月 8 日\n- 负责人：张伟",
  "finance/budget-2026.xlsx": "(binary spreadsheet)",
};

let counters: Record<string, number> = {};
const next = (p: string) => {
  counters[p] = (counters[p] ?? 0) + 1;
  return `${p}_${counters[p]}`;
};
export function resetStubs() {
  counters = {};
}

export function runStub(tool: string, args: Record<string, unknown>): string {
  const s = (k: string) => String(args[k] ?? "");
  const q = (s("query") || s("pattern")).toLowerCase();
  const src = s("source");
  const kb = () => (/差旅|住宿|报销|travel|hotel/.test(q) ? "[knowledge_base] 《差旅管理办法 v3》第 4 条：一线城市住宿标准每晚不超过 450 元，其他城市 350 元。" : "");
  const notes = () => (/向量|vector|数据库|database/.test(q) ? "[notes] 笔记《向量数据库选型》(2026-08)：对比 Milvus / Qdrant / pgvector，结论：选 Qdrant。" : "");
  const memory = () => (/音乐|music|喜欢|like|偏好|prefer/.test(q) ? "[memory] The user likes jazz music and cats; allergic to peanuts." : "");
  switch (tool) {
    case "web_search":
      if (/bun/.test(q)) return "Bun v1.4.7 released on 2026-09-18: faster installs and a new SQL driver.";
      if (/qdrant/.test(q)) return "Qdrant v1.15.2 released on 2026-09-10 (latest stable).";
      return "No relevant results.";
    case "web_fetch":
      if (/example-blog\.dev\/valkey/.test(s("url"))) return "Why we moved from Redis to Valkey: after the Redis license change we switched to the BSD-licensed Valkey fork.";
      if (/bun/i.test(s("url"))) return "Bun v1.4.7 (2026-09-18) release notes: faster installs, a new SQL driver, bug fixes.";
      if (/qdrant/i.test(s("url"))) return "Qdrant v1.15.2 (2026-09-10) release notes.";
      return "The page loaded but has nothing relevant to the question.";
    case "knowledge_search":
      return kb() || "No matching documents.";
    case "note_search":
      return notes() || "No matching notes.";
    case "note_list":
      return "#12 向量数据库选型 (2026-08)\n#15 周会记录 (2026-09)";
    case "note_read":
      return s("id") === "12" ? "笔记《向量数据库选型》：结论：选 Qdrant。" : "Note not found.";
    case "memory_search":
      return memory() || "Nothing remembered about that.";
    case "recall": {
      const hits = [
        src === "notes" || src === "memory" ? "" : kb(),
        src === "knowledge_base" || src === "memory" ? "" : notes(),
        src === "knowledge_base" || src === "notes" ? "" : memory(),
      ].filter(Boolean);
      return hits.join("\n") || "Nothing found.";
    }
    case "memory_save":
    case "remember":
      return "Saved to memory.";
    case "read_file": {
      const path = s("path").replace(/^\.?\//, "");
      return FILES[path] ?? `Error: ${path} not found.`;
    }
    case "write_file":
      return `Wrote ${s("content").length} chars to ${s("path")}.`;
    case "edit_file":
      return FILES[s("path").replace(/^\.?\//, "")]?.includes(s("old_str")) ? `Edited ${s("path")}.` : "Error: old_str not found in file.";
    case "find":
    case "glob":
    case "grep":
    case "list_dir":
    case "bash":
      if (/budget|预算|xlsx|表格/.test(q + s("command") + s("path"))) return "finance/budget-2026.xlsx";
      return tool === "bash" || tool === "list_dir" ? "docs/  finance/  notes/" : "No matches.";
    case "ask_user":
      return "The user has not answered yet; end your turn and wait.";
    case "generate_image":
      return `Image saved as ${next("img")}.`;
    case "generate_speech":
      return `Audio saved as ${next("aud")}.`;
    case "generate_video":
      return `Video saved as ${next("vid")}.`;
    case "media_search":
      return /猫|cat/.test(q) ? "img_prev_7 — orange cat wearing an astronaut helmet (2026-09-20)" : "No matching media.";
    case "media_export":
      return `Copied ${JSON.stringify(args.refs)} to ${s("save_to")}.`;
    case "transcribe_audio":
      return /weekly-sync/.test(s("file"))
        ? "[03:40] 李娜：那就把上线从 10 月 8 日推到 10 月 15 日。[04:02] 张伟：同意，就定 10 月 15 日。"
        : "Error: file not found.";
    case "ocr_document":
      return /invoice/.test(s("file")) ? "增值税普通发票 No.20260917 金额合计 ¥3,860.00 付款截止日：2026-10-10" : "Error: file not found.";
    case "set_reminder":
      return `Reminder set for ${s("time")}.`;
    case "create_calendar_event":
      return `Calendar event "${s("title")}" created at ${s("start")}.`;
    case "contacts_lookup":
      return /王|wang/i.test(s("name")) ? "王总 (Wang Lei), wang.lei@kunpeng.example" : "No contact found.";
    case "send_email":
      return `Email sent to ${s("to")}.`;
    case "load_tools": {
      const g = GROUPS[s("group")];
      if (!g) return `Unknown group. Available: ${Object.keys(GROUPS).join(", ")}`;
      return `Loaded ${s("group")} tools. You can now call them:\n<tools>\n${g.tools.map((t) => JSON.stringify({ type: "function", function: t })).join("\n")}\n</tools>`;
    }
    default:
      return `Tool ${tool} ran.`;
  }
}

/**
 * 任务没要求时也允许的额外调用（至多 2 次）：只读工具，以及只往工作区复制文件、可撤销的 media_export
 * （真实 generate_image 的描述本身就引导模型把图存进工作区）。其余有副作用的多余调用判错。
 */
export const READ_ONLY = new Set([
  "media_export",
  "web_search", "web_fetch", "knowledge_search", "note_search", "note_list", "note_read", "memory_search", "recall",
  "read_file", "find", "glob", "grep", "list_dir", "media_search", "load_tools", "think", "todo_write", "get_context_remaining", "read_skill",
]);
