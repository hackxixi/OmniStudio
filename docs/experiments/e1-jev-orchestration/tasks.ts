/**
 * E1 任务集：L2 单步任务（一个工具 + 最终回答）、L1 干扰项（不该调工具）、少量 L3 两步任务。
 *
 * 工具全部是确定性桩：同一个参数永远返回同样的结果，所以三组的差异只来自「谁来决定、谁来填参数」。
 * 桩结果里埋了**只能从工具拿到**的信息（编号、数字、专名），最终回答的检查靠它们判断模型
 * 是真用了工具结果，还是自己编的。
 */

export type ToolSpec = {
  name: string;
  description: string;
  /** JSON Schema（同时用于 A/C 组的 tools 声明与 B 组的约束解码）。 */
  parameters: {
    type: "object";
    properties: Record<string, { type: string; description: string; enum?: string[] }>;
    required: string[];
  };
  /** 需要某类输入的工具（附件或前面步骤的产物）：没有时 B 组把它从选项里拿掉。 */
  needs?: "audio" | "document" | "image";
  /** HINTS=1 时追加到描述后面的说明（工具侧修正：把枚举值的含义写清楚）。 */
  hint?: string;
};

export const TOOLS: ToolSpec[] = [
  {
    name: "web_search",
    description: "Search the web for up-to-date public information. Returns the top results as text.",
    parameters: {
      type: "object",
      properties: { query: { type: "string", description: "Search query" } },
      required: ["query"],
    },
  },
  {
    name: "knowledge_search",
    description: "Search the user's private knowledge base (their own documents, notes and company files).",
    parameters: {
      type: "object",
      properties: { query: { type: "string", description: "What to look for" } },
      required: ["query"],
    },
  },
  {
    name: "generate_image",
    description: "Generate an image from a text prompt. Returns a reference to the saved image.",
    parameters: {
      type: "object",
      properties: {
        prompt: { type: "string", description: "Detailed description of the image" },
        size: { type: "string", description: "Image size", enum: ["1024x1024", "1024x1792", "1792x1024"] },
      },
      required: ["prompt"],
    },
  },
  {
    name: "generate_speech",
    description: "Convert text to spoken audio (text-to-speech). Returns a reference to the saved audio file.",
    parameters: {
      type: "object",
      properties: {
        text: { type: "string", description: "The exact text to speak" },
        voice: { type: "string", description: "Voice style", enum: ["female", "male", "child"] },
      },
      required: ["text"],
    },
  },
  {
    name: "transcribe_audio",
    description: "Transcribe an attached audio or video recording into text (speech recognition).",
    parameters: {
      type: "object",
      properties: { file: { type: "string", description: "File name of the attached recording" } },
      required: ["file"],
    },
    needs: "audio",
  },
  {
    name: "ocr_document",
    description: "Extract the text from an attached scanned document, PDF or photo of a page (OCR).",
    parameters: {
      type: "object",
      properties: { file: { type: "string", description: "File name of the attached document or image" } },
      required: ["file"],
    },
    needs: "document",
  },
  {
    name: "memory_save",
    description: "Save a fact about the user to long-term memory so it can be recalled in future conversations.",
    parameters: {
      type: "object",
      properties: { content: { type: "string", description: "The fact to remember, as one sentence" } },
      required: ["content"],
    },
  },
  {
    name: "set_reminder",
    description: "Create a reminder that notifies the user at a given time.",
    parameters: {
      type: "object",
      properties: {
        time: { type: "string", description: "When to remind, ISO 8601 local time, e.g. 2026-09-25T09:00" },
        message: { type: "string", description: "What to remind the user about" },
      },
      required: ["time", "message"],
    },
  },
];

/** 参数检查：参数名 → 值必须匹配的正则（不区分大小写）。 */
type ArgCheck = Record<string, RegExp>;

export type Step = { tool: string; args: ArgCheck };

export type Task = {
  id: string;
  level: "L1" | "L2" | "L3";
  request: string;
  attachments?: { name: string; kind: "audio" | "document" | "image" }[];
  /** 期望的工具调用序列；空数组 = 不该调任何工具。 */
  steps: Step[];
  /** 最终回答必须命中的每一组关键词（组内任一即可）。 */
  answer: RegExp[];
  /** 步骤之间没有先后依赖，顺序不计。 */
  anyOrder?: boolean;
  /** 信息不足，正确做法是反问用户（不调工具、回答里带问号）。 */
  ask?: boolean;
};

/** 「现在」固定下来，定提醒类任务的日期。 */
export const NOW = "2026-09-24T15:00 (Thursday)";

export const TASKS: Task[] = [
  // ---------------- L2：单步 ----------------
  {
    id: "search-release",
    level: "L2",
    request: "帮我查一下 Bun 最新发布的版本号是多少",
    steps: [{ tool: "web_search", args: { query: /bun/ } }],
    answer: [/1\.4\.7/],
  },
  {
    id: "search-weather-en",
    level: "L2",
    request: "What's the weather forecast for Shenzhen tomorrow?",
    steps: [{ tool: "web_search", args: { query: /shenzhen|深圳/ } }],
    answer: [/27/, /thunder|雷/],
  },
  {
    id: "kb-policy",
    level: "L2",
    request: "我们公司的差旅报销住宿标准是多少？查一下我的知识库",
    steps: [{ tool: "knowledge_search", args: { query: /差旅|住宿|报销|travel|hotel/ } }],
    answer: [/450/],
  },
  {
    id: "kb-contract",
    level: "L2",
    request: "跟星河科技签的那份合同，付款账期是几天？",
    steps: [{ tool: "knowledge_search", args: { query: /星河|合同|账期/ } }],
    answer: [/45/],
  },
  {
    id: "image-cat",
    level: "L2",
    request: "画一只戴着宇航员头盔的橘猫，竖版的，我要做手机壁纸",
    steps: [{ tool: "generate_image", args: { prompt: /猫|cat/, size: /1024x1792/ } }],
    answer: [/img_/],
  },
  {
    id: "image-logo-en",
    level: "L2",
    request: "Create a minimalist logo for a coffee shop called Moonbean.",
    steps: [{ tool: "generate_image", args: { prompt: /moonbean/ } }],
    answer: [/img_/],
  },
  {
    id: "tts-greeting",
    level: "L2",
    request: "把「欢迎来到鲲鹏工作室，今天也要加油」这句话用男声读出来",
    steps: [{ tool: "generate_speech", args: { text: /欢迎来到鲲鹏工作室/, voice: /^male$/ } }],
    answer: [/aud_/],
  },
  {
    id: "tts-story-en",
    level: "L2",
    request: 'Read this aloud in a child\'s voice: "The little fox found a golden key under the old oak tree."',
    steps: [{ tool: "generate_speech", args: { text: /golden key/, voice: /child/ } }],
    answer: [/aud_/],
  },
  {
    id: "asr-meeting",
    level: "L2",
    request: "这段会议录音里，最后定下来的上线日期是哪天？",
    attachments: [{ name: "weekly-sync.m4a", kind: "audio" }],
    steps: [{ tool: "transcribe_audio", args: { file: /weekly-sync\.m4a/ } }],
    answer: [/10\s*月\s*15|10-15|10\/15|October 15/],
  },
  {
    id: "asr-voice-memo",
    level: "L2",
    request: "Transcribe my voice memo.",
    attachments: [{ name: "memo-0923.wav", kind: "audio" }],
    steps: [{ tool: "transcribe_audio", args: { file: /memo-0923\.wav/ } }],
    answer: [/dentist/],
  },
  {
    id: "ocr-invoice",
    level: "L2",
    request: "这张发票的金额是多少？",
    attachments: [{ name: "invoice-scan.jpg", kind: "document" }],
    steps: [{ tool: "ocr_document", args: { file: /invoice-scan\.jpg/ } }],
    answer: [/3,?860/],
  },
  {
    id: "ocr-contract-en",
    level: "L2",
    request: "What is the termination notice period in the attached lease?",
    attachments: [{ name: "lease.pdf", kind: "document" }],
    steps: [{ tool: "ocr_document", args: { file: /lease\.pdf/ } }],
    answer: [/60/],
  },
  {
    id: "memory-allergy",
    level: "L2",
    request: "记住：我对花生过敏，以后推荐菜谱的时候注意",
    steps: [{ tool: "memory_save", args: { content: /花生|peanut/ } }],
    answer: [/记住|记下|保存|noted|saved|remember/],
  },
  {
    id: "memory-name-en",
    level: "L2",
    request: "Please remember that my manager's name is Priya Raman.",
    steps: [{ tool: "memory_save", args: { content: /priya/ } }],
    answer: [/remember|saved|noted|got it|记/],
  },
  {
    id: "reminder-tomorrow",
    level: "L2",
    request: "明天早上九点提醒我给房东转房租",
    steps: [{ tool: "set_reminder", args: { time: /2026-09-25T0?9:00/, message: /房租|房东/ } }],
    answer: [/提醒|remind/],
  },
  {
    id: "reminder-monday-en",
    level: "L2",
    request: "Remind me next Monday at 2pm to submit the quarterly report.",
    steps: [{ tool: "set_reminder", args: { time: /2026-09-28T14:00/, message: /quarterly|report/ } }],
    answer: [/remind/],
  },
  {
    id: "search-vs-kb",
    level: "L2",
    request: "我上个月写的那篇关于向量数据库选型的笔记，结论是选了哪个？",
    steps: [{ tool: "knowledge_search", args: { query: /向量|vector/ } }],
    answer: [/qdrant/i],
  },
  {
    id: "search-news",
    level: "L2",
    request: "最近魔搭社区上新发布了哪个 Qwen 小模型？",
    steps: [{ tool: "web_search", args: { query: /qwen|魔搭|modelscope/ } }],
    answer: [/qwen3\.5-4b/i],
  },
  // ---------------- L1：不该调工具 ----------------
  {
    id: "l1-translate",
    level: "L1",
    request: "把这句话翻译成英文：今天的会议改到下午三点。",
    steps: [],
    answer: [/3|three/, /meeting/],
  },
  {
    id: "l1-rewrite",
    level: "L1",
    request: "帮我把「收到，马上处理」改得更礼貌一点，给客户回邮件用",
    steps: [],
    answer: [/.{6,}/],
  },
  {
    id: "l1-math-en",
    level: "L1",
    request: "If a train travels 180 km in 1.5 hours, what is its average speed?",
    steps: [],
    answer: [/120/],
  },
  {
    id: "l1-explain",
    level: "L1",
    request: "用一句话解释什么是量化（模型量化）",
    steps: [],
    answer: [/精度|位|bit|压缩|权重/],
  },
  // ---------------- L3：两步 ----------------
  {
    id: "l3-memo-to-speech",
    level: "L3",
    request: "把这段录音转成文字，然后用女声把内容读一遍给我",
    attachments: [{ name: "memo-0923.wav", kind: "audio" }],
    steps: [
      { tool: "transcribe_audio", args: { file: /memo-0923\.wav/ } },
      { tool: "generate_speech", args: { text: /dentist/, voice: /female/ } },
    ],
    answer: [/aud_/],
  },
  {
    id: "l3-invoice-reminder",
    level: "L3",
    request: "看一下这张发票的付款截止日，截止日前一天上午十点提醒我付款",
    attachments: [{ name: "invoice-scan.jpg", kind: "document" }],
    steps: [
      { tool: "ocr_document", args: { file: /invoice-scan\.jpg/ } },
      { tool: "set_reminder", args: { time: /2026-10-09T10:00/, message: /发票|付款|invoice|pay/ } },
    ],
    answer: [/提醒|remind/],
  },
  {
    id: "l3-search-image",
    level: "L3",
    request: "查一下深圳明天天气，然后按天气画一张适合发朋友圈的横版插画",
    steps: [
      { tool: "web_search", args: { query: /深圳|shenzhen/ } },
      { tool: "generate_image", args: { prompt: /雷|thunder|storm|rain|雨/, size: /1792x1024/ } },
    ],
    answer: [/img_/],
  },
  {
    id: "l3-kb-memory",
    level: "L3",
    request: "查一下知识库里我们的差旅住宿标准，把这个标准记到长期记忆里",
    steps: [
      { tool: "knowledge_search", args: { query: /差旅|住宿|travel|hotel/ } },
      { tool: "memory_save", args: { content: /450/ } },
    ],
    answer: [/450|记/],
  },
];

let imageCounter = 0;
let audioCounter = 0;

export function resetStubs(): void {
  imageCounter = 0;
  audioCounter = 0;
}

/** 桩工具：只看参数，不看任务 —— 模型得自己把对的参数填进来才拿得到对的信息。 */
export function runStub(tool: string, args: Record<string, unknown>): string {
  const s = (k: string) => String(args[k] ?? "");
  const q = s("query").toLowerCase();
  switch (tool) {
    case "web_search":
      if (/bun/.test(q)) return "1. Bun v1.4.7 released (2026-09-18) — changelog: faster installs, new SQL driver.\n2. Bun v1.4.6 notes.";
      if (/shenzhen|深圳/.test(q)) return "Shenzhen forecast for 2026-09-25: thunderstorms in the afternoon, 27°C–33°C, humidity 85%.";
      if (/qwen|魔搭|modelscope/.test(q)) return "ModelScope news: Qwen/Qwen3.5-4B released this week, a 4B multimodal model with tool calling.";
      return "No relevant results.";
    case "knowledge_search":
      if (/差旅|住宿|报销|travel|hotel/.test(q)) return "《差旅管理办法 v3》第 4 条：一线城市住宿标准每晚不超过 450 元，其他城市 350 元。";
      if (/星河|合同|账期|contract/.test(q)) return "《星河科技技术服务合同》第 7.2 条：甲方应在验收合格后 45 日内付款。";
      if (/向量|vector|数据库/.test(q)) return "笔记《向量数据库选型》(2026-08)：对比 Milvus / Qdrant / pgvector，结论：选 Qdrant（部署简单、过滤性能好）。";
      return "No matching documents.";
    case "generate_image":
      imageCounter++;
      return `Image saved as img_${imageCounter} (${s("size") || "1024x1024"}).`;
    case "generate_speech":
      audioCounter++;
      return `Audio saved as aud_${audioCounter} (${s("voice") || "female"} voice, ${Math.max(1, Math.round(s("text").length / 6))}s).`;
    case "transcribe_audio":
      if (/weekly-sync/.test(s("file"))) return "[00:12] 张伟：测试还有两个阻塞问题。[03:40] 李娜：那就把上线从 10 月 8 日推到 10 月 15 日。[04:02] 张伟：同意，就定 10 月 15 日。";
      if (/memo-0923/.test(s("file"))) return "Don't forget the dentist appointment on Friday at 4pm, and buy printer ink.";
      return "Error: file not found.";
    case "ocr_document":
      if (/invoice-scan/.test(s("file"))) return "增值税普通发票 No.20260917\n购买方：鲲鹏工作室\n金额合计（大写）叁仟捌佰陆拾元整 ¥3,860.00\n付款截止日：2026-10-10";
      if (/lease/.test(s("file"))) return "Section 12. Termination: either party may terminate this lease with 60 days' written notice.";
      return "Error: file not found.";
    case "memory_save":
      return "Saved to memory.";
    case "set_reminder":
      return `Reminder set for ${s("time")}.`;
    default:
      return `Error: unknown tool ${tool}.`;
  }
}
