/**
 * E1 第二轮任务集：把任务推到小模型会崩的区域。
 *
 * - 23 个工具，含容易混淆的近义工具（天气 / 网页搜索，知识库 / 笔记 / 记忆，提醒 / 日历，
 *   生图 / 改图 / 抠图 / 找旧图）；
 * - L2 单步 13 题（专挑近义工具之间的区分）、L1 干扰 4 题（有诱人的工具但不该调）、
 *   信息不足该反问 4 题、L3 三到四步 10 题（步骤间有依赖）。
 *
 * 导出名与 `tasks.ts` 一致，`bench.ts` 用 `SET=r2` 切换。
 */
import type { Task, ToolSpec } from "./tasks";

export const NOW = "2026-09-24T15:00 (Thursday)";

/** HINTS=1 时放进上下文的日历（工具侧修正：相对日期不必让模型心算）。 */
export const CALENDAR = (() => {
  const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  const out: string[] = [];
  for (let i = 0; i < 14; i++) {
    const d = new Date(Date.UTC(2026, 8, 24 + i));
    out.push(`${d.toISOString().slice(0, 10)} ${days[d.getUTCDay()]}${i === 0 ? " (today)" : i === 1 ? " (tomorrow)" : ""}`);
  }
  return `Calendar: ${out.join(", ")}.`;
})();

const str = (description: string) => ({ type: "string", description });
const obj = (properties: ToolSpec["parameters"]["properties"], required: string[]): ToolSpec["parameters"] => ({
  type: "object",
  properties,
  required,
});

export const TOOLS: ToolSpec[] = [
  { name: "web_search", description: "Search the public web for up-to-date information (news, releases, facts).", parameters: obj({ query: str("Search query") }, ["query"]) },
  { name: "web_fetch", description: "Open a web page by URL and return its main text.", parameters: obj({ url: str("Full URL") }, ["url"]) },
  {
    name: "get_weather",
    description: "Get the weather forecast for a city on a given date.",
    parameters: obj({ city: str("City name"), date: str("Date, YYYY-MM-DD") }, ["city", "date"]),
  },
  {
    name: "currency_convert",
    description: "Convert an amount of money between currencies at today's exchange rate.",
    parameters: obj(
      { amount: { type: "number", description: "Amount to convert" }, from: str("ISO currency code, e.g. USD"), to: str("ISO currency code, e.g. CNY") },
      ["amount", "from", "to"],
    ),
  },
  { name: "knowledge_search", description: "Search the company knowledge base (policies, contracts, official documents).", parameters: obj({ query: str("What to look for") }, ["query"]) },
  { name: "note_search", description: "Search the user's own personal notes (things the user wrote down themselves).", parameters: obj({ query: str("What to look for") }, ["query"]) },
  { name: "memory_search", description: "Recall facts the user told the assistant earlier (preferences, personal details).", parameters: obj({ query: str("What to recall") }, ["query"]) },
  { name: "memory_save", description: "Save a fact about the user to long-term memory for future conversations.", parameters: obj({ content: str("The fact, as one sentence") }, ["content"]) },
  { name: "contacts_lookup", description: "Look up a person in the user's contacts to get their email address.", parameters: obj({ name: str("Person's name") }, ["name"]) },
  {
    name: "send_email",
    description: "Send an email.",
    parameters: obj({ to: str("Recipient email address"), subject: str("Subject line"), body: str("Email body") }, ["to", "subject", "body"]),
  },
  {
    name: "set_reminder",
    description: "Create a one-off reminder notification at a given time.",
    parameters: obj({ time: str("When, ISO 8601 local time, e.g. 2026-09-25T09:00"), message: str("What to remind about") }, ["time", "message"]),
  },
  {
    name: "create_calendar_event",
    description: "Add an event (meeting, appointment, deadline) to the user's calendar.",
    parameters: obj(
      { title: str("Event title"), start: str("Start, ISO 8601 local time"), end: str("End, ISO 8601 local time (optional)") },
      ["title", "start"],
    ),
  },
  {
    name: "generate_image",
    description: "Generate a new image from a text prompt.",
    parameters: obj(
      { prompt: str("Detailed description of the image"), size: { type: "string", description: "Image size", enum: ["1024x1024", "1024x1792", "1792x1024"] } },
      ["prompt", "size"],
    ),
    hint: "Sizes: 1024x1024 = square, 1024x1792 = portrait / vertical (竖版, phone wallpaper), 1792x1024 = landscape / horizontal (横版, poster, banner).",
  },
  {
    name: "edit_image",
    description: "Change the content of an existing image according to an instruction.",
    parameters: obj({ image: str("File name or reference of the image"), instruction: str("What to change") }, ["image", "instruction"]),
    needs: "image",
  },
  {
    name: "remove_background",
    description: "Cut out the main subject of an existing image and make the background transparent.",
    parameters: obj({ image: str("File name or reference of the image") }, ["image"]),
    needs: "image",
  },
  { name: "search_media", description: "Find images, audio or videos the assistant generated earlier.", parameters: obj({ query: str("What the media shows") }, ["query"]) },
  {
    name: "generate_video",
    description: "Generate a short video from a prompt, optionally starting from an existing image as the first frame.",
    parameters: obj(
      { prompt: str("What happens in the video"), first_frame: str("Optional file name or reference of an image to animate"), seconds: { type: "number", description: "Length in seconds" } },
      ["prompt"],
    ),
  },
  {
    name: "generate_music",
    description: "Compose instrumental music from a description.",
    parameters: obj({ prompt: str("Style and mood"), duration_seconds: { type: "number", description: "Length in seconds" } }, ["prompt", "duration_seconds"]),
  },
  {
    name: "generate_speech",
    description: "Convert text to spoken audio (text-to-speech).",
    parameters: obj({ text: str("The exact text to speak"), voice: { type: "string", description: "Voice", enum: ["female", "male", "child"] } }, ["text", "voice"]),
  },
  { name: "transcribe_audio", description: "Transcribe an attached recording into text.", parameters: obj({ file: str("File name of the recording") }, ["file"]), needs: "audio" },
  { name: "ocr_document", description: "Extract the text from an attached scanned document or PDF.", parameters: obj({ file: str("File name of the document") }, ["file"]), needs: "document" },
  { name: "read_file", description: "Read a text file from the user's workspace.", parameters: obj({ path: str("Relative path") }, ["path"]) },
  { name: "write_file", description: "Write a text file in the user's workspace.", parameters: obj({ path: str("Relative path"), content: str("Full file content") }, ["path", "content"]) },
];

const ASK = [/[?？]/];

export const TASKS: Task[] = [
  // ---------------- L2：近义工具之间的区分 ----------------
  { id: "c-weather", level: "L2", request: "深圳明天会下雨吗？", steps: [{ tool: "get_weather", args: { city: /深圳|shenzhen/, date: /2026-09-25/ } }], answer: [/雷|雨|rain|thunder/] },
  { id: "c-note", level: "L2", request: "我自己笔记里记的向量数据库选型结论是什么？", steps: [{ tool: "note_search", args: { query: /向量|vector/ } }], answer: [/qdrant/] },
  { id: "c-kb", level: "L2", request: "公司差旅制度里一线城市住宿标准是多少？", steps: [{ tool: "knowledge_search", args: { query: /差旅|住宿|travel|hotel/ } }], answer: [/450/] },
  { id: "c-memory", level: "L2", request: "我之前跟你说过我喜欢什么音乐来着？", steps: [{ tool: "memory_search", args: { query: /音乐|music|喜欢|like/ } }], answer: [/jazz|爵士/] },
  {
    id: "c-calendar",
    level: "L2",
    request: "明天下午三点到四点和王总开会，帮我放到日历上",
    steps: [{ tool: "create_calendar_event", args: { title: /王|wang/, start: /2026-09-25T15:00/ } }],
    answer: [/日历|calendar|创建|添加|added/],
  },
  {
    id: "c-reminder",
    level: "L2",
    request: "今晚八点提醒我给妈妈打电话",
    steps: [{ tool: "set_reminder", args: { time: /2026-09-24T20:00/, message: /妈|mom|mother/ } }],
    answer: [/提醒|remind/],
  },
  {
    id: "c-cutout",
    level: "L2",
    request: "把这张图的背景去掉",
    attachments: [{ name: "product.png", kind: "image" }],
    steps: [{ tool: "remove_background", args: { image: /product\.png/ } }],
    answer: [/img_/],
  },
  {
    id: "c-edit",
    level: "L2",
    request: "把这张照片里的天空改成晚霞",
    attachments: [{ name: "beach.jpg", kind: "image" }],
    steps: [{ tool: "edit_image", args: { image: /beach\.jpg/, instruction: /晚霞|sunset/ } }],
    answer: [/img_/],
  },
  {
    id: "c-fetch",
    level: "L2",
    request: "帮我看看这篇文章讲了什么 https://example-blog.dev/valkey",
    steps: [{ tool: "web_fetch", args: { url: /example-blog\.dev\/valkey/ } }],
    answer: [/license|许可|协议/],
  },
  {
    id: "c-currency",
    level: "L2",
    request: "100 美元现在合多少人民币？",
    steps: [{ tool: "currency_convert", args: { amount: /^100(\.0+)?$/, from: /usd/, to: /cny/ } }],
    answer: [/710/],
  },
  {
    id: "c-music",
    level: "L2",
    request: "给我的 vlog 做一段 30 秒轻快的背景音乐",
    steps: [{ tool: "generate_music", args: { prompt: /轻快|upbeat|cheerful|light|bright/, duration_seconds: /^30/ } }],
    answer: [/mus_/],
  },
  {
    id: "c-find-media",
    level: "L2",
    request: "我之前让你画过一张橘猫宇航员的图，帮我找出来",
    steps: [{ tool: "search_media", args: { query: /猫|cat/ } }],
    answer: [/img_prev_7/],
  },
  {
    id: "c-wallpaper",
    level: "L2",
    request: "画一张雪山日出的竖版手机壁纸",
    steps: [{ tool: "generate_image", args: { prompt: /雪|snow|mountain|山/, size: /1024x1792/ } }],
    answer: [/img_/],
  },
  // ---------------- L1：有诱人的工具，但不该调 ----------------
  { id: "l1-storm", level: "L1", request: "解释一下雷阵雨是怎么形成的", steps: [], answer: [/对流|convection|积雨云|cumulonimbus|上升|暖湿/] },
  { id: "l1-poem", level: "L1", request: "写一首关于橘猫的四行短诗", steps: [], answer: [/猫/] },
  { id: "l1-divide", level: "L1", request: "1200 除以 16 等于多少？", steps: [], answer: [/75/] },
  { id: "l1-translate", level: "L1", request: "Translate to Chinese: The meeting is postponed to Friday.", steps: [], answer: [/周五|星期五|礼拜五/] },
  // ---------------- 信息不足：该反问 ----------------
  { id: "ask-reminder", level: "L2", request: "帮我设个提醒", steps: [], answer: ASK, ask: true },
  { id: "ask-email", level: "L2", request: "给他发封邮件，说我今天会晚点到", steps: [], answer: ASK, ask: true },
  { id: "ask-edit", level: "L2", request: "把这张图改一下", attachments: [{ name: "photo.jpg", kind: "image" }], steps: [], answer: ASK, ask: true },
  { id: "ask-convert", level: "L2", request: "帮我把钱换算一下", steps: [], answer: ASK, ask: true },
  // ---------------- L3：三到四步，有依赖 ----------------
  {
    id: "m-minutes-email",
    level: "L3",
    request: "把这段会议录音整理成纪要，用邮件发给王总",
    attachments: [{ name: "weekly-sync.m4a", kind: "audio" }],
    anyOrder: true,
    steps: [
      { tool: "transcribe_audio", args: { file: /weekly-sync\.m4a/ } },
      { tool: "contacts_lookup", args: { name: /王|wang/ } },
      { tool: "send_email", args: { to: /wang\.lei@kunpeng\.example/, body: /10\s*月\s*15|10-15|october 15/ } },
    ],
    answer: [/发送|发给|sent|已/],
  },
  {
    id: "m-rain-umbrella",
    level: "L3",
    request: "查一下深圳明天的天气，如果会下雨就提醒我早上八点带伞",
    steps: [
      { tool: "get_weather", args: { city: /深圳|shenzhen/, date: /2026-09-25/ } },
      { tool: "set_reminder", args: { time: /2026-09-25T0?8:00/, message: /伞|umbrella/ } },
    ],
    answer: [/提醒|remind/],
  },
  {
    id: "m-cutout-video",
    level: "L3",
    request: "把这张产品图去掉背景，然后用它做一段 5 秒的旋转展示视频",
    attachments: [{ name: "product.png", kind: "image" }],
    steps: [
      { tool: "remove_background", args: { image: /product\.png/ } },
      { tool: "generate_video", args: { first_frame: /img_1/ } },
    ],
    answer: [/vid_/],
  },
  {
    id: "m-kb-usd-memory",
    level: "L3",
    request: "查一下公司一线城市住宿标准，换算成美元，把美元金额记到长期记忆里",
    steps: [
      { tool: "knowledge_search", args: { query: /差旅|住宿|travel|hotel/ } },
      { tool: "currency_convert", args: { amount: /^450(\.0+)?$/, from: /cny/, to: /usd/ } },
      { tool: "memory_save", args: { content: /63\.38/ } },
    ],
    answer: [/63\.38/],
  },
  {
    id: "m-invoice-usd-speech",
    level: "L3",
    request: "把这张发票的金额换算成美元，然后用女声播报出来",
    attachments: [{ name: "invoice-scan.pdf", kind: "document" }],
    steps: [
      { tool: "ocr_document", args: { file: /invoice-scan\.pdf/ } },
      { tool: "currency_convert", args: { amount: /^3,?860(\.0+)?$/, from: /cny/, to: /usd/ } },
      { tool: "generate_speech", args: { text: /543\.66/, voice: /female/ } },
    ],
    answer: [/aud_/],
  },
  {
    id: "m-note-web",
    level: "L3",
    request: "我笔记里向量数据库最后选了哪个？再上网查一下它最新的版本号",
    steps: [
      { tool: "note_search", args: { query: /向量|vector/ } },
      { tool: "web_search", args: { query: /qdrant/ } },
    ],
    answer: [/1\.15\.2/],
  },
  {
    id: "m-weather-email",
    level: "L3",
    request: "查一下王总的邮箱，把深圳明天的天气发给他",
    anyOrder: true,
    steps: [
      { tool: "contacts_lookup", args: { name: /王|wang/ } },
      { tool: "get_weather", args: { city: /深圳|shenzhen/, date: /2026-09-25/ } },
      { tool: "send_email", args: { to: /wang\.lei@kunpeng\.example/, body: /雷|雨|thunder|rain|storm/ } },
    ],
    answer: [/发送|发给|sent|已/],
  },
  {
    id: "m-release-poster",
    level: "L3",
    request: "搜一下 Bun 最新发布的版本，生成一张庆祝这个版本发布的横版海报",
    steps: [
      { tool: "web_search", args: { query: /bun/ } },
      { tool: "generate_image", args: { prompt: /1\.4\.7/, size: /1792x1024/ } },
    ],
    answer: [/img_/],
  },
  {
    id: "m-todo-file",
    level: "L3",
    request: "读一下 notes/todo.md，把还没完成的事项整理到新文件 notes/pending.md",
    steps: [
      { tool: "read_file", args: { path: /notes\/todo\.md/ } },
      { tool: "write_file", args: { path: /notes\/pending\.md/, content: /(报销[\s\S]*牙医|牙医[\s\S]*报销)/ } },
    ],
    answer: [/pending/],
  },
  {
    id: "m-lease-calendar",
    level: "L3",
    request: "看一下这份租约什么时候到期，在到期前 60 天建一个「发退租通知」的日历事件",
    attachments: [{ name: "lease.pdf", kind: "document" }],
    steps: [
      { tool: "ocr_document", args: { file: /lease\.pdf/ } },
      { tool: "create_calendar_event", args: { title: /退租|notice/, start: /2026-11-01/ } },
    ],
    answer: [/日历|calendar|创建|添加/],
  },
];

let counters: Record<string, number> = {};
const next = (prefix: string) => {
  counters[prefix] = (counters[prefix] ?? 0) + 1;
  return `${prefix}_${counters[prefix]}`;
};

export function resetStubs(): void {
  counters = {};
}

const RATES: Record<string, number> = { USD: 7.1, EUR: 7.8, CNY: 1 };

export function runStub(tool: string, args: Record<string, unknown>): string {
  const s = (k: string) => String(args[k] ?? "");
  const q = s("query").toLowerCase();
  switch (tool) {
    case "web_search":
      if (/bun/.test(q)) return "Bun v1.4.7 released on 2026-09-18: faster installs and a new SQL driver.";
      if (/qdrant/.test(q)) return "Qdrant v1.15.2 released on 2026-09-10 (latest stable).";
      if (/天气|weather/.test(q)) return "Weather sites: weather.com.cn, accuweather.com (use a weather service for forecasts).";
      return "No relevant results.";
    case "web_fetch":
      if (/example-blog\.dev\/valkey/.test(s("url"))) return "Why we moved from Redis to Valkey: after the Redis license change we switched to the BSD-licensed Valkey fork; migration took two days.";
      return "Error: page not found.";
    case "get_weather":
      if (/深圳|shenzhen/i.test(s("city"))) return `${s("date") || "2026-09-25"} Shenzhen: thunderstorms in the afternoon, 27–33°C, rain probability 80%.`;
      return `${s("date")} ${s("city")}: sunny, 18–26°C.`;
    case "currency_convert": {
      const amount = Number(String(args.amount ?? "").replace(/,/g, ""));
      const from = RATES[s("from").toUpperCase()];
      const to = RATES[s("to").toUpperCase()];
      if (!Number.isFinite(amount) || !from || !to) return "Error: unsupported amount or currency.";
      return `${amount} ${s("from").toUpperCase()} = ${((amount * from) / to).toFixed(2)} ${s("to").toUpperCase()}`;
    }
    case "knowledge_search":
      if (/差旅|住宿|报销|travel|hotel/.test(q)) return "《差旅管理办法 v3》第 4 条：一线城市住宿标准每晚不超过 450 元（CNY），其他城市 350 元。";
      return "No matching documents.";
    case "note_search":
      if (/向量|vector|数据库|database/.test(q)) return "笔记《向量数据库选型》(2026-08)：对比 Milvus / Qdrant / pgvector，结论：选 Qdrant。";
      return "No matching notes.";
    case "memory_search":
      if (/音乐|music|喜欢|like|偏好|prefer/.test(q)) return "Remembered: the user likes jazz music and cats; allergic to peanuts.";
      return "Nothing remembered about that.";
    case "memory_save":
      return "Saved to memory.";
    case "contacts_lookup":
      if (/王|wang/i.test(s("name"))) return "王总 (Wang Lei), wang.lei@kunpeng.example";
      return "No contact found.";
    case "send_email":
      return `Email sent to ${s("to")}.`;
    case "set_reminder":
      return `Reminder set for ${s("time")}.`;
    case "create_calendar_event":
      return `Calendar event "${s("title")}" created at ${s("start")}.`;
    case "generate_image":
      return `Image saved as ${next("img")} (${s("size") || "1024x1024"}).`;
    case "edit_image":
      return `Edited image saved as ${next("img")}.`;
    case "remove_background":
      return `Cut-out saved as ${next("img")} (transparent PNG).`;
    case "search_media":
      if (/猫|cat/.test(q)) return "Found: img_prev_7 — orange cat wearing an astronaut helmet (2026-09-20).";
      return "No matching media.";
    case "generate_video":
      return `Video saved as ${next("vid")}${s("first_frame") ? ` (animated from ${s("first_frame")})` : ""}.`;
    case "generate_music":
      return `Music saved as ${next("mus")} (${s("duration_seconds") || 30}s).`;
    case "generate_speech":
      return `Audio saved as ${next("aud")} (${s("voice") || "female"} voice).`;
    case "transcribe_audio":
      if (/weekly-sync/.test(s("file"))) return "[00:12] 张伟：测试还有两个阻塞问题。[03:40] 李娜：那就把上线从 10 月 8 日推到 10 月 15 日。[04:02] 张伟：同意，就定 10 月 15 日。";
      return "Error: file not found.";
    case "ocr_document":
      if (/invoice-scan/.test(s("file"))) return "增值税普通发票 No.20260917 金额合计 ¥3,860.00（人民币） 付款截止日：2026-10-10";
      if (/lease/.test(s("file"))) return "Lease term: 2026-01-01 to 2026-12-31. Section 12: either party may terminate with 60 days' written notice.";
      return "Error: file not found.";
    case "read_file":
      if (/notes\/todo\.md/.test(s("path"))) return "- [x] 提交周报\n- [ ] 报销差旅发票\n- [ ] 预约牙医\n- [x] 续费域名";
      return "Error: file not found.";
    case "write_file":
      return `Wrote ${s("content").length} chars to ${s("path")}.`;
    default:
      return `Error: unknown tool ${tool}.`;
  }
}
