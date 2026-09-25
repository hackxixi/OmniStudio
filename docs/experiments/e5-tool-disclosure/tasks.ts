/**
 * E5 任务：33 题，按「需要哪一组工具」标注（核心 / creation / perception / schedule，可多组），
 * 再加不该调工具与该反问两类。每一步列出可接受的实现：真实工具名与合并后的工具名都算对，
 * 只要参数对（`Check` 可以是正则，也可以是对整组参数的判断函数）。
 */

type Check = RegExp | ((args: Record<string, unknown>) => boolean);
/** 这一步可以用哪些工具完成，各自的参数要求。 */
export type Step = Record<string, Record<string, Check>>;

export type Task = {
  id: string;
  /** 需要的工具组（core 之外）；[] = 核心工具就够。 */
  groups: string[];
  kind: "core" | "group" | "multi" | "none" | "ask";
  request: string;
  attachments?: string[];
  steps: Step[];
  anyOrder?: boolean;
  answer: RegExp[];
};

export const NOW = "2026-09-24T15:00 (Thursday)";
export const CALENDAR =
  "Calendar: 2026-09-24 Thu (today), 2026-09-25 Fri (tomorrow), 2026-09-26 Sat, 2026-09-27 Sun, 2026-09-28 Mon, 2026-09-29 Tue, 2026-09-30 Wed, " +
  "2026-10-01 Thu, 2026-10-02 Fri, 2026-10-05 Mon, 2026-10-09 Fri, 2026-10-10 Sat.";

const kb = (q: RegExp): Step => ({ knowledge_search: { query: q }, recall: { query: q } });
const notes = (q: RegExp): Step => ({ note_search: { query: q }, recall: { query: q } });
const mem = (q: RegExp): Step => ({ memory_search: { query: q }, recall: { query: q } });
const save = (c: RegExp): Step => ({ memory_save: { content: c }, remember: { content: c } });
const text = (a: Record<string, unknown>) => JSON.stringify(a);
const portrait = (a: Record<string, unknown>) =>
  /9:16|2:3|3:4|portrait|竖/i.test(String(a.aspect_ratio ?? "")) || Number(a.height) > Number(a.width);
const landscape = (a: Record<string, unknown>) =>
  /16:9|3:2|4:3|landscape|横/i.test(String(a.aspect_ratio ?? "")) || Number(a.width) > Number(a.height);

export const TASKS: Task[] = [
  // ---------------- 核心工具就够 ----------------
  { id: "web-release", groups: [], kind: "core", request: "帮我查一下 Bun 最新发布的版本号", steps: [{ web_search: { query: /bun/i } }], answer: [/1\.4\.7/] },
  {
    id: "web-fetch",
    groups: [],
    kind: "core",
    request: "帮我看看这篇文章讲了什么 https://example-blog.dev/valkey",
    steps: [{ web_fetch: { url: /example-blog\.dev\/valkey/ } }],
    answer: [/license|许可|协议/i],
  },
  { id: "kb-policy", groups: [], kind: "core", request: "公司差旅制度里一线城市住宿标准是多少？", steps: [kb(/差旅|住宿|travel|hotel/i)], answer: [/450/] },
  { id: "note-lookup", groups: [], kind: "core", request: "我自己笔记里记的向量数据库选型结论是什么？", steps: [notes(/向量|vector|数据库/i)], answer: [/qdrant/i] },
  { id: "memory-recall", groups: [], kind: "core", request: "我之前跟你说过我喜欢什么音乐来着？", steps: [mem(/音乐|music|喜欢|like/i)], answer: [/jazz|爵士/i] },
  { id: "memory-save", groups: [], kind: "core", request: "记住：我对花生过敏，以后推荐菜谱的时候注意", steps: [save(/花生|peanut/i)], answer: [/./] },
  { id: "file-read", groups: [], kind: "core", request: "notes/todo.md 里还有哪些事没做完？", steps: [{ read_file: { path: /notes\/todo\.md/ } }], answer: [/报销/, /牙医/] },
  {
    id: "file-write",
    groups: [],
    kind: "core",
    request: "新建一个 notes/shopping.md，写上：牛奶、鸡蛋、面包",
    steps: [{ write_file: { path: /notes\/shopping\.md/, content: /牛奶[\s\S]*鸡蛋[\s\S]*面包/ } }],
    answer: [/./],
  },
  {
    id: "file-edit",
    groups: [],
    kind: "core",
    request: "把 docs/plan.md 里的上线日期从 10 月 8 日改成 10 月 15 日",
    steps: [
      {
        edit_file: { path: /docs\/plan\.md/, old_str: /10 ?月 ?8 ?日/, new_str: /10 ?月 ?15 ?日/ },
        write_file: { path: /docs\/plan\.md/, content: (a) => /10 ?月 ?15/.test(String(a.content)) && !/10 ?月 ?8 ?日/.test(String(a.content)) },
      },
    ],
    answer: [/./],
  },
  {
    id: "file-find",
    groups: [],
    kind: "core",
    request: "我工作区里有没有一个跟 budget 有关的表格文件？在哪？",
    steps: [{ find: { pattern: /budget/i }, glob: { pattern: /budget/i }, grep: { pattern: /budget/i }, list_dir: {}, bash: { command: /budget|find|ls/i } }],
    answer: [/budget-2026\.xlsx/],
  },
  // ---------------- 需要一组 ----------------
  {
    id: "img-wallpaper",
    groups: ["creation"],
    kind: "group",
    request: "画一张雪山日出的竖版手机壁纸",
    steps: [{ generate_image: { prompt: /雪|snow|mountain|山/i, _: portrait } }],
    answer: [/img_/],
  },
  { id: "tts-greeting", groups: ["creation"], kind: "group", request: "把「欢迎来到鲲鹏工作室」这句话读出来，做成语音", steps: [{ generate_speech: { text: /欢迎来到鲲鹏工作室/ } }], answer: [/aud_/] },
  { id: "video-waves", groups: ["creation"], kind: "group", request: "做一段 5 秒的海浪拍打礁石的视频", steps: [{ generate_video: { prompt: /浪|wave|sea|ocean|海/i } }], answer: [/vid_/] },
  { id: "media-find", groups: ["creation"], kind: "group", request: "我之前让你画过一张橘猫宇航员的图，帮我找出来", steps: [{ media_search: { query: /猫|cat/i } }], answer: [/img_prev_7/] },
  {
    id: "asr-meeting",
    groups: ["perception"],
    kind: "group",
    request: "这段会议录音里，最后定下来的上线日期是哪天？",
    attachments: ["weekly-sync.m4a"],
    steps: [{ transcribe_audio: { file: /weekly-sync\.m4a/ } }],
    answer: [/10\s*月\s*15|10-15|October 15/i],
  },
  {
    id: "ocr-invoice",
    groups: ["perception"],
    kind: "group",
    request: "这张发票的金额是多少？",
    attachments: ["invoice-scan.pdf"],
    steps: [{ ocr_document: { file: /invoice-scan\.pdf/ } }],
    answer: [/3,?860/],
  },
  {
    id: "reminder",
    groups: ["schedule"],
    kind: "group",
    request: "今晚八点提醒我给妈妈打电话",
    steps: [{ set_reminder: { time: /2026-09-24T20:00/, message: /妈|mom|mother/i } }],
    answer: [/提醒|remind/i],
  },
  {
    id: "calendar",
    groups: ["schedule"],
    kind: "group",
    request: "明天下午三点到四点和王总开会，帮我放到日历上",
    steps: [{ create_calendar_event: { title: /王|wang/i, start: /2026-09-25T15:00/ } }],
    answer: [/./],
  },
  {
    id: "email",
    groups: ["schedule"],
    kind: "group",
    request: "给王总发封邮件，说周五的会取消了",
    steps: [{ contacts_lookup: { name: /王|wang/i } }, { send_email: { to: /wang\.lei@kunpeng\.example/, body: /取消|cancel/i } }],
    answer: [/./],
  },
  // ---------------- 不该调工具 ----------------
  { id: "l1-storm", groups: [], kind: "none", request: "解释一下雷阵雨是怎么形成的", steps: [], answer: [/对流|convection|积雨云|上升|暖湿/] },
  { id: "l1-poem", groups: [], kind: "none", request: "写一首关于橘猫的四行短诗", steps: [], answer: [/猫/] },
  { id: "l1-divide", groups: [], kind: "none", request: "1200 除以 16 等于多少？", steps: [], answer: [/75/] },
  { id: "l1-translate", groups: [], kind: "none", request: "Translate to Chinese: The meeting is postponed to Friday.", steps: [], answer: [/周五|星期五/] },
  // ---------------- 该反问 ----------------
  { id: "ask-reminder", groups: [], kind: "ask", request: "帮我设个提醒", steps: [], answer: [] },
  { id: "ask-email", groups: [], kind: "ask", request: "给他发封邮件，说我今天会晚点到", steps: [], answer: [] },
  { id: "ask-image", groups: [], kind: "ask", request: "帮我画张图", steps: [], answer: [] },
  // ---------------- 多步（含跨组） ----------------
  {
    id: "m-note-web",
    groups: [],
    kind: "multi",
    request: "我笔记里向量数据库最后选了哪个？再上网查一下它最新的版本号",
    steps: [notes(/向量|vector|数据库/i), { web_search: { query: /qdrant/i } }],
    answer: [/1\.15\.2/],
  },
  {
    id: "m-kb-file",
    groups: [],
    kind: "multi",
    request: "查一下公司一线城市住宿标准，写进 notes/travel.md",
    steps: [kb(/差旅|住宿|travel|hotel/i), { write_file: { path: /notes\/travel\.md/, content: /450/ } }],
    answer: [/./],
  },
  {
    id: "m-release-poster",
    groups: ["creation"],
    kind: "multi",
    request: "搜一下 Bun 最新发布的版本，生成一张庆祝这个版本发布的横版海报",
    steps: [{ web_search: { query: /bun/i } }, { generate_image: { prompt: /1\.4\.7/, _: landscape } }],
    answer: [/img_/],
  },
  {
    id: "m-todo-speech",
    groups: ["creation"],
    kind: "multi",
    request: "把 notes/todo.md 里还没完成的事项用语音读给我听",
    steps: [{ read_file: { path: /notes\/todo\.md/ } }, { generate_speech: { text: (a) => /报销/.test(text(a)) && /牙医/.test(text(a)) && !/周报/.test(text(a)) } }],
    answer: [/aud_/],
  },
  {
    id: "m-export-cat",
    groups: ["creation"],
    kind: "multi",
    request: "把之前那张橘猫宇航员的图复制到工作区的 docs/images 目录",
    steps: [{ media_search: { query: /猫|cat/i } }, { media_export: { refs: /img_prev_7/, save_to: /docs\/images/ } }],
    answer: [/./],
  },
  {
    id: "m-minutes-email",
    groups: ["perception", "schedule"],
    kind: "multi",
    request: "把这段会议录音整理成纪要，用邮件发给王总",
    attachments: ["weekly-sync.m4a"],
    anyOrder: true,
    steps: [{ transcribe_audio: { file: /weekly-sync/ } }, { contacts_lookup: { name: /王|wang/i } }, { send_email: { to: /wang\.lei@kunpeng\.example/, body: /10\s*月\s*15|10-15|october 15/i } }],
    answer: [/./],
  },
  {
    id: "m-invoice-reminder",
    groups: ["perception", "schedule"],
    kind: "multi",
    request: "看一下这张发票的付款截止日，截止日前一天上午十点提醒我付款",
    attachments: ["invoice-scan.pdf"],
    steps: [{ ocr_document: { file: /invoice-scan/ } }, { set_reminder: { time: /2026-10-09T10:00/, message: /发票|付款|invoice|pay/i } }],
    answer: [/./],
  },
];

// ---------------------------------------------------------------------------
// 打分
// ---------------------------------------------------------------------------

export type Call = { tool: string; args: Record<string, unknown>; result: string };

function stepMatches(step: Step, call: Call): boolean {
  const checks = step[call.tool];
  if (!checks) return false;
  return Object.entries(checks).every(([k, c]) =>
    c instanceof RegExp ? c.test(typeof call.args[k] === "string" ? String(call.args[k]) : JSON.stringify(call.args[k] ?? "")) : c(call.args),
  );
}

/**
 * 通过 = 必需步骤按顺序（anyOrder 时不计顺序）全部出现且参数对；
 * 额外调用只允许只读工具、且至多 2 次；有副作用的多余调用直接判错；最终回复命中关键词。
 * 反问题：调了 ask_user（且没有别的副作用调用），或者不调工具、回复里带问号。
 */
export function score(task: Task, calls: Call[], answer: string, readOnly: Set<string>): { pass: boolean; why: string } {
  const real = calls.filter((c) => c.tool !== "load_tools");
  if (task.kind === "ask") {
    const sideEffects = real.filter((c) => !readOnly.has(c.tool) && c.tool !== "ask_user");
    if (sideEffects.length) return { pass: false, why: `side effect: ${sideEffects[0]!.tool}` };
    const asked = real.some((c) => c.tool === "ask_user") || (real.length === 0 && /[?？]/.test(answer));
    return asked ? { pass: true, why: "" } : { pass: false, why: "did not ask" };
  }
  const used = new Set<number>();
  let cursor = 0;
  for (const step of task.steps) {
    let found = -1;
    for (let i = task.anyOrder ? 0 : cursor; i < real.length; i++) {
      if (!used.has(i) && stepMatches(step, real[i]!)) {
        found = i;
        break;
      }
    }
    if (found < 0) return { pass: false, why: `missing step ${Object.keys(step).join("/")}` };
    used.add(found);
    cursor = found + 1;
  }
  const extras = real.filter((_, i) => !used.has(i));
  const bad = extras.find((c) => !readOnly.has(c.tool));
  if (bad) return { pass: false, why: `extra side effect: ${bad.tool}` };
  if (extras.length > 2) return { pass: false, why: `too many extra calls (${extras.length})` };
  const miss = task.answer.find((re) => !re.test(answer));
  if (miss) return { pass: false, why: `answer lacks ${miss}` };
  return { pass: true, why: "" };
}
