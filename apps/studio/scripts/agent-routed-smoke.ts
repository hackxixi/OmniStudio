/**
 * 精简路由工具策略（`AGENT_TOOL_STRATEGY=routed`）端到端冒烟。
 *
 * 背景见 docs/experiments/e5-tool-disclosure：本地小窗口下，把全部工具定义一次塞给
 * 模型会吃掉几千 token 的前缀。精简策略改成「常驻核心工具 + 按需加载工具组」：
 *
 * ```
 *   开局      只把 核心工具 + load_tools 发给模型
 *   每轮开始  调 JEV（/v1/systemone）判断这一轮要哪些组，选中的组追加到工具列表末尾；
 *             JEV 不可用 → 加载全部组（不能因为判定服务挂了就少给工具）
 *   模型侧    调 load_tools({group}) → 下一次请求的工具列表里出现该组（核心在前，组追加在后）
 *   参数兜底  参数是占位符（"<收件人>"）的调用被拦下，模型收到的工具结果提示先 ask_user
 * ```
 *
 * 这个脚本用一个内置桩服务同时扮演**推理服务**（`/v1/chat/completions`，流式，记录每次
 * 请求里的 `tools` 列表与完整 messages）与 **JEV 服务**（`/v1/systemone`，TypeSafe 协议，
 * 按脚本返回 choice 答案），然后对五种场景各跑一条真实的 `runAgentTurn`：
 *
 *   A. routed，JEV 选 none     → 核心工具 + load_tools，没有组工具；load_tools 之后下一次
 *                                请求出现 creation 组，且核心工具顺序不变
 *   B. routed，JEV 选 creation → 第一次请求就带 generate_image
 *   C. routed，JEV 500         → 加载全部组（generate_image 和 bash 都在），状态说明
 *                                写了「判定服务不可用」
 *   D. routed，占位符参数      → web_fetch(url="<网址>") 没真正执行，工具结果提示
 *                                ask_user，模型照做
 *   E. classic（默认）        → 工具列表与历史版本一致：全量，没有 recall / load_tools
 *
 * 跑法：`bun run scripts/agent-routed-smoke.ts`（已接进 test:smoke），确定性、不依赖
 * 任何真实推理 / 判定服务。
 */
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import path from "path";

import { startStubLlm, textChunks, toolCallChunks, type StubRequest } from "../src/bun/test-stub-llm";

const dataDir = mkdtempSync(path.join(tmpdir(), "omni-routed-"));
process.env.OMNI_DATA_DIR = dataDir;
const workspace = mkdtempSync(path.join(tmpdir(), "omni-routed-ws-"));

// ---------------------------------------------------------------------------
// 脚本化桩服务：一个端口同时扮演推理服务与 JEV 服务
// ---------------------------------------------------------------------------

/** 每种 JEV 剧本：none = 核心够用；creation = 生成类；error = 判定服务挂了（500）。 */
type JevMode = "none" | "creation" | "error";
/** 每个场景的模型剧本。 */
type ModelScript = "A" | "B" | "C" | "D" | "E";

const seen = {
  /** 每次 /v1/chat/completions 请求里的工具名列表（按请求顺序），断言用。 */
  toolLists: [] as string[][],
  /** 每次请求的完整 messages（wire 形状，场景 D 要检查其中那条工具结果）。 */
  messageDumps: [] as string[],
  /** 每次 /v1/systemone 请求的 state（验证选组问题真的带上了用户请求）。 */
  jevStates: [] as string[],
  /** 每次 /v1/systemone 的完整请求体（验证 question 形状）。 */
  jevRequests: [] as Record<string, unknown>[],
  /** /v1/systemone 被调用的总次数。 */
  jevCalls: 0,
};

let jevMode: JevMode = "none";
let script: ModelScript = "A";

/** JEV 的 choice 答案：选组问题只有一道（`group`），候选是 none / creation / dev / mcp。 */
function jevChoice(body: Record<string, unknown>): string | null {
  seen.jevCalls += 1;
  seen.jevRequests.push(body);
  seen.jevStates.push(typeof body.state === "string" ? body.state : JSON.stringify(body.state ?? ""));
  if (jevMode === "error") return null; // 调用方回 500（官方形状的错误 body）
  const top = jevMode === "none" ? "none" : "creation";
  // usage 用正数：成功的判定会记一行用量（channel=systemone），顺带把这条路走通。
  return JSON.stringify({
    model: typeof body.model === "string" ? body.model : "stub-jev",
    answers: {
      group: {
        type: "choice",
        choice: top,
        confidence: 0.7,
        probabilities: { none: jevMode === "none" ? 0.7 : 0.15, creation: jevMode === "none" ? 0.1 : 0.7 },
      },
    },
    usage: { input_tokens: 40, output_tokens: 0 },
  });
}

/**
 * 模型剧本：
 * - A：第一次回应 `load_tools({group: creation})`，之后收尾（验证「下一轮工具列表带上新组」）；
 * - D：第一次回应占位符参数 `web_fetch({url: "<网址>"})`，被拦下后按提示 ask_user，拿到答案收尾；
 * - B / C / E：第一次就收尾（它们的断言都在第一次请求上）。
 */
function modelResponse(request: StubRequest): string[] | Response {
  const { model, messages } = request;
  const toolResults = messages.filter((m) => m.role === "tool");

  if (script === "A") {
    if (toolResults.length === 0) {
      return toolCallChunks(model, "load_tools", { group: "creation" });
    }
    return textChunks(
      model,
      "A：已加载创作工具组（load_tools 返回 Loaded group creation），本任务到这儿就够了。",
    );
  }
  if (script === "D") {
    if (toolResults.length === 0) {
      return toolCallChunks(model, "web_fetch", { url: "<网址>" });
    }
    if (toolResults.length === 1) {
      // 第一条工具结果 = 被拦下的 web_fetch（提示先 ask_user）→ 照着做。
      return toolCallChunks(model, "ask_user", {
        questions: [{ header: "网址", question: "请给我要抓取的网址。" }],
      });
    }
    return textChunks(model, "D：拿到网址 example.com 了，任务完成。");
  }
  return textChunks(model, "B/C/E：好的，收到。");
}

const llm = startStubLlm({
  respond: async (request) => {
    seen.toolLists.push((request.tools ?? []).map((t) => t.function?.name ?? "").filter(Boolean));
    seen.messageDumps.push(JSON.stringify(request.messages));
    return modelResponse(request);
  },
});

// JEV 的 `/v1/systemone` 挂在另一个端口上（startStubLlm 的 handler 只认 /v1/models 与
// chat completions，没有 fetch 钩子）。场景 C 的 500 从这里回（官方形状的错误 body，
// detail.message 会被透传进轨迹说明）。
const jev = Bun.serve({
  port: 0,
  async fetch(req) {
    const url = new URL(req.url);
    if (url.pathname !== "/v1/systemone") return new Response("not here", { status: 404 });
    const body = (await req.json()) as Record<string, unknown>;
    const answer = jevChoice(body);
    if (answer === null) {
      return Response.json(
        { detail: { error_type: "api_error", message: "stub jev exploded" } },
        { status: 500 },
      );
    }
    return Response.json(JSON.parse(answer), { headers: { "x-typesafe-request-id": "req-routed-smoke" } });
  },
});

const base = `http://127.0.0.1:${llm.port}/v1`;
const jevBase = `http://127.0.0.1:${jev.port}`;

// ---------------------------------------------------------------------------
// 准备设置（照 resilience smoke 的方式：独立数据目录 + remote 模式指向桩）
// ---------------------------------------------------------------------------
const { updateSettings } = await import("../src/bun/db/settings");
const Chat = await import("../src/bun/chat");
const Agent = await import("../src/bun/agent");
const Interactions = await import("../src/bun/agent-interactions");

function baseSettings(): Record<string, string> {
  return {
    SETUP_COMPLETE: "1",
    SERVER_MODE: "remote",
    VLLM_API_BASE: base,
    VLLM_API_KEY: "EMPTY",
    // 云端模式窗口跟着模型 id 走（chat-context.ts）：`-8k` 后缀把窗口钉在 8192，
    // 与 resilience smoke 同一个口径。
    VLLM_MODEL_NAME: "stub-model-8k",
    CHAT_MODEL: "stub-model-8k",
    SERVER_CTX_SIZE: "8192",
    /** auto：这条冒烟不该被授权弹窗打断（授权链路由 live-check 专门验）。 */
    AGENT_APPROVAL_MODE: "auto",
    AGENT_MAX_STEPS: "4",
    MEMORY_ENABLED: "0",
    AGENT_RETRY_MAX: "2",
    // JEV（SystemOne）走云端后端，打到本脚本的桩服务（resolveBackend：cloud + 有 key 即可；
    // 本地服务地址留空，托管运行时不参与）。
    SYSTEMONE_BACKEND: "cloud",
    SYSTEMONE_CLOUD_BASE_URL: jevBase,
    SYSTEMONE_CLOUD_API_KEY: "stub-jev-key",
    SYSTEMONE_CLOUD_MODEL: "jev-latest",
    SYSTEMONE_LOCAL_BASE_URL: "",
  };
}

let failed = 0;
const check = (name: string, ok: boolean, detail?: string) => {
  console.log(`${ok ? "✓" : "✗"} ${name}${ok ? "" : ` — ${detail ?? ""}`}`);
  if (!ok) failed++;
};

function resetSeen() {
  seen.toolLists.length = 0;
  seen.messageDumps.length = 0;
  seen.jevStates.length = 0;
  seen.jevRequests.length = 0;
  seen.jevCalls = 0;
}

/** 轨迹里的选组状态说明（toolName=tool_routing）。 */
function routingEvent(conversationId: number) {
  return Agent.listAgentEvents(conversationId).find(
    (event) => event.kind === "status" && event.toolName === "tool_routing",
  );
}

// 与 agent-routed-tools.ts 的 ROUTED_CORE_TOOL_NAMES 一致（这里只列 agent mode 实际会
// 出现的那些：remember 的合成目标 memory_save 在 MEMORY_ENABLED=0 时不存在，不出现）。
const CORE = [
  "web_search",
  "web_fetch",
  "recall",
  "read_file",
  "write_file",
  "edit_file",
  "find",
  "note_read",
  "view_image",
  "ask_user",
  "read_skill",
];

const toolNames = (request: number) => seen.toolLists[request] ?? [];
function corePositions(names: string[]): number[] {
  return CORE.map((name) => names.indexOf(name)).filter((p) => p !== -1);
}
/** core 的相对顺序与 CORE 一致（各名字都得在，且先后不变）。 */
function coreOrderKept(names: string[]): boolean {
  const positions = corePositions(names);
  if (positions.length !== CORE.length) return false;
  return positions.every((p, i) => i === 0 || p > (positions[i - 1] ?? -1));
}
/** 核心工具里最后一个（read_skill）的下标；-1 表示核心不完整（前面的检查会抓）。 */
function lastCorePos(names: string[]): number {
  const last = CORE[CORE.length - 1];
  if (last === undefined) return -1;
  return names.indexOf(last);
}

console.log(`推理服务：${base}（内置桩）；JEV 桩：${jevBase}/v1\n`);

// ---------------------------------------------------------------------------
// 场景 A：routed，JEV 选 none → 开局核心 + load_tools；load_tools 之后下一轮带 creation
// ---------------------------------------------------------------------------
{
  resetSeen();
  updateSettings({ ...baseSettings(), AGENT_TOOL_STRATEGY: "routed" });
  jevMode = "none";
  script = "A";

  const conversation = Chat.createConversation("routed A", "agent");
  Agent.setConversationWorkspace(conversation.id, workspace);
  const turn = await Agent.runAgentTurn({
    conversationId: conversation.id,
    content: "routed-A：查一下最新的天气 API 文档（纯核心工具任务）。",
    mode: "agent",
    workspace,
  });

  const first = toolNames(0);
  const second = toolNames(1);
  const routing = routingEvent(conversation.id);

  check("A：回合整体跑完（returned ok）", turn.ok, turn.error);
  check("A：JEV 桩确实被调了一次（选组真的走了 /v1/systemone）", seen.jevCalls === 1, `JEV 调用 ${seen.jevCalls} 次`);
  check(
    "A：选组问题带上了用户请求（state 里有本轮提问）",
    (seen.jevStates[0] ?? "").includes("routed-A"),
    seen.jevStates[0] ?? "(无)",
  );
  check(
    "A：JEV 请求是 TypeSafe choice 问题（questions.group，选项含 none / creation / dev）",
    (() => {
      const q = (seen.jevRequests[0]?.questions ?? {}) as Record<string, { type?: string; criteria?: Record<string, unknown> }>;
      const criteria = Object.keys(q.group?.criteria ?? {});
      return q.group?.type === "choice" && ["none", "creation", "dev"].every((name) => criteria.includes(name));
    })(),
    JSON.stringify(seen.jevRequests[0]?.questions),
  );
  check(
    "A：第一次请求带 web_search / recall / find / load_tools",
    ["web_search", "recall", "find", "load_tools"].every((name) => first.includes(name)),
    JSON.stringify(first),
  );
  check(
    "A：第一次请求**没有** generate_image / bash / knowledge_search / glob / todo_write",
    ["generate_image", "bash", "knowledge_search", "glob", "todo_write"].every((name) => !first.includes(name)),
    JSON.stringify(first),
  );
  check(
    "A：load_tools 之后（第二次请求）出现 creation 组，且它排在全部核心工具之后（追加在末尾）",
    second.includes("generate_image") &&
      coreOrderKept(first) &&
      coreOrderKept(second) &&
      second.indexOf("generate_image") > lastCorePos(second),
    `第二次 [${second.join(",")}]`,
  );
  check(
    "A：creation 之后是未加载组的 load_tools（dev 还没有，排在 creation 后面）",
    second.indexOf("load_tools") === second.length - 1 &&
      second.indexOf("load_tools") > second.indexOf("generate_image"),
    `第二次 [${second.join(",")}]`,
  );
  check(
    "A：轨迹里有 tool_routing 状态说明（JEV 判定：核心够用）",
    (routing?.output ?? "").includes("工具组") && (routing?.output ?? "").includes("JEV"),
    routing?.output ?? "(没有 tool_routing 事件)",
  );
  Agent.deleteConversationEvents(conversation.id);
  Chat.deleteConversation(conversation.id);
}

// ---------------------------------------------------------------------------
// 场景 B：routed，JEV 选 creation → 第一次请求就带 generate_image
// ---------------------------------------------------------------------------
{
  resetSeen();
  updateSettings({ ...baseSettings(), AGENT_TOOL_STRATEGY: "routed" });
  jevMode = "creation";
  script = "B";

  const conversation = Chat.createConversation("routed B", "agent");
  Agent.setConversationWorkspace(conversation.id, workspace);
  const turn = await Agent.runAgentTurn({
    conversationId: conversation.id,
    content: "routed-B：画一张日落的图。",
    mode: "agent",
    workspace,
  });

  const first = toolNames(0);
  const routing = routingEvent(conversation.id);

  check("B：回合整体跑完（returned ok）", turn.ok, turn.error);
  check(
    "B：第一次请求就带 creation 组 generate_image，核心工具也都在",
    first.includes("generate_image") && ["web_search", "recall", "find"].every((name) => first.includes(name)),
    JSON.stringify(first),
  );
  check(
    "B：creation 组排在核心工具之后（JEV 选中的组追加，而不是插在最前面）",
    coreOrderKept(first) && first.indexOf("generate_image") > lastCorePos(first),
    JSON.stringify(first),
  );
  check(
    "B：load_tools 还在列表末尾（dev / mcp 尚未加载，随时可再加载）",
    first.includes("load_tools") && first.indexOf("load_tools") === first.length - 1,
    JSON.stringify(first),
  );
  check(
    "B：tool_routing 说明里写明了加载了 creation",
    (routing?.output ?? "").includes("creation"),
    routing?.output ?? "(没有 tool_routing 事件)",
  );
  Agent.deleteConversationEvents(conversation.id);
  Chat.deleteConversation(conversation.id);
}

// ---------------------------------------------------------------------------
// 场景 C：routed，JEV 500 → 退回全部组（generate_image 与 bash 都在），
//        状态说明写明「判定服务不可用」
// ---------------------------------------------------------------------------
{
  resetSeen();
  updateSettings({ ...baseSettings(), AGENT_TOOL_STRATEGY: "routed" });
  jevMode = "error";
  script = "C";

  const conversation = Chat.createConversation("routed C", "agent");
  Agent.setConversationWorkspace(conversation.id, workspace);
  const turn = await Agent.runAgentTurn({
    conversationId: conversation.id,
    content: "routed-C：JEV 挂了，应该退回全量工具组。",
    mode: "agent",
    workspace,
  });

  const first = toolNames(0);
  const routing = routingEvent(conversation.id);

  check("C：回合整体跑完（JEV 挂了不该把任务拖死）", turn.ok, turn.error);
  check(
    "C：JEV 500 → 第一次请求带全部组工具（generate_image 与 bash、apply_patch 都在）",
    first.includes("generate_image") && first.includes("bash") && first.includes("apply_patch"),
    JSON.stringify(first),
  );
  check(
    "C：全组都加载后 load_tools 不再出现",
    !first.includes("load_tools"),
    JSON.stringify(first),
  );
  check(
    "C：tool_routing 说明里写了「判定服务不可用」（并带上了上游错误原因）",
    (routing?.output ?? "").includes("判定服务不可用") && (routing?.output ?? "").includes("stub jev exploded"),
    routing?.output ?? "(没有 tool_routing 事件)",
  );
  Agent.deleteConversationEvents(conversation.id);
  Chat.deleteConversation(conversation.id);
}

// ---------------------------------------------------------------------------
// 场景 D：routed，模型填占位符参数（web_fetch url="<网址>"）→ 调用被拦下、
//        工具结果提示先 ask_user，模型照做（问完再收尾）
// ---------------------------------------------------------------------------
{
  resetSeen();
  updateSettings({ ...baseSettings(), AGENT_TOOL_STRATEGY: "routed" });
  jevMode = "none";
  script = "D";
  // 被拦下的调用会触发一次 ask_user，自动应答把链路走完（答案回到模型）。
  Interactions.onQuestionAsked((question) => {
    setTimeout(() => Interactions.respondQuestion(question.id, [["example.com"]]), 30);
  });

  const conversation = Chat.createConversation("routed D", "agent");
  Agent.setConversationWorkspace(conversation.id, workspace);
  const turn = await Agent.runAgentTurn({
    conversationId: conversation.id,
    content: "routed-D：帮我查一个网页（我没说地址，该问我就问我）。",
    mode: "agent",
    workspace,
  });

  const events = Agent.listAgentEvents(conversation.id);
  const second = seen.messageDumps[1] ?? "";
  const blocked = events.find((event) => event.kind === "tool_end" && event.toolName === "web_fetch");
  const asked = events.find((event) => event.kind === "tool_start" && event.toolName === "ask_user");

  check("D：回合整体跑完（returned ok）", turn.ok, turn.error);
  const blockedOutput = blocked === undefined ? "" : blocked.output ?? "";
  const fetchLooksExecuted = new RegExp("^\\s*Fetched|Page at|https?://", "m").test(blockedOutput);
  check(
    "D：占位符参数的 web_fetch **没有真正执行**（工具结果是拦截提示，没有抓到任何页面）",
    blocked !== undefined && !fetchLooksExecuted,
    blocked?.output?.slice(0, 200) ?? "(没有 web_fetch 的 tool_end)",
  );
  check(
    "D：拦下后的工具结果提示模型先 ask_user（并指出是占位符）",
    (blocked?.output ?? "").includes("ask_user") && (blocked?.output ?? "").includes("占位符"),
    blocked?.output?.slice(0, 200) ?? "(没有 web_fetch 的 tool_end)",
  );
  check(
    "D：模型接着调了 ask_user（提示是可执行的）",
    asked !== undefined,
    JSON.stringify(events.filter((e) => e.kind === "tool_start").map((e) => e.toolName)),
  );
  check(
    "D：第二次请求的上下文里那条工具结果含 ask_user 提示",
    second.includes("ask_user") && second.includes("占位符"),
    second.slice(0, 300),
  );
  Agent.deleteConversationEvents(conversation.id);
  Chat.deleteConversation(conversation.id);
}

// ---------------------------------------------------------------------------
// 场景 E：classic（默认）→ 工具列表与历史版本一致：全量，没有 recall / load_tools
// ---------------------------------------------------------------------------
{
  resetSeen();
  updateSettings({ ...baseSettings(), AGENT_TOOL_STRATEGY: "classic" });
  jevMode = "none";
  script = "E";

  const conversation = Chat.createConversation("classic E", "agent");
  Agent.setConversationWorkspace(conversation.id, workspace);
  const turn = await Agent.runAgentTurn({
    conversationId: conversation.id,
    content: "classic-E：经典策略，全量工具一次给。",
    mode: "agent",
    workspace,
  });

  const first = toolNames(0);
  const routing = routingEvent(conversation.id);

  check("E：回合整体跑完（returned ok）", turn.ok, turn.error);
  check(
    "E：第一次请求是全量工具（knowledge_search、todo_write、bash 都在）",
    first.includes("knowledge_search") && first.includes("todo_write") && first.includes("bash"),
    JSON.stringify(first),
  );
  check(
    "E：经典策略没有合成 / 元工具（recall / load_tools / find 都不出现）",
    !first.includes("recall") && !first.includes("load_tools") && !first.includes("find"),
    JSON.stringify(first),
  );
  check(
    "E：经典策略不触发选组（JEV 一次都没调、没有 tool_routing 说明）",
    seen.jevCalls === 0 && routing === undefined,
    `JEV 调用 ${seen.jevCalls} 次`,
  );
  Agent.deleteConversationEvents(conversation.id);
  Chat.deleteConversation(conversation.id);
}

// ---------------------------------------------------------------------------
// 清理
// ---------------------------------------------------------------------------
Agent.stopAllAgentRuns();
llm.stop();
jev.stop();
rmSync(dataDir, { recursive: true, force: true });
rmSync(workspace, { recursive: true, force: true });

if (failed > 0) {
  console.error(`routed smoke: ${failed} 项失败`);
  process.exit(1);
}
console.log("routed smoke 全部通过");
