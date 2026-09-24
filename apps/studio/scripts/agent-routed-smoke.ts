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
 * 按脚本返回 choice 答案），然后对七种场景各跑一条真实的 `runAgentTurn`：
 *
 *   A. routed，JEV 选 none     → 核心工具 + load_tools，没有组工具；load_tools 之后下一次
 *                                请求出现 creation 组，且核心工具顺序不变
 *   B. routed，JEV 选 creation → 第一次请求就带 generate_image
 *   C. routed，JEV 500         → 加载全部组（generate_image 和 bash 都在），状态说明
 *                                写了「判定服务不可用」
 *   D. routed，占位符参数      → web_fetch(url="<网址>") 没真正执行，工具结果提示
 *                                ask_user，模型照做
 *   E. classic（默认）        → 工具列表与历史版本一致：全量，没有 recall / load_tools
 *   F. routed，JEV 选 none     → 本轮循环守卫（搜索打转，agent-loop-guard.ts）：模型连续
 *                                换关键词调 4 次 find（工作区为空 → 每次都没命中），
 *                                第 2 次（连续 2 次空）的结果末尾追加「不要再换关键词」
 *                                提示，第 4 次被拦下，结果含「不再继续搜索」
 *   G. routed，JEV 选 creation → 本轮循环守卫（生成失败）：generate_image 以 errorResult
 *                                失败（桩数据目录里没配生图后端）→ 结果追加「不要重试」；
 *                                第 2 次 generate_image 被拦下（「已经失败过」）；改调 bash
 *                                绕路也被拦下（dev 组未加载 → "Tool bash not found"，它既被
 *                                守卫 check 命中，也根本不在当前工具列表里）
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
type ModelScript = "A" | "B" | "C" | "D" | "E" | "F" | "G";

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
  /**
   * 场景 F：已经发出过几次 find（每次换 pattern —— 用 pattern 文本去重，
   * 而不是数请求里的工具结果：压缩会改写历史，那种计数会倒退，resilience 踩过这个坑）。
   */
  fQueries: [] as string[],
  /**
   * 场景 G：已经发出过几次 generate_image（带 arguments 原文去重）。
   * 用请求体里的 arguments 原文数，而不是 tool_call id（stub 里 id 恒等于工具名）。
   */
  gCalls: [] as string[],
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

const F_PATTERNS = ["deploy-notes", "部署记录", "rollout plan", "release checklist"];

/** 上一条 assistant 消息里发出的工具调用（wire 形状：name + arguments 原文，断言用）。 */
function lastAssistantCalls(
  messages: { role: string; content?: unknown; tool_calls?: unknown }[],
): { name: string; args: string }[] {
  const last = [...messages].reverse().find((m) => m.role === "assistant");
  if (!last || !Array.isArray(last.tool_calls)) return [];
  return last.tool_calls.map((c) => {
    const call = c as { function?: { name?: string; arguments?: string } };
    return { name: call.function?.name ?? "", args: call.function?.arguments ?? "" };
  });
}

/**
 * 模型剧本：
 * - A：第一次回应 `load_tools({group: creation})`，之后收尾（验证「下一轮工具列表带上新组」）；
 * - D：第一次回应占位符参数 `web_fetch({url: "<网址>"})`，被拦下后按提示 ask_user，拿到答案收尾；
 * - F：照着上一条工具结果继续搜索（E6 观察到的原地打转）→ 换下一个 pattern 调 find；
 *   结果含「不再继续搜索」（第 4 次被拦）→ 收尾；
 * - G：第一条工具结果是失败的生图（含守卫的「不要重试」提示）→ 照 E6 里观察到的打转，
 *   再调一次 generate_image（会被拦）；第 3 条是拦下的重复生图（含「已经失败过」）→ 改调 bash
 *   绕路（也会被拦）→ 收尾；
 * - B / C / E：第一次就收尾（它们的断言都在第一次请求上）。
 */
function modelResponse(request: StubRequest): string[] | Response {
  const { model, messages } = request;
  const toolResults = messages.filter((m) => m.role === "tool");
  const lastResult = toolResults[toolResults.length - 1]?.content;
  const lastResultText = typeof lastResult === "string" ? lastResult : JSON.stringify(lastResult ?? "");

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
  if (script === "F") {
    if (toolResults.length === 0) {
      const first = F_PATTERNS[0] ?? "deploy-notes";
      seen.fQueries.push(first);
      return toolCallChunks(model, "find", { pattern: first });
    }
    // 前两条工具结果（真实执行的 find）之后照着打转剧本再调 find；
    // 第三条工具结果（第 3 次 find）之后发起第 4 次，它会被守卫拦下
    // （连续 3 次空 ≥ 拦截阈值），结果里是「不再继续搜索」而不是执行输出。
    if (toolResults.length < 3) {
      const next = F_PATTERNS[seen.fQueries.length] ?? "deploy notes";
      seen.fQueries.push(next);
      return toolCallChunks(model, "find", { pattern: next });
    }
    if (lastResultText.includes("不再继续搜索")) {
      // 第 4 次被拦下了 → 按提示收尾。
      return textChunks(model, "F：工作区里没有部署记录，确认没找到，不再继续搜索。");
    }
    const next = F_PATTERNS[seen.fQueries.length] ?? "release checklist";
    seen.fQueries.push(next);
    return toolCallChunks(model, "find", { pattern: next });
  }
  if (script === "G") {
    const calls = lastAssistantCalls(messages);
    for (const call of calls) {
      if (call.name === "generate_image") seen.gCalls.push(`generate_image:${call.args}`);
    }
    if (toolResults.length === 0) {
      return toolCallChunks(model, "generate_image", { prompt: "日落的图" });
    }
    // 第一条工具结果 = 失败的 generate_image（守卫追加「不要重试」）→ 照 E6，再试一次。
    if (toolResults.length === 1) {
      return toolCallChunks(model, "generate_image", { prompt: "日落的图（再试一次）" });
    }
    // 第二条工具结果 = 被拦下的重复生图（「已经失败过」）→ 改调 bash 绕路。
    if (lastResultText.includes("已经失败过")) {
      return toolCallChunks(model, "bash", { command: "echo curl -s example-image-api" });
    }
    return textChunks(model, "G：生图后端还没配好，先不重试，直接告诉用户去「图像」页配置。");
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
  seen.fQueries.length = 0;
  seen.gCalls.length = 0;
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
// 场景 F：routed（JEV 选 none），本轮循环守卫 · 搜索打转（agent-loop-guard.ts）
//
// 工作区是空的临时目录，所以 find 的真实输出就是 "(no matches)"（命中守卫的空结果判据）。
// 剧本让桩模型按 E6 观察到的打转方式**连续 4 次**换关键词调 find，看守卫怎么逐步收手：
//   find ×1  真实执行（空结果，streak=1）
//   find ×2  真实执行（streak=2，结果末尾追加「不要再换关键词」提示）
//   find ×3  真实执行（streak=3，结果末尾再次追加提示）
//   find ×4  被 beforeToolCall 拦下（连续 3 次空 ≥ 拦截阈值），结果含「不再继续搜索」
// （guard.record 在 afterToolCall 里记账：提示追加在「连续第 N 次空」那一次的结果上，
// 拦下发生在第 3 次之后的下一次调用。）
// ---------------------------------------------------------------------------
{
  resetSeen();
  updateSettings({ ...baseSettings(), AGENT_TOOL_STRATEGY: "routed" });
  jevMode = "none";
  script = "F";

  const conversation = Chat.createConversation("routed F", "agent");
  Agent.setConversationWorkspace(conversation.id, workspace);
  const turn = await Agent.runAgentTurn({
    conversationId: conversation.id,
    content: "routed-F：帮我在项目里找一下上次部署记录的笔记。",
    mode: "agent",
    workspace,
  });

  const events = Agent.listAgentEvents(conversation.id);
  const findEnds = events
    .filter((event) => event.kind === "tool_end" && event.toolName === "find")
    .map((event) => event.output ?? "");
  const findArgs = events
    .filter((event) => event.kind === "tool_start" && event.toolName === "find")
    .map((event) => event.args ?? "");

  check("F：回合整体跑完（returned ok）", turn.ok, turn.error);
  check(
    "F：桩模型真的把 find 换着 pattern 连调 4 次（前 3 次真实执行，第 4 次被拦，之后按提示收尾）",
    findEnds.length === 4 && seen.fQueries.length === 4 && new Set(seen.fQueries).size === 4,
    `轨迹 ${findEnds.length} 次、桩发出 ${seen.fQueries.join(" / ")}`,
  );
  check(
    "F：4 次调用的 pattern 各不相同（打转特征：换关键词，而不是重复同一调用）",
    findArgs.length === 4 && new Set(findArgs).size === 4 && findArgs.every((args) => args.includes("pattern")),
    JSON.stringify(findArgs),
  );
  check(
    "F：第 1~3 次 find 都真实执行了，且都没有命中任何内容（空工作区）",
    findEnds.slice(0, 3).every((text) => text.includes("no matches")),
    JSON.stringify(findEnds.map((t) => t.slice(0, 80))),
  );
  check(
    "F：第 2 次 find（连续 2 次空）的结果末尾追加了「不要再换关键词」提示",
    (findEnds[1] ?? "").includes("不要再换关键词") && (findEnds[1] ?? "").includes("连续 2 次没有结果"),
    findEnds[1]?.slice(-200) ?? "(没有第 2 次 find)",
  );
  check(
    "F：第 3 次 find（连续 3 次空）的结果末尾同样追加了提示",
    (findEnds[2] ?? "").includes("不要再换关键词") && (findEnds[2] ?? "").includes("连续 3 次没有结果"),
    findEnds[2]?.slice(-200) ?? "(没有第 3 次 find)",
  );
  check(
    "F：第 4 次 find 没有真正执行，工具结果含「不再继续搜索」",
    (findEnds[3] ?? "").includes("不再继续搜索") && !(findEnds[3] ?? "").includes("no matches"),
    findEnds[3]?.slice(0, 200) ?? "(没有第 4 次 find)",
  );
  Agent.deleteConversationEvents(conversation.id);
  Chat.deleteConversation(conversation.id);
}

// ---------------------------------------------------------------------------
// 场景 G：routed（JEV 选 creation），本轮循环守卫 · 生成失败
//
// 桩数据目录里没有配生图后端（没有厂商 / 引擎 / 权重，headless 下配置弹窗直接
// 按取消收尾），generate_image 第一次调用就会以 errorResult 失败（details.error），
// 守卫把「不要重试」提示追加到这条工具结果末尾。桩模型照着 E6 的打转剧本：
//   generate_image ×1  真实执行 → errorResult，结果追加「不要重试」
//   generate_image ×2  被 beforeToolCall 拦下（结果含「已经失败过」，没有 details.error）
//   bash              绕路也被拦下。JEV 只选了 creation，bash（dev 组）不在工具列表里
//                     —— 真实小模型手里没有 bash，绕路只能是 load_tools（也是绕路工具）；
//                     这里直接调 bash 模拟「工具列表外绕路」，守卫的 check 同样命中它，
//                     内核回 "Tool bash not found"，没有真跑任何命令
// ---------------------------------------------------------------------------
{
  resetSeen();
  updateSettings({ ...baseSettings(), AGENT_TOOL_STRATEGY: "routed" });
  jevMode = "creation";
  script = "G";

  const conversation = Chat.createConversation("routed G", "agent");
  Agent.setConversationWorkspace(conversation.id, workspace);
  const turn = await Agent.runAgentTurn({
    conversationId: conversation.id,
    content: "routed-G：帮我画一张日落的图。",
    mode: "agent",
    workspace,
  });

  const events = Agent.listAgentEvents(conversation.id);
  const genEnds = events
    .filter((event) => event.kind === "tool_end" && event.toolName === "generate_image")
    .map((event) => event.output ?? "");
  const bashEnds = events
    .filter((event) => event.kind === "tool_end" && event.toolName === "bash")
    .map((event) => event.output ?? "");

  check("G：回合整体跑完（returned ok）", turn.ok, turn.error);
  check(
    "G：JEV 选 creation → 第一次请求就带 generate_image（生成组开局就在）",
    toolNames(0).includes("generate_image"),
    JSON.stringify(toolNames(0)),
  );
  check(
    "G：桩模型发出 generate_image ×2 + bash ×1（E6 打转剧本：重试 → 改道）",
    seen.gCalls.filter((c) => c.startsWith("generate_image")).length === 2 &&
      bashEnds.length === 1,
    JSON.stringify([...seen.gCalls, ...bashEnds.map((t) => `bash:${t.slice(0, 40)}`)]),
  );
  check(
    "G：第一次 generate_image 真的执行过（桩数据目录没配生图后端 → 失败），结果含「不要重试」提示",
    genEnds.length === 2 && (genEnds[0] ?? "").includes("不要重试"),
    genEnds[0]?.slice(0, 200) ?? "(没有第 1 次 generate_image)",
  );
  check(
    "G：第 2 次 generate_image 被拦下，没有真正执行（结果含「已经失败过」，没有失败原因）",
    (genEnds[1] ?? "").includes("已经失败过") && !(genEnds[1] ?? "").includes("生图失败"),
    genEnds[1]?.slice(0, 200) ?? "(没有第 2 次 generate_image)",
  );
  check(
    "G：生成失败后改调 bash 绕路也被拦下（dev 组未加载 → 工具不在列表，没真正执行）",
    bashEnds.length === 1 && (bashEnds[0] ?? "").includes("not found"),
    bashEnds[0]?.slice(0, 200) ?? "(没有 bash 调用)",
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
