/**
 * 「让 Agent 解决」填进输入框的那一段话。
 *
 * 单独成一个纯模块：格式本身就是这个按钮的全部价值（带齐了 Agent 第一轮就能动手，
 * 带漏了它只能先反问），要能脱离界面直接测；而按钮组件会拉进 RPC 桥，测试里一加载
 * 就把同进程其它用例的 mock 顶掉。
 */

/** 带过去的日志行数上限：够看清卡在哪一步，又不至于把输入框撑成一篇日志。 */
export const LOG_TAIL_LINES = 40;

/**
 * 引擎日志是给终端看的，带 ANSI 颜色码（`\x1b[34m…\x1b[0m`）。原样塞进输入框就是
 * 满屏 `⌧[34m`：人看着乱，模型也白白多吃一截 token —— 真机上第一次点就是这样。
 * 顺带去掉回车与其它控制字符，只留正文。
 */
// eslint-disable-next-line no-control-regex
const ANSI = /\x1b\[[0-9;?]*[ -/]*[@-~]/g;
// eslint-disable-next-line no-control-regex
const CONTROL = /[\x00-\x08\x0b-\x1f\x7f]/g;

export function cleanLogLine(line: string): string {
  return line.replace(ANSI, "").replace(CONTROL, "").trimEnd();
}

export function buildDiagnosisPrompt(p: {
  intro: string;
  error: string;
  context?: string[];
  logs?: string[];
}): string {
  const tail = (p.logs ?? [])
    .map(cleanLogLine)
    .filter((line) => line.trim())
    .slice(-LOG_TAIL_LINES);
  return [
    p.intro,
    "",
    `报错：${p.error}`,
    ...(p.context ?? []),
    tail.length > 0 ? `\n日志（末尾 ${tail.length} 行）：\n${tail.join("\n")}` : "",
    // 行动建议：把"找东西"的步数省掉。真机现场里最常见的浪费是 Agent 先花十几步
    // glob/grep 去找日志文件和数据库在哪（还撞上工作区权限被拒、只能改用 bash 绕）。
    // 提示词里把路径直接给它（见 agent-diagnose-button 的环境行），再明确"先读、别搜"。
    "先读上面给出的日志与报错，工作区外的文件用 bash（cat / grep / sed）读取；判断清楚后直接给出结论与修复步骤，不要大范围搜索源码。",
  ]
    .filter((line) => line !== "")
    .join("\n");
}
