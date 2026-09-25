/**
 * 「追加参数」文本 ⇄ argv 的互转（四个引擎共用）。
 *
 * 以前一律 `split(/\s+/)`：`--chat-template-kwargs '{"enable_thinking": false}'` 会被切成
 * 三段、引号原样进 argv，引擎直接报参数错误。这里按 POSIX shell 的引号规则切分，
 * 但**只做切分**——不做变量展开 / 通配 / 命令替换（argv 直接交给 spawn，没有 shell 参与，
 * 用户粘进来的 `$(...)` 就是字面量，不能被执行）。
 */

/**
 * 按 shell 引号规则切分：
 *  - 空白分隔；单引号内全部字面量；双引号内只认 `\"` `\\` `\$` `` \` `` 转义；
 *  - 引号外的反斜杠转义下一个字符（行尾续行 `\<换行>` 吞掉）；
 *  - `''` / `""` 产生一个空参数（与 shell 一致）；
 *  - 引号没闭合：宽松处理，余下内容按字面量收进当前参数（不抛错 —— 启动不该因为这个挂掉，
 *    引擎自己会对怪参数报错，用户在日志里能看到原样）。
 */
export function splitShellArgs(input: string): string[] {
  const out: string[] = [];
  let cur = "";
  // 当前参数是否「已开始」：区分 `''`（一个空参数）与纯空白（没有参数）。
  let started = false;
  let i = 0;
  const s = input ?? "";
  while (i < s.length) {
    const ch = s[i]!;
    if (ch === "'") {
      started = true;
      const end = s.indexOf("'", i + 1);
      if (end === -1) {
        cur += s.slice(i + 1);
        i = s.length;
      } else {
        cur += s.slice(i + 1, end);
        i = end + 1;
      }
      continue;
    }
    if (ch === '"') {
      started = true;
      i++;
      while (i < s.length && s[i] !== '"') {
        if (s[i] === "\\" && i + 1 < s.length && '"\\$`'.includes(s[i + 1]!)) {
          cur += s[i + 1];
          i += 2;
        } else if (s[i] === "\\" && s[i + 1] === "\n") {
          i += 2;
        } else {
          cur += s[i];
          i++;
        }
      }
      i++; // 跳过收尾的引号（没闭合时越界，循环自然结束）
      continue;
    }
    if (ch === "\\") {
      if (i + 1 < s.length) {
        if (s[i + 1] !== "\n") {
          cur += s[i + 1];
          started = true;
        }
        i += 2;
      } else {
        i++;
      }
      continue;
    }
    if (/\s/.test(ch)) {
      if (started) out.push(cur);
      cur = "";
      started = false;
      i++;
      continue;
    }
    cur += ch;
    started = true;
    i++;
  }
  if (started) out.push(cur);
  return out;
}

/** 不需要加引号的字符集（与 Python shlex.quote 同一口径）。 */
const SAFE_ARG = /^[\w@%+=:,./-]+$/;

/** 单个参数转成可粘进终端的形式：安全字符原样，其余用单引号包起来。 */
export function shellQuote(arg: string): string {
  if (arg === "") return "''";
  if (SAFE_ARG.test(arg)) return arg;
  return `'${arg.replace(/'/g, `'\\''`)}'`;
}

/**
 * argv → 可复制的命令行。`splitShellArgs(shellJoin(argv))` 与 argv 逐项相等，
 * 所以「复制的命令」与实际发出去的 argv 是同一份（路径里带空格也不会被拆开）。
 */
export function shellJoin(argv: string[]): string {
  return argv.map(shellQuote).join(" ");
}
