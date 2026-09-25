import { describe, expect, test } from "bun:test";
import { join } from "path";
import { SOCKET_PATH_MAX_BYTES, controlSocketPathFor } from "./control-socket";

const byteLength = (s: string) => Buffer.byteLength(s, "utf8");

/** 拼一条比平台上限还长的数据目录（在 /var/... 前缀上不断加深）。 */
function longDir(limit: number, extra = 20): string {
  let dir = join("/var", "state", "omni");
  while (byteLength(join(dir, "omni-control.sock")) <= limit + extra) {
    dir = join(dir, "very-deep-directory-level");
  }
  return dir;
}

describe("SOCKET_PATH_MAX_BYTES", () => {
  test("darwin 103，其余 107（用恰好卡边界的路径交叉验证）", () => {
    // 全路径 = "/p/"（3）+ 文件名 N + "/" + "omni-control.sock"（17）= 20 + N。
    // N=84 → 104：darwin 超（103）→ 走 tmpdir；linux 不超（≤107）→ 留在数据目录里。
    const case104 = join("/p", `n${"x".repeat(84 - 2)}y`);
    expect(byteLength(join(case104, "omni-control.sock"))).toBe(105);
    expect(controlSocketPathFor(case104, { platform: "darwin", tmpDir: "/t" }).startsWith("/t/")).toBe(true);
    expect(controlSocketPathFor(case104, { platform: "linux", tmpDir: "/t" })).toBe(join(case104, "omni-control.sock"));
    // N=88 → 108：两个平台都超长。
    const case108 = join("/p", `n${"x".repeat(88 - 2)}y`);
    expect(byteLength(join(case108, "omni-control.sock"))).toBe(109);
    expect(controlSocketPathFor(case108, { platform: "darwin", tmpDir: "/t" }).startsWith("/t/")).toBe(true);
    expect(controlSocketPathFor(case108, { platform: "linux", tmpDir: "/t" }).startsWith("/t/")).toBe(true);
  });
});

describe("controlSocketPathFor", () => {
  test("短路径原样返回 <dataDir>/omni-control.sock", () => {
    const p = controlSocketPathFor("/Users/u/data", { platform: "darwin" });
    expect(p).toBe("/Users/u/data/omni-control.sock");
  });

  test("超长路径落到 tmpdir 且长度在上限内", () => {
    const dir = longDir(103);
    const p = controlSocketPathFor(dir, { platform: "darwin", tmpDir: "/var/folders/tmptest" });
    expect(p.startsWith("/var/folders/tmptest/omni-")).toBe(true);
    expect(p.endsWith(".sock")).toBe(true);
    expect(byteLength(p)).toBeLessThanOrEqual(103);
    // 同一平台 linux 上限更宽，同样处理
    const p2 = controlSocketPathFor(dir, { platform: "linux", tmpDir: "/tmp" });
    expect(p2.startsWith("/tmp/omni-")).toBe(true);
    expect(byteLength(p2)).toBeLessThanOrEqual(107);
  });

  test("同一目录两次结果相同、不同目录不同", () => {
    const a = longDir(103);
    const b = join(a, "other");
    const opts = { platform: "darwin" as const, tmpDir: "/t" };
    expect(controlSocketPathFor(a, opts)).toBe(controlSocketPathFor(a, opts));
    expect(controlSocketPathFor(a, opts)).not.toBe(controlSocketPathFor(b, opts));
    // 短目录不受影响，仍用数据目录
    expect(controlSocketPathFor("/d/x", opts)).toBe("/d/x/omni-control.sock");
  });

  test("override 优先且原样返回", () => {
    const override = "/custom/place/sock";
    expect(controlSocketPathFor("/d", { override })).toBe(override);
    expect(controlSocketPathFor(longDir(103), { override, platform: "darwin", tmpDir: "/t" })).toBe(override);
  });

  test("tmpdir 本身超长时退到 /tmp", () => {
    const dir = longDir(103);
    // 构造一个把 /tmp/omni-<16>.sock 都挤超 103 的 tmpdir：97 字节的 tmp 前缀 + 24 字节的文件名 = 121 > 103
    const hugeTmp = `/v/${"x".repeat(96)}`;
    const p = controlSocketPathFor(dir, { platform: "darwin", tmpDir: hugeTmp });
    expect(p).toBe(`/tmp/omni-${controlSocketPathFor(dir, { platform: "darwin", tmpDir: "/t" }).split("omni-")[1]}`);
    expect(byteLength(p)).toBeLessThanOrEqual(103);
  });

  test("SOCKET_PATH_MAX_BYTES 导出值：darwin 103 / 其余 107", () => {
    if (process.platform === "darwin") {
      expect(SOCKET_PATH_MAX_BYTES).toBe(103);
    } else {
      expect(SOCKET_PATH_MAX_BYTES).toBe(107);
    }
  });
});

test("不传 platform 时按当前平台的上限算（真实调用方都不传）", () => {
  // 长度恰好卡在 macOS 上限与 Linux 上限之间的路径
  const name = "/omni-control.sock";
  const dir = `/${"d".repeat(105 - name.length - 1)}`;
  expect(Buffer.byteLength(`${dir}${name}`)).toBe(105);
  const result = controlSocketPathFor(dir, { tmpDir: "/tmp" });
  if (process.platform === "darwin") expect(result.startsWith("/tmp/omni-")).toBe(true);
  else expect(result).toBe(`${dir}${name}`);
});
