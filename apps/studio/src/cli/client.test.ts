import { describe, expect, test } from "bun:test";

import { launchCommand } from "./client";

describe("launchCommand（omi 拉起应用）", () => {
  test("macOS 用 open，没找到安装包时按应用名打开", () => {
    expect(launchCommand("/Applications/OmniStudio.app", "darwin")).toEqual({
      cmd: ["open", "/Applications/OmniStudio.app"],
      wait: true,
    });
    expect(launchCommand(null, "darwin")).toEqual({ cmd: ["open", "-a", "OmniStudio"], wait: true });
  });

  test("Windows 用 start（空标题占位，带空格的路径不会被当成窗口标题）", () => {
    expect(launchCommand("C:\\Program Files\\OmniStudio\\launcher.exe", "win32")).toEqual({
      cmd: ["cmd", "/c", "start", "", "C:\\Program Files\\OmniStudio\\launcher.exe"],
      wait: true,
    });
  });

  test("非 macOS 不知道安装位置时返回 null（要求 --app-path），不去跑不存在的 open", () => {
    expect(launchCommand(null, "win32")).toBeNull();
    expect(launchCommand(null, "linux")).toBeNull();
  });

  test("Linux 直接运行可执行文件、不等它退出", () => {
    expect(launchCommand("/opt/omni/launcher", "linux")).toEqual({ cmd: ["/opt/omni/launcher"], wait: false });
  });
});
