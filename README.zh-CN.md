# Tina CLI

一个用 Rust 编写的终端 Agent Harness，提供上下文预算、会话归档和独立工具进程。

[English](README.md) · [上下文契约](docs/CONTEXT.md) · [构建说明](docs/BUILDING.md) · [安全边界](SECURITY.md)

## 当前可运行的功能

本次独立发布提供 **CPU 诊断、提示词导出、上下文容量检查和脚本化 Agent loop 回放**。回放使用固定模型输出，文件读取、工具进程、Git 查询、回执和归档均真实执行，无需 GPU 或模型权重。

原有交互终端与模型集成代码保留在源码中。**当前独立二进制尚未接入模型 API，也不提供实时模型聊天。** MCP 和子代理调度尚未实现。

## Windows 使用

从 Releases 下载 Windows 包，解压后安装 Git（如尚未安装），双击 `Run-Demo.cmd`。演示会新建临时 Git 工作区，读取示例文件、查询 Git 状态，并保留工具回执与会话归档。

```powershell
.\bin\tina.exe --help
.\bin\tina.exe doctor --workspace .
.\bin\tina.exe prompt --workspace . --request '检查当前项目并说明下一步'
.\bin\tina.exe context-check --input-tokens 258048
```

`context-check` 检查调用方传入的 token 数，不负责分词。超限会输出结构化拒绝原因并返回非零退出码。回放默认拒绝写入、删除与执行命令，场景文件不能提供审批。

## 上下文修复

- 带入 `cb2bcda3` 的容量检查，区分输入为空、超物理容量、输出预留耗尽和超可用输入上限。
- 适配 `3a218bdd`：工作区、Shell 等执行环境只注入 Agent 提示词，不额外改变普通聊天消息。
- 将回放提示词过时的 1 MiB 限制同步为 32 MiB 文档保护上限；字节限制与 token 预算分别处理。
- 小上下文配置也完整保留请求的输出预算，不能靠静默缩小输出预留让请求通过。

262,144 物理位置默认预留 4,096 输出位置，可用输入为 **258,048 token**，95% 交接阈值为 **245,145 token**。提高输出预留会降低可用输入。这些边界已有 CPU 测试；本包没有据此声称完成 256K 模型推理验收。

## 构建

安装 Rust 1.96.0、Git 和 Windows MSVC C++ 构建工具后运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Verify.ps1
cargo build --workspace --release --locked
```

验证脚本依次构建工具宿主、运行全特性测试、检查格式和 Clippy。源码结构、许可与功能边界见英文首页；本地验收平台为 Windows x86_64。

代码采用 MIT OR Apache-2.0。保留 Crystal Quant 来源与第三方许可，不包含模型、API 密钥或私人原画文件。
