# Tina CLI

Rust 终端 Agent：流式聊天、工作区工具、持久化会话与上下文交接。

[English](README.md) · [使用说明](docs/USAGE.md) · [上下文规则](docs/CONTEXT.md) · [构建说明](docs/BUILDING.md)

**v0.1.0-alpha.2：**终端 UI 已接入 HTTP 模型客户端与真实工具进程，Windows 协议、终端操作和真实 DeepSeek 接入验收均已通过。这是产品联调验收，不是模型能力基准或本地 GPU 性能测试。

## Windows 使用

1. 先启动模型服务。Tina 连接已有的 Chat Completions 兼容 `/v1` 接口，不包含模型权重或推理引擎。
2. 解压 Windows 包，双击 `Start-Tina.cmd`。首次配置填写接口地址、模型 ID、工作区、密钥文件路径及上下文/输出预算。
3. 输入任务。文件读取和 Git 查询可直接进行；写入、删除、执行命令会显示对应确认。

以后修改配置可双击 `Configure-Tina.cmd`。已有 Augur 配置可以导入地址、上下文容量和密钥文件引用：

```powershell
.\bin\tina.exe configure --augur --workspace C:\work\my-project
.\bin\tina.exe chat
```

这条命令不会启动 Augur，请先启动引擎。其他兼容服务：

```powershell
.\bin\tina.exe configure --base-url http://127.0.0.1:8000/v1 --model my-model --workspace C:\work\my-project --api-key-env TINA_API_KEY
.\bin\tina.exe chat
```

需要鉴权时，在自己的终端设置对应环境变量；也可配置 `--api-key-file C:\private\model-key.txt`。配置只保存引用，不写入密钥值。`tina.exe` 与 `cq-tool-host.exe` 应保持在同一个 `bin` 目录。Git 工具与项目记忆依赖 Git；运行 Tina 本身不需要 Python 或 GPU。

## DeepSeek 快速接入

环境里已设置 `DEEPSEEK_API_KEY` 后：

```powershell
.\bin\tina.exe configure --base-url https://api.deepseek.com/v1 --model deepseek-flash --api-key-env DEEPSEEK_API_KEY --thinking low --workspace C:\work\my-project
.\bin\tina.exe chat
```

官方 DeepSeek 接口使用原生思考参数：`off` 关闭，`low` 对应 low，`medium`/`high` 对应 high，Tina 的 `xhigh` 对应 DeepSeek max。工具续写所需的 `reasoning_content` 单独归档，跨进程恢复和交接保留轮次时不会丢失。

## 已接入功能

- 流式思考和回答、分片工具参数解析、真实工具结果回传。
- 文件读取、搜索、Git 检查、带批准的文件修改和受限 Cargo 命令。
- 写入输入 `APPLY`，删除输入 `DELETE`，执行输入 `EXECUTE`；非交互输入不能批准这些操作。
- 生成或工具执行时按 Ctrl-C 取消。完成的工具结果保留，不把中断回答标记为完成。
- `/new`、`/sessions`、`/resume`、`/retry`、`/compact`、`/context`、`/status`。
- `/thinking off|low|medium|high|xhigh`、目标、待办、记忆、导出、差异检查和只读审查。
- 历史不可变归档；上下文达到交接阈值时压缩旧轮次，保留最近完整轮次。

这里的 UI 是终端界面。MCP、子 Agent 和任意 shell 执行尚未实现。服务端需要支持使用到的工具和思考字段；纯聊天可使用 `--chat-only --thinking off`。

## 上下文与速度口径

默认 262,144 个位置，预留 4,096 输出 token，输入预算为 **258,048**，95% 交接阈值为 **245,145**。应按实际服务端配置容量，Tina 的配置不能扩大模型 KV 窗口。

首次请求用保守的 UTF-8/JSON 字节估算，界面标记 **estimated**。收到服务端 usage 后，仅在旧消息前缀和工具定义完全不变时使用其输入 token 数校准后续估算。这不是本地 tokenizer。单次输入或必须保留的最近轮次过大时，仍会明确拒绝，不会悄悄截断。

界面速度为 **HTTP 单轮吞吐**；服务端未返回 usage 时显示不可用。Tina 不把它标记为 native prefill/decode TPS，也不推测远端 KV 或 DFlash 接受率。

## 验证与开源

Windows 已通过 **187 项 Rust 单元/进程测试、9 项 HTTP 协议测试**、上下文边界、格式和 Clippy 检查。终端测试验证了真实 `APPLY` 写文件、Ctrl-C 关闭连接及随后新建会话。另有 **7 项脚本与 4 项终端真实 DeepSeek 用例**通过，覆盖中英文回答、算术代码执行、思考模式工具续写、恢复、压缩、批准写入、切换思考模式，以及取消后继续请求。固定响应协议测试与真实 API 验收分别记录。见[验证记录](docs/verification/summary.json)。

构建需要 Rust 1.96.0 和 MSVC C++ 工具。完整验证还需要 Python 3.11+ 标准库。支持 Windows ZIP、TAR.GZ、源码压缩包和 SHA256 校验。

MIT OR Apache-2.0。保留 Crystal Quant 上游归属、源文件校验和 142 项 Windows 依赖的许可记录。详见 [NOTICE](NOTICE)、[第三方许可](THIRD_PARTY_NOTICES.md)、[安全边界](SECURITY.md)和[变更记录](CHANGELOG.md)。
