# 使用说明 / User guide

## 一键启动与配置

先启动模型服务，再双击解压目录的 `Start-Tina.cmd`。首次运行会询问接口、模型、工作区、密钥文件路径、上下文容量、输出上限和界面语言。终端界面将显示实际配置的模型与地址。以后双击即可使用已保存的配置。

`Configure-Tina.cmd` 用于重新配置。默认保存到 `%LOCALAPPDATA%\TinaCLI\config.json`，不保存密钥值。若设置 `TINA_HOME`，配置和默认历史改用该目录；`--config PATH` 或 `TINA_CONFIG` 可以另选配置文件。

通用配置示例（路径和模型 ID 请替换为自己的）：

```powershell
.\bin\tina.exe configure --base-url http://127.0.0.1:8000/v1 --model my-model --workspace C:\work\my-project --context-positions 32768 --max-tokens 4096 --api-key-env TINA_API_KEY
.\bin\tina.exe chat
```

已有 Augur 可以使用 `tina configure --augur --workspace C:\work\my-project`。这会读取当前用户的 Augur 配置并引用其密钥文件，不会启动或修改 Augur。若模型 ID 与默认 `quasar-27b` 不同，同时传入 `--model`。

配置优先级：命令行覆盖对应设置，`TINA_BASE_URL` / `TINA_MODEL` 覆盖配置文件，剩余使用保存值或默认值。凭据优先使用配置的环境变量，未提供时再读配置的密钥文件。API 地址不得包含用户名、密码、查询参数或 fragment。请勿把密钥粘贴到聊天里。

命令行 `--workspace` 优先于保存的工作区，二者都没有时使用当前目录。保持 `bin\tina.exe` 与 `bin\cq-tool-host.exe` 在一起。Git 应在 PATH 中；执行 Cargo 工具还需要工作区相应的 Rust 工具链。

## 交互与工具

直接输入任务；以 `/` 开头的输入可查看命令提示。`/help` 列出可用命令。语言可用 `/language zh` 或 `/language en` 切换。

| 操作 | 命令或按键 |
| --- | --- |
| 查看模型、会话、上下文和上一轮统计 | `/status`、`/context` |
| 新会话、列出会话、恢复 | `/new`、`/sessions`、`/resume [ID]` |
| 替换重试当前任务 | `/retry` |
| 主动交接旧轮次 | `/compact` |
| 调整后续请求思考强度 | `/thinking off\|low\|medium\|high\|xhigh` |
| 设置或清除会话目标 | `/goal set TEXT`、`/goal clear` |
| 记住事实、查看记忆 | `/remember TEXT`、`/memory` |
| 管理项目记忆 | `/project-memory` |
| 任务清单 | `/todo` |
| Git 差异、只读审查 | `/diff`、`/review [focus]` |
| 导出会话 | `/export markdown PATH` 或 `/export jsonl PATH` |
| 查看工具、重启工具进程 | `/tools`、`/agent-restart` |
| 取消当前请求/工具 | 生成时按 Ctrl-C |
| 退出 | `/exit` |

文件写入、补丁、创建目录和重命名需要输入 `APPLY`；删除需要 `DELETE`；允许范围内的 Cargo 执行需要 `EXECUTE`。确认绑定具体操作。拒绝后，模型收到真实的拒绝结果。重定向输入和 `--request` 单次模式不能批准修改和执行。

`--read-only` 只公开只读工具。`--chat-only` 不发送工具定义，但仍使用本地归档；纯聊天的最小示例：

```powershell
.\bin\tina.exe chat --chat-only --thinking off --request 'Say hello in Chinese.'
```

工具进程不是 OS 沙箱。批准构建/测试可能执行工作区代码，具有当前用户权限。详见 [SECURITY](../SECURITY.md)。

## 保存、恢复和上下文交接

默认历史放在 `%LOCALAPPDATA%\TinaCLI\conversations\<scope-hash>`；scope 由工作区、接口与模型共同决定。可用 `--archive PATH` 显式指定目录。`tina chat --resume` 恢复该 scope 下已有会话；不加 `--resume` 会新建会话。

归档保留完整原始消息、工具批准、工具结果与交接来源。压缩的是发往模型的活跃上下文，不会删除原历史。当前交接使用有界的抽取式摘要，并保留最近一轮完整对话及当前输入；不是额外调用模型生成的语义摘要。工具事务未关闭时禁止交接。

默认上下文为 262,144，输出预留 4,096，输入预算 258,048，95% 阈值 245,145。请按实际服务端容量设置。首次计量为保守字节估算；服务端 usage 可在前缀匹配时校准。单次输入或最近保留轮次太大时，Tina 会保留历史并提示超限。可拆分输入、降低输出预留，或配置服务端实际支持的更大容量；不会无提示截断。

取消生成会关闭 HTTP 请求，部分流式文字不作为完整答案提交。已执行的工具可能已经产生效果，其最终回执仍保留。下一轮前请以回执和文件现状为准，不要假设取消自动撤销了修改。远端是否及时回收计算取决于模型服务端。

## 统计与排错

界面显示的速度为 HTTP 单轮 token/耗时；不是 GPU native TPS。未提供 usage 时显示 unavailable。`last-live-metrics.json` 保存上一轮统计，和会话一样属于本地私有数据。Tina 不统计远端 DFlash 接受率和 KV 缓存命中。

| 现象 | 处理 |
| --- | --- |
| 连接失败 | 检查模型服务已启动，base URL 含 `/v1`，端口正确 |
| 401/403 | 检查密钥环境变量/文件和服务端是否一致；不要公开密钥 |
| 不支持 reasoning 参数 | 使用 `/thinking off`，或选兼容的服务端配置 |
| 模型不支持工具 | 使用 `--chat-only`，或换支持工具的模型/接口 |
| `length` / 输出被截断 | 当前回答不提交为完成；按实际容量调整 `--max-tokens` 后重试 |
| 上下文超限 | `/compact`，拆分过大的当前输入，核对服务端容量与输出预留 |
| 找不到工具进程 | 确认两个 EXE 同目录，或指定 `--tool-host PATH` |
| 服务端返回 native generation failed | 查模型引擎日志；Tina 不将失败的流伪装成成功结果 |

Windows 候选版的固定响应协议和终端操作测试已经通过。真实模型验收状态请看 [verification/summary.json](verification/summary.json)，不要把测试夹具的速度当作模型成绩。
