# Third-party notices

VideoTrace 的实现是独立编写的；以下项目只作为架构阅读和设计参考，没有整段复制代码。

- `datawhalechina/hello-agents`，reviewed commit `45dd84e626a91997294ac8d4d44f18b29a411c6e`，CC BY-NC-SA 4.0。参考 Agent 分层、上下文工程、memory 和 trace。
- `jjyaoao/helloagents`，reviewed commit `5432566d01ea1c2095c4a717fe2a010aa1c3b0bd`，CC BY-NC-SA 4.0。参考 schema tool response、retry、circuit breaker；未直接移植其代码，保留其非商业/ShareAlike 限制。
- `nageoffer/ragent`，reviewed commit `5aabe1eaeedae70df353010fc73b315a8a1eef89`，Apache-2.0。参考多路检索、RRF、候选截断和 memory 分层；VideoTrace 用 Python 原生实现时间戳视频检索。

如未来复制或实质改编第三方代码，必须在对应文件和发行包中补充许可证与 attribution。

