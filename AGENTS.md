# Coding style guidelines

Before writing, modifying, or reviewing code, read and follow
[.claude/skills/code-review/coding-style.md](.claude/skills/code-review/coding-style.md).

## 开发与部署约定

- 服务器已 clone 本项目，后续复用已有仓库。
- 在本机进行代码开发与修改；提交并推送到远程仓库后，服务器拉取对应分支的最新代码。
- 在服务器上部署、运行并查看实际效果，以服务器验证结果确认部署效果。
- 默认遵循上述分工，不直接在服务器修改项目源码。
- 当前部署服务器为 `43.128.249.5`，SSH 用户 `root`、端口 `22`；工作台使用 `http://43.128.249.5:8081`。认证信息不写入仓库。

## 外部应用目录约定

- 工作台等通过接口使用 sglang-omni 的外部应用统一放到 `projects/<项目名>/`。
- 各应用独立管理依赖、启动入口、配置和运行结果，不混入模型推理代码。
- 随本仓库同步到服务器后，按对应应用自己的说明部署。
