# EAS

EAS（Evolving AI Systems）是一个用于长期学习、实验与求职展示的个人大一统 AI Systems Framework。

## 当前阶段

项目目前处于框架初始化阶段：已建立 Python 包边界、领域职责和开发环境入口，尚未实现任何 AI 功能。首个计划功能是推理方向的 **KV Cache**；它目前仅是规划，不代表已经可用。

## 结构

所有 Python 代码统一位于 `src/eas/`：

- `core/`：跨领域复用的底层核心能力。
- `inference/`：推理引擎、部署与服务化。
- `training/`：预训练、微调与算法；RL 位于 `training/algorithms/reinforcement_learning/`。
- `agents/`：智能体与工具协作边界。
- `embodied/`：具身感知与控制。
- `data/`：数据集与数据接入边界。
- `distributed/`：面向推理和训练的分布式基础设施。

仓库根目录的 `evaluation/`、`examples/`、`tests/` 和 `docs/` 分别用于评测、示例、验证和文档，不作为 Python 包。

## 开发命令

```bash
make help  # 查看命令和安全范围
make dev   # 使用 uv 创建或同步开发环境
```

`make dev` 会产生本地环境状态；项目初始化本身不会自动安装依赖或生成 lockfile。
