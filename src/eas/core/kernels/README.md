# Kernels

定位：承载 EAS 中可复用、经过验证的底层计算原语，为不同系统领域提供稳定的计算基础。

首条计划主线是 attention 与 KV Cache 相关原语；当前仅初始化模块边界，尚未实现任何算子。

## 边界

- 具体推理生命周期、请求调度和缓存使用策略属于 `eas.inference`。
- 模型与训练算法定义属于 `eas.training.algorithms`。
- 性能与正确性 benchmark 属于仓库根目录的 `evaluation/`。
- 本模块不提前定义通用算子接口、抽象基类或后端注册机制。
