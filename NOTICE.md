# 来源、版权与修改说明

RepForcing 的训练、蒸馏和推理实现建立在既有开源视频生成代码之上。项目名称变更不改变上游代码归属。原有版权注释和 Apache-2.0 许可证保留。

## 上游来源

- **Causal Forcing / Causal Forcing++**：因果视频训练、非对称 DMD 与推理实现基础。原项目列出的作者为 Hongzhou Zhu、Min Zhao、Guande He、Hang Su、Chongxuan Li 和 Jun Zhu。项目入口：[Causal Forcing](https://thu-ml.github.io/CausalForcing.github.io)。
- **Self Forcing**：上游实现所使用的自回归蒸馏训练组件。来源：[guandeh17/Self-Forcing](https://github.com/guandeh17/Self-Forcing)。
- **CausVid**：上游蒸馏代码的来源之一。来源：[tianweiy/CausVid](https://github.com/tianweiy/CausVid)。
- **Wan2.1**：`wan/` 内的模型、注意力、编码器、VAE 与采样工具。来源：[Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1)，Apache-2.0；具体版权信息见源文件及 `wan/README.md`。
- **FramePack**：`demo_utils/memory.py` 的内存管理实现。来源：[lllyasviel/FramePack](https://github.com/lllyasviel/FramePack/tree/main/demo_utils)，Apache-2.0，作者署名见文件头。
- **ASD**：保留的首块独立步数调度实现中的来源说明见 `model/base.py`。该调度不是当前四步正式配置的默认功能。

## 评测资源

VisionReward 问题和评分系数来自 [zai-org/VisionReward](https://github.com/zai-org/VisionReward)，具体提交及逐文件哈希记录于 `vbench/scripts/vision_reward/source.json`，其许可证随资源保留在同目录。`weights.json` 是评分系数资源，不是模型检查点。

VBench 是外部依赖。使用其代码、模型和数据时，需要遵守对应项目的许可证及资源条款。

## 此发布副本的修改

本副本保留两条 RepForcing 正式训练流程及对应评测，调整公开配置、路径和日志认证接口，去除私人历史运行依赖，补充数据工具的显式输入/输出接口、测试与使用文档。底层模型及既有算法代码保留其原始归属。

旧长视频扩展有独立的许可限制，不包含在此发布副本中。该排除不意味着其许可已变更。

根目录 `LICENSE` 适用于本仓库代码；基础模型、评测模型、数据和其他外部资源应分别检查其使用条款。
