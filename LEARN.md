# SGLang Omni 学习计划

当前主线是 **Qwen/Qwen3-TTS-12Hz-1.7B-Base**，用于参考音频声音克隆(Voice Cloning)，仍属于自回归离散 Codec 生成中的帧内码本预测路线。

按 [Base 核心链路学习计划](/home/tiger/workspage/sglang-omnivoice/sglang-omni-learn/docs/developer_reference/qwen3_tts_base_walkthrough_zh.md) 跟一条完整请求：参考音频编码与说话人向量 → 提示构造 → Talker 首码本 → 帧内其余码本 → 下一帧反馈 → 音频解码与返回。

模型计算配套阅读：[Qwen3-TTS 小张量示例](/home/tiger/workspage/sglang-omnivoice/sglang-omni-learn/docs/developer_reference/qwen3_tts_tensor_examples_zh.md)，用数值例子理解各模块。

当前重点：接着模型层计算，跟清两个连续音频帧的生成与反馈。启动和调度已讨论，按需回看；进度区分“已讨论”与“已运行验证”，不按方法数量计分。
