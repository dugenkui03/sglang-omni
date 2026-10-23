# Qwen3-TTS Base 核心链路学习计划

目标：跟清 `Qwen/Qwen3-TTS-12Hz-1.7B-Base` 的一条声音克隆(Voice Cloning)请求，理解参考声音如何成为条件，以及文本如何逐帧生成音频。
范围：单请求、单 GPU、参考音频与原文、上下文学习(In-Context Learning, ICL)、完整 WAV 返回；先记录一次预填充(Prefill)、两帧生成和结束返回。

## 1. 主线与已有进度

```text
POST /v1/audio/speech
    → Client.speech() → Coordinator.submit_request()
    ├─ ① preprocessing：参考音频编码、说话人向量、文本 → 提示向量(Prompt Embedding)
    ├─ ② tts_engine / OmniScheduler：请求入队 → 组批 → ModelRunner.execute()
    │    → Talker：预测当前帧第 0 个码本(Codebook)
    │    → Code Predictor：补齐同一帧的码本 1～15
    │    → 本帧反馈 + 后续文本向量 → 下一帧，直到结束
    └─ ③ vocoder：完整 audio_codes → 波形(Waveform)
         → Coordinator.handle_completion() → Client.speech() 编码 WAV → HTTP 返回
```

这是跨组件的数据流；队列和异步任务连接各段，并非一个同步方法调用栈。

| 段落 | 当前进度 | 本轮读到什么程度 |
|---|---|---|
| 启动与对象关系 | 已讨论，待串联 | 知道三个 Stage、模型实例和 Runner 的归属即可 |
| 请求与提示构造 | Base 分支待精读 | 找到参考编码、说话人向量和文本如何进入模型输入 |
| 入队、执行与 Talker | 入队/执行/RMSNorm 已讨论 | 补齐一层计算、最终 norm、codec_head |
| 帧内预测与下一帧反馈 | 重点待跟读 | 跟清两个连续帧的输入、输出和反馈 |
| 音频解码与返回 | 已概述，待跟读 | 找到 codes → 波形 → WAV 的三个边界 |

“已讨论”不代表已运行验证；本计划以能串起数据流为准，不再按方法数量或其他模型阅读比例计进度。

## 2. 五段阅读顺序

**① 启动定位：复习对象关系即可。**

源码：[Pipeline 配置](/home/tiger/workspage/sglang-omnivoice/sglang-omni-learn/sglang_omni/models/qwen3_tts/config.py)、[Engine Builder](/home/tiger/workspage/sglang-omnivoice/sglang-omni-learn/sglang_omni/models/qwen3_tts/engine_builder.py)、[SGLModelRunner](/home/tiger/workspage/sglang-omnivoice/sglang-omni-learn/sglang_omni/model_runner/sglang_model_runner.py)。
确认 `model_arch_override="Qwen3TTSTalker"` 与 `register_omni_model()`：检查点(Checkpoint)声明 `Qwen3TTSForConditionalGeneration`，本链路实际执行 `Qwen3TTSTalker`。
对象关系：`OmniScheduler.model_runner` → `Qwen3TTSModelRunner.tp_worker` → `ModelWorker.model_runner` → `SGLModelRunner.model` → `Qwen3TTSTalker`。
自检：Stage 是流水线阶段，Talker/Code Predictor 是模型组件；请求顺序是 `preprocessing → tts_engine → vocoder`，不同于配置中的构造顺序。

**② 请求与提示构造：跟到 waiting_queue。**

源码：[HTTP 入口](/home/tiger/workspage/sglang-omnivoice/sglang-omni-learn/sglang_omni/serve/openai_api.py)、[Client](/home/tiger/workspage/sglang-omnivoice/sglang-omni-learn/sglang_omni/client/client.py)、[请求适配](/home/tiger/workspage/sglang-omnivoice/sglang-omni-learn/sglang_omni/models/qwen3_tts/request_builders.py)、[OmniScheduler](/home/tiger/workspage/sglang-omnivoice/sglang-omni-learn/sglang_omni/scheduling/omni_scheduler.py)。
沿 `create_speech()` → `Client.speech()/generate()` → `Coordinator.submit_request()` → `Stage.execute()`，观察 `StagePayload` 如何进入各阶段。
预处理重点：`preprocess_qwen3_tts_payload()` → `prepare_qwen3_tts_request()` → `prepare_qwen3_tts_base_request()`；跟 `Qwen3TTSAdhocReferenceHook.encode_one()` 的参考编码和说话人特征，再看 `build_voice_clone_inputs()`。
引擎重点：`build_sglang_qwen3_tts_request()` 创建 `Qwen3TTSSGLangRequestData` 和 SGLang `Req`，再经 `enqueue_built_request()` 加入 `waiting_queue`。
自检：`ref_audio` 怎样得到 `ref_code` 和说话人嵌入(Speaker Embedding)？`ref_text` 怎样进入提示？找到 `prompt_input_embeds`、`pending_text_queue`、`tts_pad_embed`；参考结果命中缓存时可跳过重复编码。

**③ Talker 主干：先跟一次 Prefill，再看一层计算。**

源码：[Runner 基类](/home/tiger/workspage/sglang-omnivoice/sglang-omni-learn/sglang_omni/model_runner/base.py)、[TTS Runner](/home/tiger/workspage/sglang-omnivoice/sglang-omni-learn/sglang_omni/models/qwen3_tts/model_runner.py)、[模型定义](/home/tiger/workspage/sglang-omnivoice/sglang-omni-learn/sglang_omni/models/qwen3_tts/sglang_model.py)。
当前默认走 `event_loop_normal()`：`get_next_batch_to_run()` → `run_batch()/_run_batch()` → `ModelRunner.execute()`；`before_prefill()` 提供提示向量，随后进入模型前向(Forward)。

```text
Qwen3TTSTalker.forward()
    → Qwen3TTSTalkerTextModel.forward()
    ├─ Qwen3TTSTalkerDecoderLayer.forward() × 28
    │    → input_layernorm → self_attn → post_attention_layernorm → mlp
    └─ model.norm：最终残差相加与 RMSNorm
         → Prefill 选每个请求最后一个位置 → codec_head：输出 logits
         → Qwen3TTSModelRunner.sample_next_token_ids()：采样第 0 码本 token
```

采样(Sampling)由 Runner 调用，位于 `Talker.forward()` 返回之后；上图包含调用后的数据处理顺序。
主干计算配套读[小张量示例](/home/tiger/workspage/sglang-omnivoice/sglang-omni-learn/docs/developer_reference/qwen3_tts_tensor_examples_zh.md)①～⑦：嵌入、投影、残差、注意力(Attention)、前馈网络(MLP)、输出头。
自检：`self.norm(hidden_states, residual)` 做残差相加、归一化与逐维缩放；线性投影在 `codec_head` 等模块中完成。

**④ 一帧到下一帧：本轮最重要的一段。**

继续在 TTS Runner 与模型定义中读：

```text
post_prefill() / post_decode()
    → collect_codes()：取第 0 码本 token 和 Talker hidden state
    → Qwen3TTSTalker.code_predictor_forward()
    │    → code_predictor_forward_incremental()：普通实现，读懂帧内预测
    │    → 得到 16 个 codes，并把各码本对应的嵌入向量相加
    → post_process_outputs()：保存本帧 codes 与 pending_feedback_queue
    → 下一轮 before_decode() → write_feedback_buffers()
         → feedback + 下一份文本向量（耗尽后用 tts_pad）→ 下一次 Talker.forward()
```

运行时码本预测可能走 CUDA 图(CUDA Graph)分支；先用普通实现理解同一算法，不展开图捕获。配套读小张量示例⑧～⑨。
两种自回归(Autoregression)：Talker 沿时间轴逐帧生成；5 层 Code Predictor 在当前帧内依次预测其余 15 个码本。
自检：取两个连续帧，记录下表；只打印形状与少量元素，张量(Tensor)不完整展开。

| 观察对象 | 单请求下应看见什么 |
|---|---|
| `mr_output.next_token_ids` | `[1]`，本轮第 0 码本 token；不是完整音频编码 |
| `req_data.output_codes` | 每个有效帧追加 `[16]`；结束 token(EOS)不作为音频帧保存 |
| `pending_feedback_queue` / `decode_feedback_embedding` | 队列仅存码本向量之和；反馈表存加上文本条件后的结果，维度为 2048 |
| `audio_codes` | ICL 下形状为 `[T_ref+T_gen,16]`，结果适配器拼接参考帧和生成帧 |

`decode_feedback_embedding` 是 GPU 显存(GPU memory)中的反馈暂存表；解码(Decode)时用当前批次行号读取，该行号会写入 `input_ids`，不是固定请求槽位或词表 token。

**⑤ 结束与音频返回：追到可播放的 WAV。**

源码：[结果适配](/home/tiger/workspage/sglang-omnivoice/sglang-omni-learn/sglang_omni/models/qwen3_tts/request_builders.py)、[Stage 路由](/home/tiger/workspage/sglang-omnivoice/sglang-omni-learn/sglang_omni/pipeline/stage/runtime.py)、[声码器(Vocoder)](/home/tiger/workspage/sglang-omnivoice/sglang-omni-learn/sglang_omni/models/qwen3_tts/streaming_vocoder.py)、[Coordinator](/home/tiger/workspage/sglang-omnivoice/sglang-omni-learn/sglang_omni/pipeline/coordinator.py)。
跟 `process_batch_result()` 的结束判断 → `stream_output()` 的完成请求分支 → `apply_sglang_qwen3_tts_result()` → `Stage.route_result()` → vocoder。
再跟 `vocode_payloads()` → `tokenizer.decode()` → `store_vocoder_result()` → `Coordinator.handle_completion()` → `Client.speech()` 编码音频并返回。
自检：`make_batch_result()` 只包装调度结果；ICL 参考编码与生成编码拼接后交给 vocoder，`store_vocoder_result()` 按参考帧占比裁去参考部分，再编码 WAV。音频解码器内部网络留到主线跑通后。

Base 与此前 CustomVoice 共用 Talker、码本预测和声码器主线；新增重点是参考音频条件。小张量文档用于主干计算，其内置音色示例仍以 CustomVoice 为背景。

## 3. 用同一条请求串联验证

在依赖与模型已就绪的环境中启动，另开终端发请求。脚本使用 `qwen3_tts_1_7b.yaml`，并授权服务读取本项目 `test_data/`：

```bash
bash scripts/serve_qwen3_tts.sh
bash scripts/test_qwen3_tts_base.sh
```

请求重点：`task_type="Base"`、`ref_audio`、`ref_text`、`x_vector_only_mode=false`；`voice="default"`，音色来自参考音频。结果保存到 `test_data/outputs/base_clone.wav`。
当前 `Client.speech()` 会强制内部请求 `stream=False`；码本增量发送同时要求 `stream` 与 `stream_codec_output` 为真，vocoder 也按 `stream` 分支。本例显式关闭两个开关，便于阅读。
完成标准：用同一 `request_id` 串起参考编码与说话人向量、提示向量、首帧、下一帧、`audio_codes`、波形和 WAV；能说明每一步由谁生成、交给谁。
建议下一步：先接着当前模型层阅读完成③，再重点跟④的两帧反馈闭环，最后补齐②与⑤；启动内容只按需回看。
后续选读：流式输出、取消与异常清理、缓存管理、并行与性能、其他 TTS 模型；这些不作为本轮前置条件。
