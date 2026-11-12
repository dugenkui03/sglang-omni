# OmniVoice 待办事项


## 1. 推理优化：逐项讨论、逐项接入


### 1.1 核心代码文件：加速实现在哪里

| 标记 | 仓库及核心代码文件 | 关键函数／类与职责 |
|---|---|---|
| O1 | OmniVoice：[omnivoice_flashinfer.py][O1] | `apply_flashinfer()` 总入口；`_generate_iterative_packed()` 打包和批量采样；`_forward_logits()` 仅目标位置输出头；`_patch_attention_forward()`、`_patch_mlp()`、`_patch_rmsnorm()` 局部替换；`_get_or_capture_graph()`、`_get_or_capture_bucket_graph()` 图捕获。 |
| O2 | OmniVoice：[cli/infer_batch.py][O2] | `cluster_samples_by_batch_size()`、`cluster_samples_by_duration()` 按长度／时长组批；`process_init()` 加载后按开关启用 FlashInfer。 |
| O3 | OmniVoice：[models/omnivoice.py][O3] | `create_voice_clone_prompt()`、`VoiceClonePrompt` 复用参考音频编码；`_generate_iterative()`、`_predict_tokens_with_scoring()` 是普通路径和采样对照。 |
| V1 | vLLM-Omni：[omnivoice_generator.py][V1] | `OmniVoiceAttention`、`OmniVoiceMLP` 合并投影；`triton_rms_norm()`、`triton_fused_add_rms_norm()`、`triton_swiglu()` 融合算子；`_OmniVoiceCUDAGraphForward` 分桶捕获／重放；`forward()` 固定文本向量复用、目标 logits 转 FP32；`_unmask_one_request()` 单次 log-softmax CFG。 |
| V2 | vLLM-Omni：[fused_qkv_rope.py][V2] | `fused_qkv_norm_rope()`、`_qkv_norm_rope_kernel()` 将 Q/K 归一化、RoPE、K/V 广播和布局转换融合为一次 Triton 内核调用。 |
| V3 | vLLM-Omni：[pipeline_omnivoice.py][V3] | `OmniVoicePipeline.denoise_step()` 对当前活跃请求打包执行单轮生成；`step_scheduler()` 推进请求状态；参考音频编码复用由 `_speaker_cache` 接入。 |

[O1]: https://github.com/k2-fsa/OmniVoice/blob/28bc0889d92110491d726a9c79f26a895db5a074/omnivoice/models/omnivoice_flashinfer.py
[O2]: https://github.com/k2-fsa/OmniVoice/blob/28bc0889d92110491d726a9c79f26a895db5a074/omnivoice/cli/infer_batch.py
[O3]: https://github.com/k2-fsa/OmniVoice/blob/28bc0889d92110491d726a9c79f26a895db5a074/omnivoice/models/omnivoice.py
[V1]: https://github.com/vllm-project/vllm-omni/blob/f7d012bf3c2f63468a76b86acef5e5033872f7f5/vllm_omni/model_executor/models/omnivoice/omnivoice_generator.py
[V2]: https://github.com/vllm-project/vllm-omni/blob/f7d012bf3c2f63468a76b86acef5e5033872f7f5/vllm_omni/model_executor/models/omnivoice/fused_qkv_rope.py
[V3]: https://github.com/vllm-project/vllm-omni/blob/f7d012bf3c2f63468a76b86acef5e5033872f7f5/vllm_omni/diffusion/models/omnivoice/pipeline_omnivoice.py

### 1.2 有源码实现依据的优化清单

#### 01. 仅目标位置输出头（Target-only Projection）

- [x] 参考 O1 `_forward_logits()`，先提取两个 CFG 分支的目标隐藏状态，再计算 `audio_heads` 和转为 FP32，减少无用投影及 logits 占用的 GPU 显存。V1 `forward()` 只实现了提取目标 logits 后再转 FP32，`_get_logits()` 仍投影全部位置；两者不能混称。

实现：`generation.py` 每段构造一次目标索引，`model.py` 在输出头前取行；返回 `[num_audio_codebook, total_target_frames * 2, audio_vocab_size]`，条件目标在前、无条件目标在后。主干和采样规则不变，不保留完整 logits 返回入口。

2026-10-31 验证：77 项单测／回归通过；官方权重的 4 项 CPU FP32 对齐通过，生成码本一致、波形最大误差均为 0。GPU 时延及峰值 GPU 显存收益尚未实测；缺少 `pre_commit`，全量钩子未运行。

#### 02. 合并 QKV 投影（Fused QKV Projection）

- [ ] 代码已实现，待 GPU 验收：参考 O1 `_patch_attention_forward()`、V1 `OmniVoiceAttention`，在权重加载后通过 `fuse_qkv_projections()` 就地融合各层 QKV，复用原注意力对象；一次投影后按 Q/K/V 各自宽度拆分，保留原有归一化、RoPE、SDPA 和输出投影。本次仅保留推理融合，不提供融合后检查点的保存／重载格式转换。

2026-11-02：初版曾通过 17 项微型模型 CPU 测试；随后按用户要求删除保存／重载兼容层，改为在原对象上直接拼接权重、绑定 forward，并执行 F.linear 和 split。简化版仅做静态检查，保留注意力对照测试待后续运行；未下载权重，未执行真实权重对齐或 GPU 性能测试。

#### 03. 合并 MLP 投影及激活（Fused Gate/Up、SwiGLU）

- [ ] Gate/Up 投影及 SiLU/Mul 融合代码已实现，待 GPU 验收：参考 O1 `_patch_mlp()`、V1 `OmniVoiceMLP`，加载后调用 `fuse_gate_up_projections()`，在 `mlp.py` 中拼接权重并执行一次 F.linear；CUDA FP16/BF16 的 SiLU 使用 FlashInfer `silu_and_mul()`，一次内核完成激活与逐元素乘法，再调用原 down_proj。CPU、FP32 及非 SiLU 激活保留原计算，不捕获融合内核错误或静默切换。2026-11-11：MLP、Attention 及生成回归 47 项通过，4 项真实 CUDA 对照因本机无 GPU 跳过；全套回归缺少 sglang、全量钩子缺少 pre_commit，GPU 数值、真实权重及性能仍待验证，未下载权重。

#### 04. 归一化融合（Fused RMSNorm）

- [ ] 参考 O1 `_fi_rmsnorm_forward()`、V1 `triton_rms_norm()`；进一步评估 V1 的残差加法与 RMSNorm 融合，减少中间读写和内核启动。

#### 05. 注意力前处理融合（QK Norm / RoPE / Layout Fusion）

- [ ] O1 `_fi_attention_module_forward()` 保持 NHD 布局、分别执行 Q/K 归一化和 FlashInfer RoPE；V2 `fused_qkv_norm_rope()` 则把 Q/K 归一化、RoPE、K/V 广播和输出布局合到一个 Triton 内核。根据设备和依赖选择路径，不把两者描述成相同的融合程度。

#### 06. 批量采样（Vectorized Sampling）

- [ ] 参考 O1 `_generate_iterative_packed()`，批量执行 CFG、log-softmax、argmax、Gumbel；其 top-k 位置选择和写回仍按请求循环。适配时保留本项目每请求独立随机数生成器、不同去掩码数量和结果对应关系；V1 的采样仍逐请求执行。

#### 07. 紧凑序列打包（Sequence Packing）

- [ ] 参考 O1 `_generate_iterative_packed()`、V3 `denoise_step()`，按实际长度排列条件／无条件序列，使用累计边界隔离请求和分支，减少 Padding；核对位置编号、目标偏移、结果还原和取消行为。

#### 08. 变长注意力后端（Variable-length Attention）

- [ ] 参考 O1 `PackedAttnRunner.plan()` 的 ragged attention 和 V1 `OmniVoiceAttention` 的变长后端；与 07 配合，减少 Padding 计算和二维掩码分配。必须保持双向注意力与请求隔离；O1 固定槽位的 bucket graph 路径实际使用 SDPA，不能把所有图模式都称为无 Padding 的 FlashInfer 路径。

#### 09. 固定输入准备复用

- [ ] 部分实现（2026-11-03，31 项微型 CPU 算法测试通过，GPU 待验收）：参考 V1 `forward()` 循环外的 `text_embeds_cached`、`audio_mask_3d`，本项目 `prepare_fixed_inputs()` 在每批、每段迭代前准备文本向量、音频掩码视图和位置编号，所有轮次复用；不保存到模型实例，也不跨请求缓存。音频向量、主干隐藏状态及 Q/K/V 仍每轮计算。RoPE 的 cos/sin 复用仍待单独实现：当前 Transformers 主干入口内部生成它们，后续参考 V1 `_ensure_rope()`、`_rope_table_for()`；CUDA 图路径另行处理。

#### 10. CUDA 图（CUDA Graph）

- [ ] 参考 O1 两种图捕获函数和 V1 `_OmniVoiceCUDAGraphForward`，评估按批大小／长度分桶、静态缓冲区和重放；验证额外 Padding、首次捕获开销、GPU 显存、动态长度、随机采样、取消及并发缓冲区隔离。与通用编译开关分别验证。

#### 11. 按长度组批（Length-aware Batching）

- [ ] 参考 O2 的时长排序和组批函数，评估在现有 `SimpleScheduler` 中降低批内长度差异；在线服务需同时衡量排队时延，不能直接照搬离线排序。

#### 12. 按步调度（Step Scheduling）

- [ ] V3 `denoise_step()` 对活跃请求组批执行一轮，`step_scheduler()` 推进轮次；接入本项目需另行设计调度器。主要面向并发吞吐和请求等待，不能等同于单请求加速；最终仍返回完整音频，不属于本次删除的流式输出。

#### 13. CFG 化简（Single Log-softmax CFG）

- [x] 参考 V1 `_unmask_one_request()` 与 `forward()`，已将 CFG 化简为 `log_softmax((1+s)*conditional - s*unconditional)`；先转 FP32 再做引导计算。保留屏蔽 mask、类别温度、top-10% 候选截断和位置置信度规则，CFG 关闭时只归一化条件分支。

2026-11-01 验证：90 项单测／回归通过，包含原三次归一化公式的数值对照（rtol=1e-5、atol=1e-5）、FP16/BF16、随机采样和随机状态检查；4 项官方权重 CPU FP32 对齐通过，生成码本一致、波形最大误差均为 0。GPU 性能待测；缺少 `pre_commit`，全量钩子未运行。


### 1.3 暂不列为实施项

- **区域编译（Regional Compilation）**：V1 有 `_repeated_blocks` 支持声明；仅据此不能认定已有可直接搬用、默认启用且验证过收益的编译路径，先从实施清单移出。
- **TF32**：V1 `_maybe_enable_tf32()` 是可选的进程级设置，面向支持该格式的 GPU 上的 FP32 计算；本项目主干默认 FP16，不把它作为通用加速项。
- **参考编码复用**：O3 `VoiceClonePrompt` 与 V3 `_speaker_cache` 有实现，但跨请求缓存会改变当前不缓存参考结果的约定。保留参考资料，另行确认范围后再安排。
- **减少迭代步数**：O3 已提供 `num_step`，属于生成参数和质量／速度权衡，不作为本次保持采样设置的实现优化。

### 1.4 接入与验收约定

先建立 GPU 基线，再按第 2 节验证数值、质量、时延、吞吐及峰值 GPU 显存。逐项确认、逐项接入，优先小范围修改；07／08 属于相互配合的布局与后端修改，需一起设计。记录依赖、设备、开关、改动文件和实测结果后再勾选，不把上游性能数字当作本项目收益。

只借鉴加速实现，不照搬生成行为：V1 温度采样直接对完整分布加入 Gumbel，本项目保留官方类别采样的 top-10% 候选截断；批量采样也要保留请求级随机状态。当前长文本每段独立完成生成、最后合并响应的行为不变；不新增跨轮 K/V 复用或静默降级路径。

无参考 TTS、VoiceDesign 当前属于首版范围之外，尚未列为本次承诺的实现任务。

## 2. 技术方案中尚未完成的验证与交付

- [ ] 服务器部署：本机提交推送，服务器在独立目录、环境和端口拉取验证；不影响其他服务。
- [ ] GPU 官方对齐：在相同检查点与生成配置下，比较文本 token、参考码本、目标长度、单步 logits、生成码本和波形，记录精度及容差。
- [ ] 补齐官方对齐场景：明确覆盖跨语言克隆及自然触发分段的长文本；现有强制分段测试不能替代全部长文本验收。
- [ ] GPU HTTP 验证：完整音频、不同长度微批、失败隔离、取消及资源释放。
- [ ] SeedTTS 冒烟：中文和英文各 32 条，保留逐样本结果。
- [ ] SeedTTS 全量：英文 1,088 条；中文使用固定数据版本的完整 split，分别统计词错误率(WER)和字错误率(CER)。
- [ ] 完整质量评测：音色相似度、UTMOS；官方版与接入版使用相同样本、检查点、ASR 和文本归一化。
- [ ] 性能评测：固定环境重复三轮，并发 1/2/4，记录吞吐、P50/P95 时延、实时率(RTF)和峰值 GPU 显存。
- [ ] 覆盖率报告：保留成功、失败、跳过记录；失败不混入成功时延统计，也不从报告中消失。
- [ ] CI 校准：按现有 `apply_wer_slack` 规则确定质量阈值，再更新 `tests/test_model/tts_ci_config.py` 的非流式预设；不预填未经实测的阈值。
- [ ] 验收归档：保存环境版本、模型 revision／权重哈希、评测配置、音频和统计结果，更新技术方案及 Cookbook 的交付状态。

## 3. 已完成基线

四阶段原生接入、音色克隆、完整音频响应、微批及长文本分段已有实现；71 项单测／回归、4 项 CPU FP32 真实权重对齐和本机 HTTP／并发验证已通过。上述结果不能代替 GPU 验证、全量质量评测或服务器部署。
