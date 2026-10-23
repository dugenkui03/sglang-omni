# OmniVoice

OmniVoice 原生接入使用 `preprocessing → audio_encoder → tts_engine → vocoder` 四个 Stage。生成阶段由 `SimpleScheduler` 组批，执行双向注意力和迭代去掩码；不使用自回归 KV cache，也不调用官方整段 `generate()`。首版提供一条参考音频及其文本的音色克隆，返回完整音频。

## 安装与启动

按仓库安装说明准备独立环境。保留当前项目的 PyTorch、Transformers 和 SGLang 版本；新增依赖为 `pydub>=0.25.1`，不需要安装 OmniVoice 的演示或训练依赖。

```bash
sgl-omni serve --model-path k2-fsa/OmniVoice \
  --config examples/configs/omnivoice.yaml --host 127.0.0.1 --port 8000
```

从 Hugging Face 官方仓库下载完整快照，包括 `audio_tokenizer/`；可通过 `k2-fsa/OmniVoice@<revision>` 固定版本。验收时记录模型 revision、配置及权重 SHA256、依赖版本和 GPU 型号。缺少 Codec 时启动失败，不自动替换为另一版本。

默认单 GPU、同一进程，共享一份 FP32 Codec，主干为 FP16，批大小为 1。扩大批次示例：

```bash
sgl-omni serve --model-path k2-fsa/OmniVoice \
  --tts_engine.factory.max_batch_size 4 \
  --vocoder.factory.max_batch_size 4 --port 8000
```

CPU 数值验证需给三个 GPU Stage 显式设置 `--audio_encoder.factory.device cpu --tts_engine.factory.device cpu --vocoder.factory.device cpu`，并使用 `--tts_engine.factory.dtype float32`。GPU 性能必须单独测量。

## 请求

复用 `POST /v1/audio/speech`，参考音频支持受现有媒体权限限制的路径、URL 和 data URL。本地路径需配置 `--allowed-local-media-path`。

```json
{
  "model": "k2-fsa/OmniVoice",
  "input": "Welcome home.",
  "references": [{"audio_path": "data:audio/wav;base64,...", "text": "欢迎回家。"}],
  "language": "English",
  "speed": 1.0,
  "seed": 1234,
  "response_format": "wav",
  "stream": false,
  "stage_params": {
    "tts_engine": {"num_step": 32, "guidance_scale": 2.0},
    "preprocessing": {"preprocess_prompt": true},
    "vocoder": {"postprocess_output": true, "pad_duration": 0.1, "fade_duration": 0.1}
  }
}
```

也可使用 `ref_audio`＋`ref_text`。参考文本必填，不自动启动 ASR。`language` 支持官方语言表、语言名称以及 `auto`；不额外筛除短音频，无法编码的输入会返回失败。参考预处理遵循官方静音处理和 RMS 调整；可用 `preprocess_prompt=false` 关闭静音处理。不会因为参考超过 20 秒而截断带文本的输入。

`stage_params.tts_engine` 接受 `num_step=32`、`guidance_scale=2`、`t_shift=0.1`、`layer_penalty_factor=5`、`position_temperature=5`、`class_temperature=0`、`denoise=true`、可选 `duration`（秒）、`audio_chunk_duration=15` 和 `audio_chunk_threshold=30`。显式 duration 优先于 speed；实际波形时长还受官方静音处理、段间间隔和边缘 padding 影响。码本数、帧率与采样率从检查点读取。

长文本沿用官方标点分段策略，每段使用相同参考音色，不截断剩余台词。无标点的超长单句可能超过上下文上限，此时明确报错。不同配置分别组批，相同配置的不同长度使用有效位置掩码；每个请求持有独立随机数生成器。固定输入及计算形状可复现，不保证不同精度或不同批次形状下逐位一致。

首版不提供流式输出、VoiceDesign、无参考 TTS、TP、CUDA Graph 或跨请求音色缓存。AR 参数 `max_new_tokens/temperature/top_p/top_k/repetition_penalty` 会报错；使用上述模型参数。日志只输出请求 ID、阶段、耗时和形状。取消在迭代／分段边界生效，不返回半成品。

## 测试与评测

```bash
python -m pytest tests/unit_test/omnivoice tests/unit_test/benchmarks/test_tts_seedtts.py -q
python -m benchmarks.eval.benchmark_tts_seedtts --model k2-fsa/OmniVoice \
  --lang en --max-samples 32 --output-dir results/omnivoice_en_smoke
python -m benchmarks.eval.benchmark_tts_seedtts --model k2-fsa/OmniVoice \
  --lang zh --max-samples 32 --output-dir results/omnivoice_zh_smoke
```

去掉 `--max-samples` 后跑完整 split。英文全量 1,088 条；中文按字统计 CER，沿用共享评测代码的 `wer_corpus` 字段。评测默认并发 1、warmup 1、seed 1234；不发送 AR engine 参数或 max_new_tokens。使用 `--generate-only`／`--transcribe-only` 可分开运行 TTS 和 Qwen3-ASR，避免同时占用 GPU 显存。

保存 `speed_results.json`、`generated.json`、`wer_results.json` 及逐样本音频；必须报告成功、失败、跳过数量。完整验收使用同一检查点、ASR、样本及生成参数对比官方结果，重复三轮；质量门槛按 `apply_wer_slack` 校准，性能报告并发 1/2/4 的吞吐、P50/P95、RTF 和峰值 GPU 显存。没有全量 GPU 实测前不添加伪造的 CI 阈值。

官方逐阶段对齐测试（参考仓库只读）：

```bash
OMNIVOICE_UPSTREAM_SOURCE=/path/to/OmniVoice \
OMNIVOICE_PARITY_CHECKPOINT=/path/to/complete/checkpoint \
OMNIVOICE_PARITY_REFERENCE=/path/to/reference.wav \
OMNIVOICE_PARITY_REFERENCE_TEXT='参考音频的完整文本' \
python -m pytest tests/test_model/test_omnivoice_parity.py -v -s
```

默认用 CUDA；`OMNIVOICE_PARITY_DEVICE=cpu` 可验证数值一致性。测试覆盖英文、中文、内部多段拼接和默认 32 步随机采样，比较参考码本、提示输入、首轮 logits、目标帧数、生成码本和波形。用 `OMNIVOICE_PARITY_OUTPUT_DIR` 保存验证音频和记录。参考源码固定为 `08be0b4ccbac3e13e374e86fbfead4b4cac343e2`；缺少配置时测试显式跳过。

## 来源与部署

2026-10-24 本机验证：PyTorch 2.14.1、Transformers 5.19.0，CPU FP32；四个真实权重对齐场景均通过，波形最大绝对误差为 0。正式 CLI 启动后，HTTP 请求返回 24 kHz WAV；并发三个相同 seed 的请求全部成功，日志确认批大小 2，三个 WAV 的 SHA256 相同。无效步数及流式请求返回 HTTP 400。这些结果不代表 GPU 性能或全量 WER/CER 已通过；GPU 实测及 CI 阈值校准仍待可用测试资源。

本次检查点 SHA256：

| 文件 | SHA256 |
|---|---|
| `config.json` | `5e359117e13b420c5e0c925d4aba650d624767131f1d1746928f8b850d5dc372` |
| `model.safetensors` | `730839316de585f4c8298ec0e1712efc10fb19c6fa4e36eb741cb8d51ebcf6aa` |
| `audio_tokenizer/config.json` | `eefb20806f7104e77c9a5277c9df0f9bb8826b08eb1d4e8ab2b9829b6ef9fac1` |
| `audio_tokenizer/model.safetensors` | `fe7c5e8785e0a05833e1bfc3e002ec7f55af21e306b2e7154a448c1f54ccfb0d` |

模型及必要文本／音频算法来自 [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)，保留 Xiaomi 的 Apache-2.0 授权及版权说明，见模型目录的 `LICENSE`。本项目只迁移推理代码，不修改参考仓库。

本机提交推送后，服务器在独立目录／环境拉取分支并验证；不要切换正在运行其他模型的工作目录、更新其环境或抢占其 GPU 显存。GPU 条件不足时保留验证状态，不能将 CPU 对齐结果当作 GPU 性能或全量 WER/CER 验收。
