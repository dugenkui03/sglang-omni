# SGLang-Omni Documentation

Most docs live under `docs/` as Markdown cookbooks and guides. Start there if you are new to the codebase.

## Docs Workflow

### Install Dependency

```bash
apt-get update && apt-get install -y pandoc parallel retry
pip install -r requirements.txt
```

### Update Documentation

Edit the Markdown (or RST) under `docs/`. New pages must be listed in `index.rst` (or the relevant toctree).

- Run `pre-commit run --all-files` before opening a PR. Re-run once if the first pass auto-fixes files.
- Preview locally:

```bash
bash serve.sh

# custom port
PORT=8080 make serve
```

## Style

- Prefer **Markdown** for cookbooks and usage guides. Use notebooks only when the example must be executed in docs CI.
- Prefer relative links (`../get_started/installation.md`), not absolute docs URLs.
- Match existing launch / request examples in neighboring cookbooks.

## 中文学习笔记

- [8.1 Qwen3-TTS 数据流：预处理部分](<备注/8.1 qwen3-tts 数据流：预处理部分.md>)
- [8.2 Qwen3-TTS 推理组件：组合、继承与职责](<备注/8.2 qwen3-tts 推理组件：组合、继承与职责.md>)（[两页 A4 横版 PDF](<备注/pdf/8.2 qwen3-tts 推理组件：组合、继承与职责.pdf>)）
