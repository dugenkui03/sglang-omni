# SPDX-License-Identifier: Apache-2.0
"""OmniVoice 非自回归语音克隆流水线。"""

from sglang_omni.models.model_capabilities import ModelCapabilities

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=True,
    supports_batch_vocoder=True,
    supports_streaming_vocoder=False,
    supports_cuda_graph=False,
    supports_torch_compile=False,
    supports_breakable_prefill_cuda_graph=False,
    supports_full_prefill_cuda_graph=False,
)
