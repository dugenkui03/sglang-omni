/** Play sampled results directly and inspect the actual outgoing HTTP inputs. */
"use strict";

/** Start audio within the click gesture and reuse one player across table refreshes. */
function playResult(identifier) {
    const record = state.rows.find((row) => row.id === identifier);
    if (!record?.audio_path) {
        return;
    }
    const audio = element("result-audio");
    if (audio.dataset.request === identifier && !audio.paused) {
        audio.pause();
        return;
    }
    if (audio.dataset.request !== identifier) {
        audio.src = `${API}/audio/${identifier}`;
        audio.dataset.request = identifier;
    }
    element("result-player").hidden = false;
    element("result-audio-label").textContent =
        `#${record.sequence} · ${record.speaker} · ${record.text}`;
    audio.play().catch((error) => notify(`音频播放失败：${error.message}`));
}

/** Keep row playback labels synchronized without replacing the active audio element. */
function updatePlaybackButtons() {
    const audio = element("result-audio");
    document.querySelectorAll("[data-play]").forEach((button) => {
        button.textContent =
            button.dataset.play === audio.dataset.request && !audio.paused
                ? "Ⅱ 暂停"
                : "▶ 试听";
    });
}

/** Show HTTP inputs alongside recorded results without automatically playing audio. */
async function showRequest(identifier) {
    const record = state.rows.find((row) => row.id === identifier);
    if (!record) {
        return;
    }
    const dialog = element("request-dialog");
    dialog.dataset.request = identifier;
    element("request-detail").innerHTML =
        `<span class="badge ${record.status === "success" ? "" : "neutral"}">${statusLabels[record.status]}</span><div class="detail-grid"><div><span>台词 / 角色</span>${escapeText(record.line_id)} · ${escapeText(record.speaker)}</div><div><span>HTTP / 结束原因</span>${record.http_status ?? "—"} / ${escapeText(record.finish_reason ?? "—")}</div><div><span>总时延 / 调度延迟</span>${latency(record.latency_ms)} / ${latency(record.scheduling_delay_ms)}</div><div><span>音频时长 / 大小</span>${record.audio_duration_seconds?.toFixed(2) ?? "—"} s / ${(record.audio_bytes / 1024).toFixed(1)} KiB</div></div><div class="detail-section"><h3>服务请求 ID</h3><p class="detail-id">${escapeText(record.service_request_id ?? "服务未返回请求 ID")}</p></div><div id="http-request-detail" class="detail-section">正在读取 HTTP 请求参数…</div>${record.error ? `<div class="detail-section"><h3>错误 · ${escapeText(record.error_type)}</h3><pre>${escapeText(record.error)}</pre></div>` : ""}<div class="detail-section"><h3>中文参考音频${record.reference_text_needs_review ? " · 转写待复核" : ""}</h3><audio controls preload="none" src="${API}/references/${encodeURIComponent(record.line_id)}"></audio></div>`;
    dialog.showModal();
    try {
        const details = await api(`/requests/${identifier}/http`);
        if (dialog.dataset.request !== identifier || !dialog.open) {
            return;
        }
        const preview = {
            ...details.body,
            references: details.body.references.map((reference) => ({
                ...reference,
                audio_path:
                    reference.audio_path.length > 120
                        ? `${reference.audio_path.slice(0, 60)}… [共 ${reference.audio_path.length.toLocaleString()} 字符，下方可展开完整 JSON]`
                        : reference.audio_path,
            })),
        };
        element("http-request-detail").innerHTML =
            `<h3>请求 sglang-omni 的 HTTP 接口</h3><p class="detail-id">${escapeText(details.method)} ${escapeText(details.url)}</p><p class="muted">${escapeText(details.note)}</p><pre class="request-json">${escapeText(JSON.stringify(details.headers, null, 2))}</pre><h3>JSON 入参</h3><pre class="request-json">${escapeText(JSON.stringify(preview, null, 2))}</pre><details class="raw-request"><summary>完整请求 JSON（含参考音频 Base64）</summary><pre class="request-json">${escapeText(JSON.stringify(details.body, null, 2))}</pre></details>`;
    } catch (error) {
        if (dialog.dataset.request === identifier && dialog.open) {
            element("http-request-detail").textContent =
                `读取请求参数失败：${error.message}`;
        }
    }
}
