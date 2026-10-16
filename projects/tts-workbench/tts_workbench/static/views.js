/** Render observations from durable workbench records. */
"use strict";

/** Render an immutable task snapshot alongside current counters. */
function renderDetail(detail) {
    state.detail = detail;
    const statistics = detail?.statistics ?? {},
        run = detail?.run;
    element("stat-sent").textContent = count(statistics.sent);
    element("stat-inflight").textContent =
        `进行中 ${count(statistics.inflight)}`;
    element("stat-success").textContent = count(statistics.success);
    const finished = (statistics.success ?? 0) + (statistics.failed ?? 0);
    element("stat-rate").textContent =
        `成功率 ${finished ? ((statistics.success / finished) * 100).toFixed(1) + "%" : "—"}`;
    element("stat-failed").textContent = count(statistics.failed);
    element("stat-fail-detail").textContent =
        `超时 ${count(statistics.timeout)} · 截断 ${count(statistics.truncated)}`;
    element("stat-p95").textContent = latency(statistics.p95_latency_ms);
    element("latency-average").textContent = latency(
        statistics.average_latency_ms,
    );
    element("latency-p50").textContent = latency(statistics.p50_latency_ms);
    element("latency-p99").textContent = latency(statistics.p99_latency_ms);
    element("target-qpm").textContent = run
        ? count(run.config.requests_per_minute)
        : "—";
    element("unknown-finish").textContent = count(statistics.unknown_finish);
    element("run-status").textContent = run
        ? statusLabels[run.status]
        : "等待开始";
    element("run-status").className =
        `badge ${run?.status === "running" ? "running" : run?.status === "failed" ? "error" : "neutral"}`;
    element("run-description").textContent = run
        ? `${run.config.language === "Chinese" ? "中文" : "英文"}克隆 · ${run.line_ids.length} 条素材 · ${run.config.max_concurrent_requests} 并发`
        : "准备好素材，开始第一次测试";
    const elapsed = run
        ? (run.ended_at ?? Date.now() / 1000) - run.created_at
        : 0;
    const target =
        run?.config.mode === "duration" ? run.config.duration_seconds : null;
    element("run-elapsed").textContent =
        `${duration(elapsed)} / ${target ? duration(target) : run ? (run.config.mode === "single" ? "单句试跑" : "完整一轮") : "—"}`;
    element("progress-fill").style.width =
        `${run ? Math.min(100, target ? (elapsed / target) * 100 : (statistics.covered_lines / run.line_ids.length) * 100) : 0}%`;
    element("coverage").textContent =
        `素材覆盖 ${statistics.covered_lines ?? 0} / ${run?.line_ids.length ?? state.catalog?.line_count ?? 0}`;
    element("skipped").textContent = `跳过调度 ${count(run?.skipped_count)}`;
    element("actual-qpm").textContent =
        `发起 QPM ${run ? count(statistics.actual_requests_per_minute) : "—"}`;
    element("actual-qpm").title =
        "最近最多 60 秒的实际发起频率；首分钟按已运行时间折算";
    element("success-qpm").textContent =
        `成功 QPM ${run ? count(statistics.successful_requests_per_minute) : "—"}`;
    element("success-qpm").title =
        "最近最多 60 秒内成功完成的请求数折算频率，不包含失败请求";
    element("audio-storage").textContent = run
        ? `已保存 ${(run.audio_bytes_saved / 1024 ** 2).toFixed(1)} MiB 抽样音频`
        : "音频按句抽样保存";
    element("clear-run").disabled =
        !run || ["running", "stopping"].includes(run.status);
    const points = detail?.timeline ?? [];
    drawChart(
        element("latency-chart"),
        [
            {
                label: "成功请求均值",
                color: "#6481eb",
                points: points.map((point) => ({
                    time: point.timestamp,
                    value:
                        point.average_latency_ms == null
                            ? null
                            : point.average_latency_ms / 1000,
                })),
            },
        ],
        "秒",
    );
    drawChart(
        element("requests-chart"),
        [
            {
                label: "请求数量",
                color: "#52aaa6",
                points: points.map((point) => ({
                    time: point.timestamp,
                    value: point.sent,
                })),
            },
        ],
        "次 / 时间段",
    );
    const eventLabels = {
        "Task started": "测试已启动，开始按设定频率发送请求。",
        "Task completed": "测试已完成，全部在途请求已结束。",
        "Task stopping": "停止发起新请求，等待已有请求结束。",
        "Task stopped": "测试已停止，结果已保留。",
        "Task interrupted by process restart":
            "工作台进程重启，先前任务标记为中断。",
        "Audio quota reached; request accounting continues":
            "音频抽样配额已满，继续记录请求结果。",
    };
    if (detail?.events.length) {
        element("events").innerHTML = detail.events
            .map(
                (event) =>
                    `<div class="event-row ${escapeText(event.level)}"><time>${shortTime(event.timestamp)}</time><span>${escapeText(eventLabels[event.message] ?? event.message)}</span></div>`,
            )
            .join("");
    } else {
        element("events").innerHTML =
            '<div class="empty-state"><span class="empty-icon">≋</span><strong>这里将记录每次测试的进展</strong><p>开始、停止、配额与异常都可以在这里回看。</p></div>';
    }
}

/** Read bounded request pages without inserting untrusted response HTML. */
async function refreshRequests() {
    if (!state.selectedRun) {
        state.rows = [];
        renderRows();
        return;
    }
    const page = await api(
        `/runs/${state.selectedRun}/requests?offset=${state.offset}&limit=30&failures_only=${element("failures-only").checked}`,
    );
    state.rows = page.records;
    state.total = page.total;
    renderRows();
}

/** Render recorded request outcomes and explicit sampled-audio availability. */
function renderRows() {
    element("request-rows").innerHTML = state.rows.length
        ? state.rows
              .map(
                  (record) =>
                      `<tr><td>#${record.sequence}<small>${shortTime(record.started_at)}</small></td><td><button class="row-text ellipsis" data-request="${record.id}" title="${escapeText(record.text)}">${escapeText(record.text)}</button><small>${escapeText(record.speaker)} · ${escapeText(record.line_id)}${record.reference_text_needs_review ? " · 参考待复核" : ""}</small></td><td><span class="badge ${record.status === "success" ? "" : record.status === "inflight" ? "neutral" : "error"}">${statusLabels[record.status]}</span></td><td>${latency(record.latency_ms)}</td><td>${record.audio_path ? `<button class="text-button" data-request="${record.id}">▶ 试听</button>` : `<button class="text-button muted" data-request="${record.id}">详情</button>`}</td></tr>`,
              )
              .join("")
        : '<tr><td colspan="5" class="table-empty">暂无符合条件的请求记录。</td></tr>';
    element("page-info").textContent =
        `${count(state.total)} 条记录${state.total ? ` · ${state.offset + 1}–${Math.min(state.offset + 30, state.total)}` : ""}`;
    element("previous-page").disabled = state.offset === 0;
    element("next-page").disabled = state.offset + 30 >= state.total;
}

/** Show request diagnostics and playable retained audio together. */
function showRequest(identifier) {
    const record = state.rows.find((row) => row.id === identifier);
    if (!record) {
        return;
    }
    element("request-detail").innerHTML =
        `<span class="badge ${record.status === "success" ? "" : "neutral"}">${statusLabels[record.status]}</span><div class="detail-grid"><div><span>台词 / 角色</span>${escapeText(record.line_id)} · ${escapeText(record.speaker)}</div><div><span>HTTP / 结束原因</span>${record.http_status ?? "—"} / ${escapeText(record.finish_reason ?? "—")}</div><div><span>总时延 / 调度延迟</span>${latency(record.latency_ms)} / ${latency(record.scheduling_delay_ms)}</div><div><span>音频时长 / 大小</span>${record.audio_duration_seconds?.toFixed(2) ?? "—"} s / ${(record.audio_bytes / 1024).toFixed(1)} KiB</div></div><div class="detail-section"><h3>合成文本</h3><p>${escapeText(record.text)}</p></div><div class="detail-section"><h3>服务请求 ID</h3><p class="detail-id">${escapeText(record.service_request_id ?? "服务未返回请求 ID")}</p></div>${record.error ? `<div class="detail-section"><h3>错误 · ${escapeText(record.error_type)}</h3><pre>${escapeText(record.error)}</pre></div>` : ""}<div class="detail-section"><h3>生成音频</h3>${record.audio_path ? `<audio controls src="${API}/audio/${record.id}"></audio>` : '<p class="muted">本次没有保存音频样本，统计记录已保留。</p>'}</div><div class="detail-section"><h3>中文参考音频${record.reference_text_needs_review ? " · 转写待复核" : ""}</h3><audio controls preload="none" src="${API}/references/${encodeURIComponent(record.line_id)}"></audio></div>`;
    element("request-dialog").showModal();
}

/** Render real device readings and distinguish missing hardware from idle GPUs. */
async function refreshGpu() {
    const view = await api(
        `/gpu${state.selectedRun ? `?run_id=${state.selectedRun}` : ""}`,
    );
    element("gpu-timestamp").textContent = view.current.sampled_at
        ? `更新于 ${shortTime(view.current.sampled_at)}`
        : "等待采样";
    element("gpu-devices").innerHTML = view.current.devices.length
        ? view.current.devices
              .map((device) => {
                  const used = device.memory_used_mib,
                      total = device.memory_total_mib;
                  return `<article class="gpu-card"><div class="gpu-card-title"><strong>GPU ${escapeText(device.index)} · ${escapeText(device.name)}</strong><span class="pill-small">实时采样</span></div><div class="gpu-fields"><div class="gpu-field"><span>显存占用</span><strong>${used == null ? "—" : (used / 1024).toFixed(1)} <small>/ ${total == null ? "—" : (total / 1024).toFixed(1)} GiB</small></strong><div class="memory-bar"><i style="width:${total && used != null ? Math.min(100, (used / total) * 100) : 0}%"></i></div></div><div class="gpu-field"><span>GPU 利用率</span><strong>${device.utilization_percent ?? "—"}<small> %</small></strong></div><div class="gpu-field"><span>温度</span><strong>${device.temperature_celsius ?? "—"}<small> °C</small></strong></div><div class="gpu-field"><span>功耗</span><strong>${device.power_watts?.toFixed(0) ?? "—"}<small> W</small></strong></div></div></article>`;
              })
              .join("")
        : `<div class="panel empty-state"><span class="empty-icon">▦</span><strong>GPU 指标暂不可用</strong><p>${escapeText(view.current.error ?? "尚未取得采样结果")}</p><p>工作台部署在 NVIDIA GPU 服务器后，将读取该主机的实际指标。</p></div>`;
    const identifiers = [...new Set(view.history.map((sample) => sample.uuid))],
        colors = ["#6481eb", "#46aba3", "#ca9760", "#a378c5"];
    drawChart(
        element("gpu-chart"),
        identifiers.map((identifier, index) => ({
            label: `GPU ${view.history.find((sample) => sample.uuid === identifier).index}`,
            color: colors[index % colors.length],
            points: view.history
                .filter((sample) => sample.uuid === identifier)
                .map((sample) => ({
                    time: sample.timestamp,
                    value: sample.utilization_percent,
                })),
        })),
        "%",
        100,
    );
}

/** Keep an incremental bounded log window, including file-rotation resets. */
async function refreshLogs() {
    const result = await api(
        `/logs${state.logCursor ? `?cursor=${encodeURIComponent(state.logCursor)}` : ""}`,
    );
    element("log-source").textContent = result.source ?? "尚未配置日志文件";
    if (result.error) {
        element("log-output").textContent = result.error;
        element("log-status").textContent = "日志暂不可用";
        return;
    }
    state.logCursor = result.cursor;
    state.logText = ((result.reset ? "" : state.logText) + result.text)
        .replace(/\x1b\[[0-9;]*m/g, "")
        .split("\n")
        .slice(-1500)
        .join("\n")
        .slice(-512000);
    element("log-status").textContent = state.autoScroll
        ? "实时跟随"
        : "持续读取 · 已暂停滚动";
    renderLogs();
}

/** Filter only the displayed window while preserving its unfiltered buffer. */
function renderLogs() {
    const keyword = element("log-filter").value.toLowerCase(),
        output = element("log-output");
    const position = output.scrollTop;
    output.textContent = keyword
        ? state.logText
              .split("\n")
              .filter((line) => line.toLowerCase().includes(keyword))
              .join("\n")
        : state.logText || "日志文件已连接，等待新内容…";
    output.scrollTop = state.autoScroll ? output.scrollHeight : position;
}

/** Copy visible logs with a fallback for HTTP hosts lacking clipboard access. */
async function copyLogs() {
    try {
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(
                element("log-output").textContent,
            );
        } else {
            const field = document.createElement("textarea");
            field.value = element("log-output").textContent;
            field.style.position = "fixed";
            field.style.opacity = "0";
            document.body.append(field);
            field.select();
            const copied = document.execCommand("copy");
            field.remove();
            if (!copied) {
                throw new Error("请手动选择日志并复制");
            }
        }
        notify("当前日志窗口已复制。", true);
    } catch (error) {
        notify(`复制失败：${error.message}`);
    }
}
