/** Render observations from durable workbench records. */
"use strict";

const DEFAULT_LOG_LINES = 2000;
const MAX_LOG_LINES = 30000;
const LOG_LIMIT_STORAGE_KEY = "workbench-log-limit-v2";

/** 展示历史任务配置、抽样方式与当前统计，不使用正在编辑的表单值。 */
function renderDetail(detail) {
    state.detail = detail;
    const statistics = detail?.statistics ?? {},
        run = detail?.run;
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
        ? `${run.config.language === "Chinese" ? "中文原声克隆" : "中文参考 → 英文配音"} · ${run.config.sample_order === "random" ? `乱序请求 · 重复 ${count(run.config.repeat_probability * 100)}%` : "顺序请求"} · ${run.line_ids.length} 条素材 · ${count(run.config.requests_per_minute)} QPM`
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
    const taskName = state.requestTaskName,
        runId = state.requestRunId || state.selectedRun,
        offset = state.offset,
        failuresOnly = element("failures-only").checked;
    if (!runId && !taskName) {
        state.rows = [];
        state.total = 0;
        renderRows();
        return;
    }
    const query = new URLSearchParams({
        offset,
        limit: 30,
        failures_only: failuresOnly,
    });
    if (taskName) {
        query.set("task_name", taskName);
    }
    const path = taskName ? "/requests" : `/runs/${runId}/requests`;
    const page = await api(`${path}?${query}`);
    if (
        taskName !== state.requestTaskName ||
        runId !== (state.requestRunId || state.selectedRun) ||
        offset !== state.offset ||
        failuresOnly !== element("failures-only").checked
    ) {
        return;
    }
    state.rows = page.records;
    state.total = page.total;
    renderRows();
}

/** Search task names across stored history while keeping normal task selection intact. */
function applyRequestTaskFilter() {
    state.requestTaskName = element("request-task-name").value.trim();
    state.requestRunId = element("request-task-select").value;
    state.offset = 0;
    state.rows = [];
    state.total = 0;
    element("request-search-scope").textContent = state.requestTaskName
        ? `全部历史任务 · 名称包含「${state.requestTaskName}」`
        : state.requestRunId
          ? "已选择任务 · 按创建时间倒序"
          : "当前所选任务";
    element("result-audio").pause();
    element("result-player").hidden = true;
    renderRows();
    poll();
}

/** Render recorded request outcomes and explicit sampled-audio availability. */
function renderRows() {
    const rowsHtml = state.rows.length
        ? state.rows
              .map(
                  (record) =>
                      `<tr><td>#${record.sequence}<small>${shortTime(record.started_at)}</small></td><td><button class="row-text ellipsis" data-request="${record.id}" title="${escapeText(record.text)}">${escapeText(record.text)}</button><small>${escapeText(record.speaker)} · ${escapeText(record.line_id)}${record.reference_text_needs_review ? " · 参考待复核" : ""}</small>${record.run_name ? `<small>任务：${escapeText(record.run_name)}</small>` : ""}</td><td><span class="badge ${record.status === "success" ? "" : record.status === "inflight" ? "neutral" : "error"}">${statusLabels[record.status]}</span></td><td>${latency(record.latency_ms)}</td><td><div class="request-actions">${record.audio_path ? `<button class="text-button" data-play="${record.id}">▶ 试听</button>` : ""}<button class="text-button" data-request="${record.id}">详情</button>${record.service_request_id ? `<a class="text-button" href="${API}/logs/download?${new URLSearchParams({ request_id: record.service_request_id })}" download title="${escapeText(record.service_request_id)}">下载日志</a>` : `<button class="text-button" disabled title="服务未返回 request_id，暂无法关联日志">下载日志</button>`}</div></td></tr>`,
              )
              .join("")
        : '<tr><td colspan="5" class="table-empty">暂无符合条件的请求记录。</td></tr>';
    if (element("request-rows").innerHTML !== rowsHtml) {
        element("request-rows").innerHTML = rowsHtml;
    }
    element("page-info").textContent =
        `${count(state.total)} 条记录${state.total ? ` · ${state.offset + 1}–${Math.min(state.offset + 30, state.total)}` : ""}`;
    element("previous-page").disabled = state.offset === 0;
    element("next-page").disabled = state.offset + 30 >= state.total;
    updatePlaybackButtons();
}

/** Render device-wide metrics and the rolling thirty-minute monitoring window. */
async function refreshGpu() {
    const view = await api("/gpu");
    element("gpu-timestamp").textContent = view.current.sampled_at
        ? `更新于 ${shortTime(view.current.sampled_at)}`
        : "等待采样";
    element("gpu-devices").innerHTML = view.current.devices.length
        ? view.current.devices
              .map((device) => {
                  const used = device.memory_used_mib,
                      total = device.memory_total_mib,
                      memoryPercent =
                          total && used != null ? (used / total) * 100 : null;
                  const metrics = [
                      [
                          "显存已用 / 总量",
                          `${used == null ? "—" : (used / 1024).toFixed(2)} / ${total == null ? "—" : (total / 1024).toFixed(2)}`,
                          "GiB",
                          "设备已分配显存与可见总量；1 GiB = 1,024 MiB。",
                      ],
                      [
                          "显存占比",
                          memoryPercent?.toFixed(1) ?? "—",
                          "%",
                          "已用显存 ÷ 总显存，表示容量占用比例。",
                      ],
                      [
                          "估算显存余量",
                          used == null || total == null
                              ? "—"
                              : ((total - used) / 1024).toFixed(2),
                          "GiB",
                          "总显存减已用显存；驱动保留会影响实际可分配空间。",
                      ],
                      [
                          "计算利用率",
                          device.utilization_percent ?? "—",
                          "%",
                          "设备采样周期内有计算内核运行的时间占比，不是核心数量占比。",
                      ],
                      [
                          "GPU 温度",
                          device.temperature_celsius ?? "—",
                          "°C",
                          "设备传感器读取的当前 GPU 核心温度。",
                      ],
                      [
                          "GPU 功耗",
                          device.power_watts?.toFixed(1) ?? "—",
                          "W",
                          "驱动报告的整卡功率，包含显存等部件。",
                      ],
                  ];
                  return `<article class="gpu-card"><div class="gpu-card-title"><strong>GPU ${escapeText(device.index)} · ${escapeText(device.name)}</strong><span class="pill-small">每 2 秒采样</span></div><div class="gpu-fields">${metrics.map(([label, value, unit, definition]) => `<div class="gpu-field"><span>${label}</span><strong>${value} <small>${unit}</small></strong><p>${definition}</p></div>`).join("")}</div></article>`;
              })
              .join("")
        : `<div class="panel empty-state"><span class="empty-icon">▦</span><strong>GPU 指标暂不可用</strong><p>${escapeText(view.current.error ?? "尚未取得采样结果")}</p></div>`;
    const identifiers = [...new Set(view.history.map((sample) => sample.uuid))],
        colors = ["#6481eb", "#46aba3", "#ca9760", "#a378c5"],
        now = Date.now() / 1000;
    for (const [chart, field, divisor, unit, maximum] of [
        ["gpu-chart", "utilization_percent", 1, "%", 100],
        ["gpu-memory-chart", "memory_used_mib", 1024, "GiB", null],
    ]) {
        drawChart(
            element(chart),
            identifiers.map((identifier, index) => ({
                label: `GPU ${view.history.find((sample) => sample.uuid === identifier).index}`,
                color: colors[index % colors.length],
                points: view.history
                    .filter((sample) => sample.uuid === identifier)
                    .map((sample) => ({
                        time: sample.timestamp,
                        value:
                            sample[field] == null
                                ? null
                                : sample[field] / divisor,
                    })),
            })),
            unit,
            maximum,
            [now - 30 * 60, now],
        );
    }
}
