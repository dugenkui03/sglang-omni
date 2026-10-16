/** Bind the standalone browser UI to real, durable workbench APIs. */
"use strict";

const API = "/api/workbench";
const state = {
    catalog: null,
    mode: "duration",
    tab: "overview",
    selectedRun: null,
    activeId: null,
    detail: null,
    rows: [],
    offset: 0,
    total: 0,
    logCursor: null,
    logText: "",
    logsPaused: false,
    logController: null,
    logLineLimit: 20000,
    polling: false,
    servicePolling: false,
    submitting: false,
};
const statusLabels = {
    running: "运行中",
    stopping: "等待请求结束",
    completed: "已完成",
    stopped: "已停止",
    interrupted: "已中断",
    failed: "失败",
    inflight: "进行中",
    success: "成功",
    timeout: "超时",
    truncated: "截断",
};

/** Resolve a fixed element identifier rather than relying on browser globals. */
function element(identifier) {
    return document.getElementById(identifier);
}

/** Read JSON APIs and surface validation details without swallowing failures. */
async function api(path, options = {}) {
    const response = await fetch(API + path, {
        ...options,
        headers: {
            "Content-Type": "application/json",
            "X-Workbench-Client": "1",
            ...options.headers,
        },
        signal: options.signal ?? AbortSignal.timeout(20000),
    });
    const payload = await response.json();
    if (!response.ok) {
        const detail = Array.isArray(payload.detail)
            ? payload.detail.map((issue) => issue.msg).join("；")
            : payload.detail;
        throw new Error(detail || `HTTP ${response.status}`);
    }
    return payload;
}

/** Show an actionable operation result outside the scrollable data panels. */
function notify(message, success = false) {
    const notice = element("notice");
    notice.textContent = message;
    notice.className = `notice${success ? " success" : ""}`;
    notice.hidden = false;
}

/** Format a count while retaining zero as a valid observation. */
function count(value) {
    return Number(value ?? 0).toLocaleString("zh-CN", {
        maximumFractionDigits: 1,
    });
}

/** Format measured latency and leave missing observations blank. */
function latency(milliseconds) {
    return milliseconds == null
        ? "—"
        : milliseconds >= 1000
          ? `${(milliseconds / 1000).toFixed(2)} s`
          : `${Math.round(milliseconds)} ms`;
}

/** Format an elapsed duration without wrapping at 24 hours. */
function duration(seconds) {
    seconds = Math.max(0, Math.floor(seconds));
    const hours = Math.floor(seconds / 3600),
        minutes = Math.floor((seconds % 3600) / 60);
    return `${hours ? `${hours}:` : ""}${String(minutes).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

/** Switch observation panels while leaving the task entirely server-owned. */
function selectTab(tab) {
    state.tab = tab;
    document
        .querySelectorAll(".tab-content")
        .forEach((panel) => (panel.hidden = panel.id !== `tab-${tab}`));
    document.querySelectorAll("[data-tab]").forEach((button) => {
        const selected = button.dataset.tab === tab;
        button.classList.toggle("active", selected);
        button.setAttribute("aria-selected", String(selected));
    });
    poll();
}

/** Refresh service availability separately from the workload polling cycle. */
async function refreshService() {
    if (state.servicePolling) {
        return;
    }
    state.servicePolling = true;
    try {
        const service = await api("/service");
        element("api-base").textContent = service.api_base;
        element("service-status").textContent = service.healthy
            ? "TTS 服务在线"
            : "TTS 服务未就绪";
        element("service-status").className =
            `badge ${service.healthy ? "" : "error"}`;
        element("service-status").title =
            service.error ?? "健康检查通过；生成能力以实际请求为准";
    } catch (error) {
        element("service-status").textContent = "工作台连接异常";
        element("service-status").className = "badge error";
        element("service-status").title = error.message;
    } finally {
        state.servicePolling = false;
    }
}

/** Poll server-owned state without overlapping cycles or resetting edited inputs. */
async function poll() {
    if (state.polling) {
        return;
    }
    state.polling = true;
    try {
        const collection = await api("/runs");
        state.activeId = collection.active_id;
        if (
            !state.selectedRun ||
            !collection.runs.some((run) => run.id === state.selectedRun)
        ) {
            state.selectedRun = collection.runs[0]?.id ?? null;
            state.offset = 0;
        }
        const history = element("history"),
            choices =
                collection.runs
                    .map(
                        (run) =>
                            `<option value="${run.id}">${escapeText(run.name)} · ${statusLabels[run.status]}</option>`,
                    )
                    .join("") || '<option value="">尚无测试任务</option>';
        if (history.innerHTML !== choices) {
            history.innerHTML = choices;
        }
        history.value = state.selectedRun ?? "";
        element("stop").disabled = !state.activeId;
        if (state.selectedRun) {
            renderDetail(await api(`/runs/${state.selectedRun}`));
        } else {
            state.total = 0;
            state.rows = [];
            renderDetail(null);
            renderRows();
        }
        if (collection.runtime_error) {
            notify(`后台任务异常：${collection.runtime_error}`);
        }
        if (state.tab === "requests") {
            await refreshRequests();
        } else if (state.tab === "gpu") {
            await refreshGpu();
        } else if (state.tab === "logs" && !state.logsPaused) {
            await refreshLogs();
        }
        updateEstimate();
    } catch (error) {
        notify(`读取运行状态失败：${error.message}`);
    } finally {
        state.polling = false;
    }
}

/** Attach controls once and initialize the actual dataset and task history. */
async function initialize() {
    generateRunName();
    element("log-limit").value =
        localStorage.getItem("workbench-log-limit") ?? "20000";
    updateLogLimit();
    document
        .querySelectorAll("[data-tab]")
        .forEach((button) =>
            button.addEventListener("click", () =>
                selectTab(button.dataset.tab),
            ),
        );
    document.querySelectorAll("[data-mode]").forEach((button) =>
        button.addEventListener("click", () => {
            state.mode = button.dataset.mode;
            document
                .querySelectorAll("[data-mode]")
                .forEach((control) =>
                    control.classList.toggle("selected", control === button),
                );
            if (state.mode === "single") {
                document.querySelector(".reference-panel").open = true;
            }
            updateEstimate();
        }),
    );
    element("run-form").addEventListener("submit", startRun);
    element("stop").addEventListener("click", stopRun);
    element("clear-run").addEventListener("click", () => clearRuns(false));
    element("clear-all").addEventListener("click", () => clearRuns(true));
    element("refresh-service").addEventListener("click", refreshService);
    element("speaker").addEventListener("change", updateLines);
    element("line").addEventListener("change", updateReference);
    ["qpm", "duration-preset", "duration-value", "duration-unit"].forEach(
        (identifier) =>
            element(identifier).addEventListener("input", updateEstimate),
    );
    element("select-all").addEventListener("click", () => {
        document
            .querySelectorAll(".unit-check")
            .forEach((input) => (input.checked = true));
        updateSpeakers();
    });
    element("history").addEventListener("change", () => {
        state.selectedRun = element("history").value;
        state.offset = 0;
        localStorage.setItem("workbench-run", state.selectedRun);
        poll();
    });
    element("failures-only").addEventListener("change", () => {
        state.offset = 0;
        poll();
    });
    element("previous-page").addEventListener("click", () => {
        state.offset = Math.max(0, state.offset - 30);
        poll();
    });
    element("next-page").addEventListener("click", () => {
        state.offset += 30;
        poll();
    });
    element("request-rows").addEventListener("click", (event) => {
        const playback = event.target.closest("[data-play]");
        if (playback) {
            playResult(playback.dataset.play);
            return;
        }
        const button = event.target.closest("[data-request]");
        if (button) {
            showRequest(button.dataset.request);
        }
    });
    ["play", "pause", "ended"].forEach((event) =>
        element("result-audio").addEventListener(event, updatePlaybackButtons),
    );
    element("close-dialog").addEventListener("click", () => {
        element("request-dialog").close();
        element("request-detail").innerHTML = "";
    });
    element("pause-log").addEventListener("click", toggleLogFetching);
    element("log-limit").addEventListener("change", updateLogLimit);
    element("log-filter").addEventListener("input", renderLogs);
    element("copy-log").addEventListener("click", copyLogs);
    try {
        state.catalog = await api("/dataset");
        state.selectedRun = localStorage.getItem("workbench-run");
        element("dataset-summary").textContent =
            `${state.catalog.units.length} 个配音单元 · ${state.catalog.line_count} 条真实台词`;
        element("unit-options").innerHTML = state.catalog.units
            .map(
                (unit) =>
                    `<label><input class="unit-check" type="checkbox" value="${escapeText(unit.unit_id)}" checked>${escapeText(unit.title.replace("甄嬛传 ", "").replace(" ", ""))}<span>${unit.line_count}</span></label>`,
            )
            .join("");
        document
            .querySelectorAll(".unit-check")
            .forEach((input) =>
                input.addEventListener("change", updateSpeakers),
            );
        updateSpeakers();
        await poll();
    } catch (error) {
        notify(`初始化失败：${error.message}`);
    }
    refreshService();
    setInterval(poll, 2000);
    setInterval(refreshService, 10000);
}

initialize();
