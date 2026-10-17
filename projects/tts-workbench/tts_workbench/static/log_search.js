/** 搜索全部保留日志，页面不自动刷新或滚动。 */
"use strict";

/** 仅在搜索或翻页时获取日志，避免刷新打断阅读。 */
async function refreshLogs(page = state.logPage) {
    state.logController?.abort();
    const controller = new AbortController();
    state.logController = controller;
    renderLogPagination();
    element("log-status").textContent = "正在搜索全部保留历史…";
    try {
        const query = new URLSearchParams({
            request_id: state.logRequestId, keyword: state.logKeyword,
            location: state.logLocation, scope: state.logScope, limit: state.logLineLimit,
            page,
            order: state.logOrder,
        });
        if (state.logSnapshot !== null) query.set("snapshot", state.logSnapshot);
        const result = await api(`/logs/search?${query}`, {
            signal: AbortSignal.any([controller.signal, AbortSignal.timeout(30000)]),
        });
        if (controller.signal.aborted) return;
        state.logResult = result;
        state.logSnapshot = result.snapshot;
        state.logPage = result.page;
        state.logText = result.records.map(record => record.text).join("");
        renderLogs();
        element("log-status").textContent = result.warning
            ? `已读取历史；新增日志暂不可用：${result.warning}`
            : `匹配 ${count(result.matched_records)} 条记录 / ${count(result.matched_lines)} 行 · 已保留 ${count(result.retained_lines)} 行 · 不自动刷新`;
    } catch (error) {
        if (!controller.signal.aborted) {
            element("log-status").textContent = `搜索失败：${error.message}。可缩小范围或点击刷新重试。`;
        }
    } finally {
        if (state.logController === controller) {
            state.logController = null;
            renderLogPagination();
        }
    }
}

/** 组合请求 ID、关键字和代码位置条件，并重置分页。 */
function applyLogRequestFilter() {
    state.logRequestId = element("log-request-id").value.trim();
    state.logKeyword = element("log-filter").value.trim();
    state.logLocation = element("log-location").value.trim();
    state.logScope = element("log-scope").value || "all";
    state.logOrder = element("log-order").value || "desc";
    state.logSnapshot = null;
    state.logPage = 1;
    state.logResult = null;
    state.logText = "";
    renderLogs();
    updateLogDownload();
    refreshLogs();
}

/** 输入请求 ID 后允许下载该请求的全部保留日志。 */
function updateLogDownload() {
    element("download-request-log").disabled = !element("log-request-id").value.trim();
}

/** 设置每页行数预算，同时保持单条日志完整。 */
function updateLogLimit() {
    state.logLineLimit = Math.max(1, Math.min(MAX_LOG_LINES,
        Math.floor(Number(element("log-limit").value)) || DEFAULT_LOG_LINES));
    element("log-limit").value = state.logLineLimit;
    localStorage.setItem(LOG_LIMIT_STORAGE_KEY, String(state.logLineLimit));
}

/** 格式化 JSON 内容，其他日志行保持原样。 */
function formatLogText(text) {
    return text.split("\n").map(line => {
        const trimmed = line.trim();
        if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) return line;
        try { return JSON.stringify(JSON.parse(trimmed), null, 2); }
        catch { return line; }
    }).join("\n");
}

/** 默认折叠每条记录，并提供代码位置和请求 ID 搜索入口。 */
function renderLogs() {
    const records = state.logResult?.records ?? [];
    element("log-output").innerHTML = records.length ? records.map(record => `
        <details class="log-record">
            <summary>
                <span class="log-time">${escapeText(record.time)}<small>${escapeText(record.level)}</small></span>
                <button class="log-link" data-log-location="${escapeText(record.location)}">${escapeText(record.location || "系统输出")}</button>
                <span class="log-summary" title="${escapeText(record.summary)}">${escapeText(record.summary)}</span>
                <span class="log-identities">${record.request_ids.length ? record.request_ids.map(id => `<button class="log-link" title="查看此请求全部日志" data-log-request="${escapeText(id)}">${escapeText(id)}</button>`).join("") : "无请求 ID"}</span>
            </summary>
            <div class="log-record-detail"><div class="log-method">${escapeText(record.method)} · ${count(record.line_count)} 行 · #${record.id}</div><pre>${escapeText(formatLogText(record.text))}</pre></div>
        </details>`).join("") : '<p class="log-empty">暂无匹配日志。可修改请求 ID、关键字或代码位置后重新搜索。</p>';
    renderLogPagination();
}

/** 根据实际页边界显示页码，搜索中禁用翻页以防重复提交。 */
function renderLogPagination() {
    const total = state.logResult?.total_pages ?? 0;
    const page = total ? state.logResult.page : 0;
    const busy = Boolean(state.logController);
    element("log-window-count").textContent = `第 ${count(page)} / ${count(total)} 页 · ${count(state.logResult?.records.length ?? 0)} 条完整记录`;
    element("log-page-number").value = page || 1;
    element("log-page-number").max = Math.max(1, total);
    element("log-page-number").disabled = busy || !total;
    element("log-page-jump").disabled = busy || !total;
    ["log-first", "log-previous"].forEach(id => element(id).disabled = busy || page <= 1);
    ["log-last", "log-next"].forEach(id => element(id).disabled = busy || page >= total);
}

/** 直接请求指定页，保持搜索快照；失败时保留原页内容与页码。 */
function jumpToLogPage(page) {
    const total = state.logResult?.total_pages ?? 0;
    if (state.logController || !total || !Number.isInteger(page)) return;
    const target = Math.max(1, Math.min(total, page));
    if (target === state.logPage) return;
    return refreshLogs(target);
}

/** 最大化日志面板并保留当前内容和操作按钮。 */
function toggleLogMaximized() {
    const dialog = element("log-dialog");
    if (dialog.open) {
        dialog.close();
        return;
    }
    const position = element("log-output").scrollTop;
    element("maximize-log").textContent = "还原";
    element("maximize-log").setAttribute("aria-expanded", "true");
    dialog.append(element("tab-logs"));
    dialog.showModal();
    element("log-output").scrollTop = position;
}

/** 关闭最大化窗口时恢复面板位置与按钮焦点。 */
function restoreLogWindow() {
    const position = element("log-output").scrollTop;
    element("log-home").append(element("tab-logs"));
    const button = element("maximize-log");
    button.textContent = "最大化";
    button.setAttribute("aria-expanded", "false");
    button.focus({ preventScroll: true });
    element("log-output").scrollTop = position;
}

/** 复制当前页日志，兼容不提供剪贴板接口的 HTTP 页面。 */
async function copyLogs() {
    try {
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(
                state.logText,
            );
        } else {
            const field = document.createElement("textarea");
            field.value = state.logText;
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
