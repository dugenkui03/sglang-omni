/** Configure real workloads using the portable dialogue catalog. */
"use strict";

/** Fill a fresh editable name using the browser's local time, never during polling. */
function generateRunName() {
    const now = new Date();
    const parts = [
        now.getMonth() + 1,
        now.getDate(),
        now.getHours(),
        now.getMinutes(),
        now.getSeconds(),
    ].map((value) => String(value).padStart(2, "0"));
    element("run-name").value =
        `Qwen3-TTS 中→英 ${now.getFullYear()}-${parts[0]}-${parts[1]} ${parts[2]}:${parts[3]}:${parts[4]}`;
}

/** Return source-ordered lines matching the current unit and role filters. */
function selectedLines() {
    const units = [...document.querySelectorAll(".unit-check:checked")].map(
        (input) => input.value,
    );
    const speaker = element("speaker").value;
    return (state.catalog?.lines ?? []).filter(
        (line) =>
            units.includes(line.unit_id) &&
            (!speaker || line.speaker_id === speaker),
    );
}

/** Rebuild role choices after the selected units change. */
function updateSpeakers() {
    const previous = element("speaker").value;
    const units = [...document.querySelectorAll(".unit-check:checked")].map(
        (input) => input.value,
    );
    const speakers = new Map();
    state.catalog.lines
        .filter((line) => units.includes(line.unit_id))
        .forEach((line) =>
            speakers.set(line.speaker_id, `${line.speaker} · ${line.unit_id}`),
        );
    element("speaker").innerHTML =
        '<option value="">全部角色</option>' +
        [...speakers]
            .map(
                ([identifier, label]) =>
                    `<option value="${escapeText(identifier)}">${escapeText(label)}</option>`,
            )
            .join("");
    if (speakers.has(previous)) {
        element("speaker").value = previous;
    }
    updateLines();
}

/** Refresh selectable reference lines without changing their source order. */
function updateLines() {
    const lines = selectedLines(),
        previous = element("line").value;
    element("line").innerHTML = lines
        .map(
            (line) =>
                `<option value="${escapeText(line.line_id)}">${escapeText(line.unit_id)} · ${line.line_index + 1} · ${escapeText(line.speaker)}：${escapeText(line.text_zh)}</option>`,
        )
        .join("");
    if (lines.some((line) => line.line_id === previous)) {
        element("line").value = previous;
    }
    updateReference();
    updateEstimate();
}

/** Preview the actual Chinese reference and language-specific target text. */
function updateReference() {
    const line = state.catalog?.lines.find(
        (row) => row.line_id === element("line").value,
    );
    const audio = element("reference-audio");
    if (line) {
        element("reference-text").innerHTML =
            `<div>${escapeText(line.speaker)}：${escapeText(line.text_zh)}</div><div class="english">${escapeText(line.text_en_tts)}</div>${line.reference_text_needs_review ? '<span class="warning-text">此参考片段曾扩窗，参考转写需要复核。</span>' : ""}`;
        if (audio.dataset.line !== line.line_id) {
            audio.src = `${API}/references/${encodeURIComponent(line.line_id)}`;
            audio.dataset.line = line.line_id;
        }
    } else {
        element("reference-text").textContent = "未选择素材";
        audio.removeAttribute("src");
        delete audio.dataset.line;
    }
    updateEstimate();
}

/** Convert the chosen duration to the scheduler's seconds contract. */
function chosenSeconds() {
    return element("duration-preset").value === "custom"
        ? Number(element("duration-value").value) *
              Number(element("duration-unit").value)
        : Number(element("duration-preset").value);
}

/** Explain the sending cadence and selected coverage before a task starts. */
function updateEstimate() {
    const lines = selectedLines(),
        qpm = Number(element("qpm").value),
        seconds = chosenSeconds();
    const effectiveLines =
        state.mode === "single"
            ? lines.filter((line) => line.line_id === element("line").value)
            : lines;
    const lineCount = effectiveLines.length;
    element("selected-count").textContent = `${lineCount} 条台词`;
    element("review-count").textContent = effectiveLines.some(
        (line) => line.reference_text_needs_review,
    )
        ? `${effectiveLines.filter((line) => line.reference_text_needs_review).length} 条参考待复核`
        : "";
    element("duration-fields").hidden = state.mode !== "duration";
    element("custom-duration").hidden =
        element("duration-preset").value !== "custom";
    const expected =
        state.mode === "duration" ? Math.ceil((seconds * qpm) / 60) : lineCount;
    element("estimate").textContent =
        qpm > 0
            ? `每 ${(60 / qpm).toFixed(qpm > 60 ? 2 : 1)} 秒发起一次 · ${state.mode === "duration" ? "计划约" : "共"} ${count(expected)} 次请求`
            : "请设置大于 0 的 QPM";
    element("start").disabled =
        Boolean(state.activeId) || !lineCount || state.submitting;
}

/** Persist a new task before switching the UI to its recorded identity. */
async function startRun(event) {
    event.preventDefault();
    if (!selectedLines().length) {
        notify("请至少选择一个有台词的配音单元。");
        return;
    }
    state.submitting = true;
    updateEstimate();
    try {
        if (!element("run-name").value.trim()) {
            generateRunName();
        }
        const payload = {
            name: element("run-name").value,
            mode: state.mode,
            unit_ids: [...document.querySelectorAll(".unit-check:checked")].map(
                (input) => input.value,
            ),
            speaker_ids: element("speaker").value
                ? [element("speaker").value]
                : [],
            line_ids: state.mode === "single" ? [element("line").value] : [],
            requests_per_minute: Number(element("qpm").value),
            duration_seconds: chosenSeconds(),
            request_timeout_seconds: Number(element("timeout").value),
        };
        const run = await api("/runs", {
            method: "POST",
            body: JSON.stringify(payload),
        });
        state.selectedRun = run.id;
        state.activeId = run.id;
        state.offset = 0;
        localStorage.setItem("workbench-run", run.id);
        element("notice").hidden = true;
        generateRunName();
        await poll();
    } catch (error) {
        notify(`无法开始测试：${error.message}`);
    } finally {
        state.submitting = false;
        updateEstimate();
    }
}

/** Stop sending new requests while retaining the current in-flight results. */
async function stopRun() {
    if (!state.activeId) {
        return;
    }
    try {
        await api(`/runs/${state.activeId}/stop`, { method: "POST" });
        notify("已停止发起新请求，正在等待已有请求结束。", true);
        await poll();
    } catch (error) {
        notify(`停止失败：${error.message}`);
    }
}

/** Clear terminal results directly; the server protects running tasks and inputs. */
async function clearRuns(all) {
    if (!all && !state.selectedRun) {
        return;
    }
    try {
        const result = await api(
            `/runs/${all ? "completed" : state.selectedRun}`,
            { method: "DELETE" },
        );
        state.selectedRun = null;
        state.detail = null;
        state.offset = 0;
        localStorage.removeItem("workbench-run");
        notify(`已清理 ${result.cleared} 个已结束任务的记录与抽样音频。`, true);
        await poll();
    } catch (error) {
        notify(`清理失败：${error.message}`);
    }
}
