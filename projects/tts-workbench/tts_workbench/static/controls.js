/** Configure real workloads using the portable dialogue catalog. */
"use strict";

/** Collapse the configuration rail without replacing any edited form values. */
function setConfigurationCollapsed(collapsed) {
    element("workspace").classList.toggle("configuration-collapsed", collapsed);
    element("run-form").hidden = collapsed;
    const button = element("toggle-config");
    button.textContent = collapsed ? "展开" : "收起";
    button.setAttribute("aria-expanded", String(!collapsed));
    button.setAttribute(
        "aria-label",
        collapsed ? "展开配置测试" : "收起配置测试",
    );
    localStorage.setItem("workbench-config-collapsed", String(collapsed));
}

/** Preview the server sequence without overwriting a name edited during the fetch. */
async function generateRunName() {
    const previous = element("run-name").value;
    try {
        const result = await api("/task-name");
        if (element("run-name").value === previous) {
            element("run-name").value = result.name;
            state.generatedRunName = result.name;
        }
    } catch (error) {
        notify(`自动名称暂未获取，创建时由服务器分配：${error.message}`);
    }
}

/** 按原始顺序返回所选配音单元的全部角色台词。 */
function selectedLines() {
    const units = [...document.querySelectorAll(".unit-check:checked")].map(
        (input) => input.value,
    );
    return (state.catalog?.lines ?? []).filter(
        (line) => units.includes(line.unit_id),
    );
}

/** Convert the chosen duration to the scheduler's seconds contract. */
function chosenSeconds() {
    return element("duration-preset").value === "custom"
        ? Number(element("duration-value").value) *
              Number(element("duration-unit").value)
        : Number(element("duration-preset").value);
}

/** 展示所选素材数量、抽样规则和按 QPM 与时长计算的请求数。 */
function updateEstimate() {
    const lines = selectedLines(),
        qpm = Number(element("qpm").value),
        seconds = chosenSeconds();
    const lineCount = lines.length;
    element("selected-count").textContent = `${lineCount} 条台词`;
    element("review-count").textContent = lines.some(
        (line) => line.reference_text_needs_review,
    )
        ? `${lines.filter((line) => line.reference_text_needs_review).length} 条参考待复核`
        : "";
    const random = state.sampleOrder === "random";
    element("random-options").hidden = !random;
    element("repeat-probability").disabled = !random;
    element("sampling-note").textContent = random
        ? "从所选素材中逐次随机抽取，允许重复，直到持续时长结束。"
        : "从所选素材第一句开始，按原顺序循环，直到持续时长结束。";
    element("custom-duration").hidden =
        element("duration-preset").value !== "custom";
    const expected = Math.ceil((seconds * qpm) / 60);
    element("estimate").textContent =
        qpm > 0
            ? `每 ${(60 / qpm).toFixed(qpm > 60 ? 2 : 1)} 秒发起一次 · 计划约 ${count(expected)} 次请求`
            : "请设置大于 0 的 QPM";
    element("start").disabled =
        Boolean(state.activeId) || !lineCount || state.submitting;
}

/** 保存任务的抽样方式与概率，再切换到服务器分配的任务记录。 */
async function startRun(event) {
    event.preventDefault();
    if (!selectedLines().length) {
        notify("请至少选择一个有台词的配音单元。");
        return;
    }
    state.submitting = true;
    updateEstimate();
    try {
        const payload = {
            name:
                element("run-name").value === state.generatedRunName
                    ? ""
                    : element("run-name").value,
            mode: "duration",
            sample_order: state.sampleOrder,
            repeat_probability: state.sampleOrder === "random"
                ? Number(element("repeat-probability").value) / 100
                : 0.2,
            unit_ids: [...document.querySelectorAll(".unit-check:checked")].map(
                (input) => input.value,
            ),
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
        element("run-name").value = "";
        await generateRunName();
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
        state.requestRunId = "";
        state.detail = null;
        state.offset = 0;
        localStorage.removeItem("workbench-run");
        notify(`已清理 ${result.cleared} 个已结束任务的记录与抽样音频。`, true);
        await poll();
    } catch (error) {
        notify(`清理失败：${error.message}`);
    }
}
