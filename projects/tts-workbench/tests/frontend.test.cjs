/** Verify log polling and direct playback without requesting a real model. */
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

/** Provide only the DOM surfaces needed by the actual frontend functions. */
function harness() {
    const nodes = new Map();
    const state = {
        logLineLimit: 20000,
        logText: "",
        logsPaused: false,
        logController: null,
        rows: [],
    };
    /** Resolve a stable element so asynchronous responses affect the same view. */
    function element(identifier) {
        if (!nodes.has(identifier)) {
            nodes.set(identifier, {
                value: "",
                textContent: "",
                scrollTop: 0,
                scrollHeight: 1,
                dataset: {},
                classList: { toggle() {} },
            });
        }
        return nodes.get(identifier);
    }
    const context = vm.createContext({
        state,
        element,
        AbortController,
        AbortSignal,
        URLSearchParams,
        Date,
        API: "/api/workbench",
        localStorage: { setItem() {} },
        count: String,
        poll() {},
        notify() {},
        document: {
            querySelectorAll() {
                return [];
            },
        },
    });
    for (const file of ["views.js", "controls.js", "requests.js"]) {
        vm.runInContext(
            fs.readFileSync(
                path.join(__dirname, "../tts_workbench/static", file),
                "utf8",
            ),
            context,
        );
    }
    return { state, element, context };
}

test("pause stops new log fetches and ignores an already pending response", async () => {
    const { state, element, context } = harness();
    let calls = 0,
        resolve;
    context.api = () => {
        calls += 1;
        return new Promise((done) => {
            resolve = done;
        });
    };
    state.logText = "frozen\n";
    element("log-output").textContent = "frozen\n";
    const pending = context.refreshLogs();
    context.toggleLogFetching();
    await context.refreshLogs();
    assert.equal(calls, 1);
    resolve({
        source: "server.log",
        cursor: "next",
        text: "must not appear\n",
        reset: true,
    });
    await pending;
    assert.equal(element("log-output").textContent, "frozen\n");
    assert.equal(state.logText, "frozen\n");
    assert.equal(element("pause-log").textContent, "恢复获取");
    context.toggleLogFetching();
    assert.equal(state.logsPaused, false);
    assert.equal(state.logCursor, null);
});

test("log window counts exactly 20000 rows and can be reduced while paused", () => {
    const { state, element, context } = harness();
    state.logText = Array.from(
        { length: 25000 },
        (_, index) => `row ${index}\n`,
    ).join("");
    context.trimLogWindow();
    assert.equal(state.logText.split("\n").length - 1, 20000);
    state.logsPaused = true;
    element("log-limit").value = "12";
    context.updateLogLimit();
    assert.equal(state.logText.split("\n").length - 1, 12);
    assert.match(state.logText, /^row 24988\n/);
});

test("listen plays the result immediately without opening details", () => {
    const { state, element, context } = harness();
    state.rows = [
        {
            id: "sample",
            audio_path: "saved.wav",
            sequence: 1,
            speaker: "角色",
            text: "Hello",
        },
    ];
    let played = false;
    element("result-audio").play = () => {
        played = true;
        return Promise.resolve();
    };
    context.playResult("sample");
    assert.equal(played, true);
    assert.equal(element("result-audio").src, "/api/workbench/audio/sample");
    assert.equal(element("result-player").hidden, false);
});

test("keyword matching keeps the whole multiline log record", () => {
    const { state, element, context } = harness();
    const first =
        "[02:00:47] INFO\n  request_id=speech-a\n  model.py:315\n    方法：forward()\n      内容：attention\n\n";
    const second =
        "[02:00:48] INFO\n  request_id=speech-b\n  other.py:8\n    内容：other\n\n";
    state.logText = first + second;
    element("log-filter").value = "speech-a";
    context.renderLogs();
    assert.equal(element("log-output").textContent, first);
    state.logRequestId = "speech-a";
    state.logLineLimit = 7;
    context.trimLogWindow();
    assert.equal(state.logText, second);
    assert.equal(state.logTruncated, true);
});

test("changing request filter while paused never fetches and aborts stale responses", () => {
    const { state, element, context } = harness();
    state.logsPaused = true;
    state.logController = new AbortController();
    let calls = 0;
    context.poll = () => {
        calls += 1;
    };
    element("log-request-id").value = " speech-a ";
    context.applyLogRequestFilter();
    assert.equal(state.logRequestId, "speech-a");
    assert.equal(state.logController.signal.aborted, true);
    assert.equal(calls, 0);
    assert.equal(
        element("download-request-log").href,
        "/api/workbench/logs/download?request_id=speech-a",
    );
});

test("automatic names stay previews while edits survive a pending name fetch", async () => {
    const { state, element, context } = harness();
    let resolve;
    context.api = () =>
        new Promise((done) => {
            resolve = done;
        });
    const pending = context.generateRunName();
    element("run-name").value = "手动任务名称";
    resolve({ name: "model-001" });
    await pending;
    assert.equal(element("run-name").value, "手动任务名称");
    element("run-name").value = "";
    context.api = async () => ({ name: "model-002" });
    await context.generateRunName();
    assert.equal(element("run-name").value, "model-002");
    assert.equal(state.generatedRunName, "model-002");
});
