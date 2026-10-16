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
