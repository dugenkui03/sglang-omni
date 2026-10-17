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
        logLineLimit: 2000,
        logText: "",
        logsPaused: false,
        logController: null,
        logLoading: false,
        logError: "",
        logMatchedLines: null,
        logResult: null,
        logSnapshot: null,
        logBefore: null,
        logPages: [],
        logRequestId: "",
        logScope: "all",
        logKeyword: "",
        logLocation: "",
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
    for (const file of ["charts.js", "views.js", "controls.js", "requests.js", "log_search.js"]) {
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


test("search combines all filters and renders collapsed escaped records without scrolling", async () => {
    const {state, element, context} = harness();
    element("log-request-id").value = " speech-a ";
    element("log-filter").value = "attention";
    element("log-location").value = "model.py:315";
    element("log-output").scrollTop = 45;
    let query;
    context.api = async path => {
        query = path;
        return {records:[{id:7,time:"12:00:00",level:"INFO",location:"model.py:315",method:"forward()",summary:"<script>attention</script>",request_ids:["speech-a"],line_count:6,text:'[12:00:00] INFO\n  {"value":12}\n'}],snapshot:7,next_before:null,matched_records:1,matched_lines:6,retained_lines:6};
    };
    context.applyLogRequestFilter();
    await new Promise(resolve => setImmediate(resolve));
    assert.match(query,/request_id=speech-a/);
    assert.match(query,/keyword=attention/);
    assert.match(query,/location=model.py%3A315/);
    assert.match(element("log-output").innerHTML, /<details class="log-record">/);
    assert.doesNotMatch(element("log-output").innerHTML, /<script>|<details[^>]+open/);
    assert.match(element("log-output").innerHTML,/data-log-request="speech-a"/);
    assert.equal(element("log-output").scrollTop,45);
    assert.equal(state.logSnapshot,7);
    assert.match(context.formatLogText('{"value":12}'), /\n  "value": 12\n/);
});

test("new search aborts stale response and keeps the latest result", async () => {
    const {state, element, context} = harness();
    const resolvers=[];
    context.api = () => new Promise(resolve => resolvers.push(resolve));
    const first=context.refreshLogs();
    const second=context.refreshLogs();
    const page={records:[],snapshot:22,next_before:null,matched_records:0,matched_lines:0,retained_lines:100};
    resolvers[1](page);
    await second;
    resolvers[0]({...page,snapshot:11});
    await first;
    assert.equal(state.logSnapshot,22);
    assert.equal(state.logController,null);
    assert.match(element("log-status").textContent,/不自动刷新/);
});

test("log line budget is bounded and a search failure does not schedule retries", async () => {
    const {state,element,context}=harness();
    element("log-limit").value="40000";
    context.updateLogLimit();
    assert.equal(state.logLineLimit,30000);
    element("log-limit").value="12";
    context.updateLogLimit();
    assert.equal(state.logLineLimit,12);
    let calls=0;
    context.api=async () => {calls++;throw new Error("deadline");};
    await context.refreshLogs();
    assert.equal(calls,1);
    assert.match(element("log-status").textContent,/搜索失败/);
});
