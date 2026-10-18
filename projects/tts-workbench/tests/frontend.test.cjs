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
        logPage: 1,
        logOrder: "desc",
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

test("两种抽样方式均保留时长，表单按百分比提交且包含全部角色", async () => {
    const { state, element, context } = harness();
    state.catalog = { lines: [
        { line_id: "a", unit_id: "chosen", speaker_id: "one" },
        { line_id: "b", unit_id: "chosen", speaker_id: "two" },
        { line_id: "c", unit_id: "other", speaker_id: "three" },
    ] };
    context.document.querySelectorAll = () => [{ value: "chosen" }];
    element("qpm").value = "10";
    element("duration-preset").value = "600";
    element("timeout").value = "120";
    element("repeat-probability").value = "35";
    state.sampleOrder = "sequential";
    context.updateEstimate();
    assert.equal(element("random-options").hidden, true);
    assert.equal(element("repeat-probability").disabled, true);
    assert.equal(element("selected-count").textContent, "2 条台词");
    assert.match(element("estimate").textContent, /100 次请求/);
    state.sampleOrder = "random";
    context.updateEstimate();
    assert.equal(element("random-options").hidden, false);
    assert.equal(element("repeat-probability").disabled, false);
    assert.match(element("sampling-note").textContent, /允许重复/);
    let submitted;
    context.api = async (path, options) => {
        if (path === "/runs") {
            submitted = JSON.parse(options.body);
            return { id: "new-task" };
        }
        return { name: "model-002" };
    };
    await context.startRun({ preventDefault() {} });
    assert.equal(submitted.mode, "duration");
    assert.equal(submitted.sample_order, "random");
    assert.equal(submitted.repeat_probability, 0.35);
    assert.equal(submitted.duration_seconds, 600);
    assert.deepEqual(submitted.unit_ids, ["chosen"]);
    assert.equal("speaker_ids" in submitted, false);
    assert.equal("line_ids" in submitted, false);
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
        return {records:[{id:7,position:1,time:"12:00:00",level:"INFO",location:"model.py:315",method:"forward()",summary:"<script>attention</script>",request_ids:["speech-a"],line_count:6,text:'[12:00:00] INFO\n  {"value":12}\n'}],snapshot:7,page:1,total_pages:1,next_before:null,matched_records:1,matched_lines:6,retained_lines:6};
    };
    context.applyLogRequestFilter();
    await new Promise(resolve => setImmediate(resolve));
    assert.match(query,/request_id=speech-a/);
    assert.match(query,/keyword=attention/);
    assert.match(query,/location=model.py%3A315/);
    assert.match(element("log-output").innerHTML, /<details class="log-record">/);
    assert.match(element("log-output").innerHTML, /class="log-position"[^>]*>1\/1<\/span>/);
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
    const page={records:[],snapshot:22,page:1,total_pages:0,next_before:null,matched_records:0,matched_lines:0,retained_lines:100};
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

test("jump goes directly to a server page and repeated clicks cannot advance while loading", async () => {
    const {state, element, context} = harness();
    state.logResult = {records:[], page:1, total_pages:36};
    state.logSnapshot = 3508;
    state.logRequestId = "speech-pages";
    const queries = [];
    let resolve;
    context.api = path => {
        queries.push(path);
        return new Promise(done => { resolve = done; });
    };
    const pending = context.jumpToLogPage(35);
    context.jumpToLogPage(36);
    assert.equal(queries.length, 1);
    assert.equal(state.logPage, 1);
    assert.equal(element("log-next").disabled, true);
    assert.match(queries[0], /page=35/);
    assert.match(queries[0], /snapshot=3508/);
    assert.match(queries[0], /request_id=speech-pages/);
    resolve({records:[], page:35, total_pages:36, snapshot:3508});
    await pending;
    assert.equal(state.logPage, 35);
    assert.match(element("log-window-count").textContent, /35 \/ 36/);
    assert.equal(element("log-page-number").max, 36);
    context.api = async () => { throw new Error("网络错误"); };
    await context.jumpToLogPage(36);
    assert.equal(state.logPage, 35);
    assert.equal(element("log-page-number").value, 35);
    assert.equal(element("log-next").disabled, false);
});

test("changing time order starts a new filtered search from the first page", async () => {
    const {state, element, context} = harness();
    state.logPage = 35;
    state.logSnapshot = 3508;
    element("log-order").value = "asc";
    element("log-request-id").value = "speech-pages";
    element("log-filter").value = "注意力";
    element("log-location").value = "model.py";
    let query;
    context.api = async path => {
        query = path;
        return {records:[],page:1,total_pages:36,snapshot:3600};
    };
    context.applyLogRequestFilter();
    await new Promise(resolve => setImmediate(resolve));
    assert.match(query, /page=1/);
    assert.match(query, /order=asc/);
    assert.match(query, /request_id=speech-pages/);
    assert.match(query, /location=model.py/);
    assert.doesNotMatch(query, /snapshot=/);
    assert.equal(state.logPage, 1);
    assert.equal(state.logSnapshot, 3600);
});
