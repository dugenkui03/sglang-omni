/** Render small, dependency-free charts from measured observations only. */
"use strict";

/** Escape external text before it enters a generated HTML or SVG fragment. */
function escapeText(value) {
    return String(value ?? "").replace(
        /[&<>"']/g,
        (character) =>
            ({
                "&": "&amp;",
                "<": "&lt;",
                ">": "&gt;",
                '"': "&quot;",
                "'": "&#39;",
            })[character],
    );
}

/** Format a Unix timestamp in the browser's local timezone. */
function shortTime(timestamp) {
    return new Date(timestamp * 1000).toLocaleTimeString("zh-CN", {
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
    });
}

/** Draw bounded series with axis units and a truthful empty state. */
function drawChart(
    element,
    series,
    unit,
    fixedMaximum = null,
    timeRange = null,
) {
    const points = series
        .flatMap((line) => line.points)
        .filter((point) => Number.isFinite(point.value));
    if (!points.length) {
        element.innerHTML =
            '<div class="chart-empty">等待运行数据<small>测试开始后自动绘制趋势</small></div>';
        return;
    }
    const width = 410,
        height = 180,
        left = 40,
        right = 9,
        top = 15,
        bottom = 25;
    const minimumTime =
        timeRange?.[0] ?? Math.min(...points.map((point) => point.time));
    const maximumTime =
        timeRange?.[1] ??
        Math.max(minimumTime + 1, ...points.map((point) => point.time));
    const maximumValue =
        fixedMaximum ??
        Math.max(1, ...points.map((point) => point.value)) * 1.15;
    /** Map an observed time to the chart's horizontal plotting range. */
    function horizontal(time) {
        return (
            left +
            ((time - minimumTime) / (maximumTime - minimumTime)) *
                (width - left - right)
        );
    }
    /** Map a measured value to the chart's vertical plotting range. */
    function vertical(value) {
        return (
            height - bottom - (value / maximumValue) * (height - bottom - top)
        );
    }
    let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeText(unit)} 趋势图">`;
    for (let tick = 0; tick <= 4; tick++) {
        const value = (maximumValue * tick) / 4,
            y = vertical(value);
        const label = value >= 10 ? value.toFixed(0) : value.toFixed(1);
        svg += `<line x1="${left}" x2="${width - right}" y1="${y}" y2="${y}" stroke="#edf1f7" stroke-dasharray="3 3"/><text x="${left - 9}" y="${y + 3}" fill="#acb7c8" text-anchor="end" font-size="9">${label}</text>`;
    }
    series.forEach((line, index) => {
        const valid = line.points.filter((point) =>
            Number.isFinite(point.value),
        );
        if (!valid.length) {
            return;
        }
        const coordinates = valid.map(
            (point) => `${horizontal(point.time)},${vertical(point.value)}`,
        );
        if (index === 0 && valid.length > 1) {
            svg += `<polygon points="${horizontal(valid[0].time)},${height - bottom} ${coordinates.join(" ")} ${horizontal(valid.at(-1).time)},${height - bottom}" fill="${line.color}" opacity=".06"/>`;
        }
        svg += `<polyline points="${coordinates.join(" ")}" fill="none" stroke="${line.color}" stroke-width="2" stroke-linejoin="round"/>`;
        valid.forEach((point) => {
            svg += `<circle cx="${horizontal(point.time)}" cy="${vertical(point.value)}" r="${valid.length === 1 ? 3 : 2}" fill="${line.color}"><title>${escapeText(line.label)} · ${shortTime(point.time)} · ${point.value.toFixed(2)} ${escapeText(unit)}</title></circle>`;
        });
    });
    svg += `<text x="${left}" y="${height - 3}" fill="#a8b5c8" font-size="9">${shortTime(minimumTime)}</text><text x="${width - right}" y="${height - 3}" text-anchor="end" fill="#a8b5c8" font-size="9">${shortTime(maximumTime)}</text><text x="${left}" y="9" fill="#a8b5c8" font-size="8">${escapeText(unit)}</text></svg>`;
    element.innerHTML = svg;
}
