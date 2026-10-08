"""Draw the README charts from bench/results, in a light and a dark version.

Usage: python bench/plot.py
"""

import json
import sys
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summary  # noqa: E402

ROOT = Path(__file__).parent
RESULTS, CHARTS = ROOT / "results", ROOT / "charts"
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e8e7e3", "ref": "#9a9893",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#2f2e2b", "ref": "#77756f",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500"]},
}
NAMES = {"GSM8K_Llama-3-8B-Instruct": "GSM8K, Llama-3 8B", "GSM8K_Llama-3-70B-Instruct": "GSM8K, Llama-3 70B",
         "MATH_Llama-3-8B-Instruct": "MATH, Llama-3 8B", "MATH_Llama-3-70B-Instruct": "MATH, Llama-3 70B",
         "CodeContests_Llama-3-8B-Instruct": "CodeContests, Llama-3 8B",
         "CodeContests_Llama-3-70B-Instruct": "CodeContests, Llama-3 70B"}


def use_geist() -> None:
    fonts = ROOT / "data" / "fonts"
    for f in fonts.glob("Geist-*.ttf"):
        font_manager.fontManager.addfont(str(f))
    if any(fonts.glob("Geist-*.ttf")):
        plt.rcParams["font.family"] = "Geist"


def style(ax, t) -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    ax.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def line(ax, xs, ys, color, label, t, width=1.6, marker=None, dashed=False, end_label=False):
    ax.plot(xs, ys, color=color, linewidth=width, marker=marker, markersize=6 if marker else 0,
            markeredgecolor=t["surface"], markeredgewidth=1.5, linestyle=(0, (4, 3)) if dashed else "-",
            label=label, solid_capstyle="round", zorder=4 if marker else 3)
    if end_label:
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=t["text"])


def panel(ax, title, series, t, xlabel, zoom=False) -> None:
    style(ax, t)
    for xs, ys, kwargs in series:
        line(ax, xs, ys, t=t, **kwargs)
    ax.set_xscale("log")
    ax.set_title(title, loc="left", fontsize=11, color=t["text"], fontweight="medium", pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=t["muted"])
    ax.set_ylabel("Share right" + (" (zoomed)" if zoom else ""), fontsize=9, color=t["muted"])
    if zoom:
        ys = [y for xs, yv, kw in series if not kw.get("dashed") for y in yv]
        low, high = min(ys), max(ys)
        pad = max(0.01, (high - low) * 0.12)
        ax.set_ylim(low - pad, high + pad)
    else:
        ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.margins(x=0.02)


def answer_series(c, t):
    s = t["series"]
    out = []
    out.append(([x for x, _ in c["majority"]], [y for _, y in c["majority"]],
                {"color": s[2], "label": "Majority vote"}))
    out.append(([x for x, _ in c["esc"]], [y for _, y in c["esc"]], {"color": s[3], "label": "Early-stopping SC"}))
    out.append(([x for x, _ in c["ac"]], [y for _, y in c["ac"]], {"color": s[1], "label": "Adaptive-Consistency"}))
    out.append(([n["used"] for n in c["needle"]], [n["right"] for n in c["needle"]],
                {"color": s[0], "label": "needle", "width": 2.6, "marker": "o", "end_label": True}))
    return out


def code_series(c, t):
    s = t["series"]
    pass1 = c["coverage"][0][1]
    xs = [x for x, _ in c["coverage"]]
    return [
        (xs, [y for _, y in c["coverage"]], {"color": t["ref"], "label": "Right program in the pile", "dashed": True,
                                             "width": 1.2}),
        (xs, [pass1] * len(xs), {"color": s[3], "label": "One random sample"}),
        ([x for x, _ in c["filter"]], [y for _, y in c["filter"]], {"color": s[2], "label": "Public-test filter"}),
        ([x for x, _ in c["cluster"]], [y for _, y in c["cluster"]], {"color": s[1], "label": "Filter + cluster"}),
        ([n["used"] for n in c["needle"]], [n["right"] for n in c["needle"]],
         {"color": s[0], "label": "needle", "width": 2.6, "marker": "o", "end_label": True}),
    ]


def ladder_series(path, t):
    res = json.load(open(path))["problems"]
    s = t["series"]
    out = []
    for key, color, label in (("ac_small", s[1], "Adaptive-Consistency, 8B"), ("ac_large", s[2], "Adaptive-Consistency, 70B")):
        caps = list(res[0][key])
        out.append(([mean(r[key][c][0] for r in res) for c in caps], [mean(r[key][c][1] for r in res) for c in caps],
                    {"color": color, "label": label}))
    scales = list(res[0]["needle"])
    out.append(([mean(r["needle"][k][0] for r in res) for k in scales], [mean(r["needle"][k][1] for r in res) for k in scales],
                {"color": s[0], "label": "needle, 8B then 70B", "width": 2.6, "marker": "o", "end_label": True}))
    return out


def figure(rows, cols, width, height, t):
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False)
    fig.patch.set_facecolor(t["surface"])
    return fig, axes.ravel()


def save(fig, name, theme, t) -> None:
    CHARTS.mkdir(exist_ok=True)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9,
               labelcolor=t["text"], bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    suffix = "" if theme == "light" else "-dark"
    fig.savefig(CHARTS / f"{name}{suffix}.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    use_geist()
    for theme, t in THEMES.items():
        answer = [RESULTS / f"{c}.json" for c in NAMES if c.startswith(("GSM8K", "MATH"))]
        if all(p.exists() for p in answer):
            fig, axes = figure(2, 2, 11, 8.2, t)
            for ax, path in zip(axes, answer):
                c = summary.curves(path)
                panel(ax, NAMES[c["config"]], answer_series(c, t), t, "Samples per problem (mean, log scale)", zoom=True)
            save(fig, "answers", theme, t)
        code = [RESULTS / f"{c}.json" for c in NAMES if c.startswith("Code")]
        if all(p.exists() for p in code):
            fig, axes = figure(1, 2, 11, 4.3, t)
            for ax, path in zip(axes, code):
                c = summary.curves(path)
                panel(ax, NAMES[c["config"]], code_series(c, t), t, "Samples per problem (mean, log scale)")
                ax.set_ylim(0, max(0.5, max(y for _, y in c["coverage"]) * 1.1))
            save(fig, "code", theme, t)
        ladder = [RESULTS / f"ladder_{d}.json" for d in ("GSM8K", "MATH")]
        if all(p.exists() for p in ladder):
            fig, axes = figure(1, 2, 11, 4.3, t)
            for ax, path, title in zip(axes, ladder, ("GSM8K, two models", "MATH, two models")):
                panel(ax, title, ladder_series(path, t), t, "Cost per problem, in 8B samples (log scale)", zoom=True)
            save(fig, "ladder", theme, t)


if __name__ == "__main__":
    main()
