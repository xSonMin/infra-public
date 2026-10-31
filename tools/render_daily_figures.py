#!/usr/bin/env python3
"""Render journal figures from published summaries, without GPU or network access.

Run from the repository root: python tools/render_daily_figures.py
Dependencies used for this export: matplotlib 3.11.2 and Pillow 13.0.0.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "daily" / "figures"
BLUE = "#2563eb"
ORANGE = "#e87924"
TEAL = "#148a84"
INK = "#172b4d"
MUTED = "#5c6b80"
GRID = "#e3e9f1"


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def style():
    families = ("Hiragino Sans GB", "Noto Sans CJK SC", "Microsoft YaHei",
                "WenQuanYi Micro Hei", "Arial Unicode MS", "DejaVu Sans")
    for family in families:
        try:
            font_manager.findfont(font_manager.FontProperties(family=family),
                                  fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams["font.family"] = family
        break
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.labelcolor": MUTED,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save(fig, filename):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    # Do not publish image text fields, EXIF or other inherited metadata.
    with Image.open(path) as image:
        clean = Image.new("RGB", image.size, "white")
        clean.paste(image.convert("RGB"))
        clean.save(path, format="PNG", optimize=True)
    print(path.relative_to(ROOT))


def heading(fig, title, subtitle):
    fig.text(.07, .94, title, fontsize=20, weight="bold", va="top")
    fig.text(.07, .865, subtitle, fontsize=11, color=MUTED, va="top")


def mtp_boundary():
    data = read("daily/data/2026-10-30-mtp-boundary.json")
    workloads = ("short", "long-context", "long-decode")
    labels = ("512 / 128", "2048 / 256", "512 / 512")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(left=.085, right=.97, bottom=.23, top=.74)
    positions = np.arange(3)
    for offset, concurrency, color in [(-.25, 1, BLUE), (0, 8, TEAL), (.25, 32, ORANGE)]:
        rows = [next(row for row in data["rows"] if row["workload"] == workload
                     and row["concurrency"] == concurrency) for workload in workloads]
        values = np.array([row["mean_paired_speedup"] for row in rows])
        lower = values - np.array([row["min_paired_speedup"] for row in rows])
        upper = np.array([row["max_paired_speedup"] for row in rows]) - values
        ax.bar(positions + offset, values, width=.22, color=color,
               label=f"C = {concurrency}", yerr=np.vstack([lower, upper]),
               capsize=3, error_kw={"elinewidth": 1, "ecolor": INK}, zorder=3)
        for x, value, hi in zip(positions + offset, values, upper):
            ax.text(x, value + hi + .065, f"{value:.3f}×", ha="center", fontsize=10)
    ax.axhline(1, color=MUTED, linestyle="--", linewidth=1.1, label="普通解码 = 1")
    ax.set_xticks(positions, labels)
    ax.set_xlabel("输入 / 输出 token", labelpad=10)
    ax.set_ylabel("固定 3 步 MTP / 普通解码")
    ax.set_ylim(0, 2.35)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.20), ncols=4, frameon=False)
    heading(fig, "MTP 加速比随并发变化", "2026-10-30 正式基准 · 四卡 TP=4 · SGLang 0.5.23 · 关闭前缀缓存")
    fig.text(.07, .11, "每点为 3 轮独立配对加速比的均值；误差线表示三轮观测最小值至最大值，并非置信区间。", fontsize=10, color=MUTED)
    fig.text(.07, .06, "来源：REPORT-MTP-BOUNDARY.md；54 个无采样窗口、7,032 个正式请求。", fontsize=10, color=MUTED)
    save(fig, "2026-10-30-mtp-boundary.png")


def single_stream():
    data = read("results/briefing-2026-10-31/single-stream-summary.json")
    workloads = ("short", "long-context", "long-decode")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(left=.085, right=.97, bottom=.23, top=.74)
    positions = np.arange(3)
    for mode, offset, color, label in [("none", -.18, BLUE, "普通解码"),
                                       ("mtp", .18, ORANGE, "固定 3 步 MTP")]:
        rows = [next(row for row in data["rows"] if row["workload"] == workload
                     and row["mode"] == mode) for workload in workloads]
        values = np.array([row["mean_output_tok_s"] for row in rows])
        lower = values - np.array([row["min_output_tok_s"] for row in rows])
        upper = np.array([row["max_output_tok_s"] for row in rows]) - values
        ax.bar(positions + offset, values, width=.30, color=color, label=label,
               yerr=np.vstack([lower, upper]), capsize=4,
               error_kw={"elinewidth": 1.1, "ecolor": INK}, zorder=3)
        for x, value, hi in zip(positions + offset, values, upper):
            ax.text(x, value + hi + 6, f"{value:.2f}", ha="center", fontsize=12, weight="bold")
    ax.set_xticks(positions, ("512 / 128", "2048 / 256", "512 / 512"))
    ax.set_xlabel("输入 / 输出 token", labelpad=10)
    ax.set_ylabel("整体输出速度（token/s，四卡服务合计）")
    ax.set_ylim(0, 260)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.17), ncols=2, frameon=False)
    heading(fig, "单流整体输出速度", "C = 1 · 包含输入处理与首字等待 · 2026-10-30 测量，2026-10-31 复核")
    fig.text(.07, .11, "柱高为三轮均值，误差线为三轮观测范围；不是首 token 后生成速率，也不是每 GPU 速度。", fontsize=10, color=MUTED)
    fig.text(.07, .06, "来源：single-stream-summary.json；18 个窗口、696 个请求。独立提示集回放不并入此图。", fontsize=10, color=MUTED)
    save(fig, "2026-10-31-single-stream.png")


def latency_tradeoffs():
    data = read("daily/data/2026-10-31-latency-tradeoffs.json")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), sharey=True)
    fig.subplots_adjust(left=.235, right=.965, bottom=.24, top=.70, wspace=.22)
    pairs = data["pairs"]
    labels = ("等待预算\n仅同步 → 100ms", "预填充分块\n2048 → 1024", "预算集合\n普通队列 → 加入续块")
    for ax, metric, title, threshold in [(axes[0], "ttft_ms", "平均首字延迟 TTFT（ms，对数刻度）", 500),
                                          (axes[1], "tpot_ms", "平均逐 token 耗时 TPOT（ms）", 30)]:
        for y, pair in enumerate(pairs):
            before, after = pair["baseline"][metric], pair["variant"][metric]
            ax.plot([before, after], [y, y], color="#aab6c8", linewidth=2.2, zorder=2)
            ax.scatter(before, y, s=75, color=BLUE, edgecolor="white", zorder=3,
                       label="对照" if y == 0 else None)
            ax.scatter(after, y, s=80, marker="D", color=ORANGE, edgecolor="white", zorder=4,
                       label="改动后" if y == 0 else None)
            ax.annotate(f"{before:,.2f}", (before, y), xytext=(0, 11),
                        textcoords="offset points", ha="center", fontsize=10, color=BLUE)
            ax.annotate(f"{after:,.2f}", (after, y), xytext=(0, -20),
                        textcoords="offset points", ha="center", fontsize=10, color=ORANGE)
        ax.axvline(threshold, linestyle="--", color=MUTED, linewidth=1.0,
                   label="研究门槛")
        ax.set_title(title, pad=16, fontsize=11)
        ax.set_yticks(np.arange(len(pairs)), labels)
        ax.set_ylim(2.65, -.65)
        ax.grid(axis="x", color=GRID, zorder=0)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=15)
    axes[0].set_xscale("log")
    axes[0].set_xlim(200, 10000)
    axes[0].xaxis.set_major_locator(FixedLocator([200, 500, 1000, 2000, 5000, 10000]))
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axes[1].set_xlim(0, 70)
    axes[1].tick_params(axis="y", labelleft=False)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(.61, .82),
               ncols=3, frameon=False)
    heading(fig, "调度与分块调整：首字延迟和生成耗时的取舍", "2048 输入 / 128 输出 · 周期到达 4 req/s · 各窗口输入 120 秒 · 无 profiler")
    fig.text(.07, .12, "三个独立对照分别解读，均为固定顺序、单轮配对；不能连成一条参数搜索或因果贡献曲线。", fontsize=10, color=MUTED)
    fig.text(.07, .065, "虚线为研究门槛 TTFT 500ms / TPOT 30ms；数值越低越好，门槛尚未确认为业务 SLO。", fontsize=10, color=MUTED)
    save(fig, "2026-10-31-latency-tradeoffs.png")


def main():
    style()
    mtp_boundary()
    single_stream()
    latency_tradeoffs()


if __name__ == "__main__":
    main()
