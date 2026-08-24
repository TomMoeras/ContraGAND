#!/usr/bin/env python3
"""Generate report figures for report.md.

Produces, in figures/report/:
  fig_overview.png          - pipeline schematic (GAND -> contrastive aug -> ICL bench -> Qwen FT)
  fig_tasks.png             - closed-vs-open task input/output contrast (one shared sentence)
  fig_closed_open_gap.png   - per-model closed vs open accuracy + gap (ICL zs, all 7 models)
  fig_ft_summary.png        - Qwen3.5-4B accuracy bars: ICL-zs vs FT conditions (closed + open)
  fig_synthetic_vs_authentic.png - DAUTH vs DA per-class recall (the RQ4 isolation)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "report"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
})

# Colorblind-friendly palette
C_AMB = "#888888"
C_MASC = "#1f77b4"
C_FEM = "#d62728"
C_BASE = "#aaaaaa"
C_FT_DAUTH = "#cccccc"
C_FT_DA = "#2ca02c"
C_FT_DMULTI = "#17becf"

# ---------------------------------------------------------------------------
# Figure 1 - Pipeline schematic
# ---------------------------------------------------------------------------
def fig_overview():
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, color="#e7eef7", edge="#3b6aa1", fontsize=8, weight="normal"):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                        linewidth=1.2, edgecolor=edge, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, weight=weight)

    def arrow(x1, y1, x2, y2, label=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.4))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 1.5, label, ha="center", va="bottom",
                    fontsize=8, color="#444", style="italic")

    # Stage 1: GAND
    box(1, 76, 22, 16,
        "GAND\n5,047 EN sentences\n+ referent\n(all gender-ambiguous)",
        color="#fff3e0", edge="#cc7a00", weight="bold", fontsize=9)

    # Stage 2: Contrastive augmentation
    box(27, 76, 28, 16,
        "Step 1: contrastive augmentation\nClaude Opus 4.6 + neurosymbolic\n9 strategies, 5-criterion eval\n5-round retry loop\n→ masc + fem variants per row",
        color="#e7f4e7", edge="#3a7a3a", fontsize=8.5)

    # Stage 3: Splits
    box(59, 76, 22, 16,
        "ContraGAND splits\ntrain 11,706\nval 1,455\ntest 1,395 (human-reviewed)",
        color="#fff8e6", edge="#b8a500", weight="bold", fontsize=9)

    arrow(23, 84, 27, 84)
    arrow(55, 84, 59, 84)

    # Stage 4: ICL benchmark
    box(1, 36, 38, 28,
        "Step 2a: ICL benchmark\n7 open-weight models (thinking on)\n\n  4 medium: Qwen3.6-27B, Gemma-4-31B-it,\n            gpt-oss-20b, Magistral-Small-2506\n  3 small:  Gemma-4-E4B-it, Qwen3.5-4B,\n            Ministral-3-3B-Instruct-2512\n\n× zero-shot / per-row sampled few-shot\n× closed task + open task\n+ ambonly ablation (4 medium models)",
        color="#f0e7f6", edge="#7a3a99", fontsize=8)

    # Stage 5: FT
    box(43, 36, 38, 28,
        "Step 2b: Qwen3.5-4B fine-tuning\nQLoRA / plain LoRA via axolotl\n\n× closed task: D-AUTH, D-A\n× open task:   D-AUTH, D-A, D-MULTI\n\n  D-AUTH  = authentic ambiguous-only\n  D-A     = full 2/2/2 contrastive\n  D-MULTI = Gemma-31B teacher distillation",
        color="#fde7e7", edge="#a52a2a", fontsize=8)

    arrow(70, 76, 25, 64, label="ICL prompts")
    arrow(70, 76, 62, 64, label="FT data")

    # Stage 6: Eval
    box(85, 36, 14, 28,
        "Evaluation\non test (1,395)\n\nF1, P, R, Acc\nper class\n+ per-strategy\n+ confusion",
        color="#e6f0fa", edge="#3b6aa1", weight="bold", fontsize=8.5)
    arrow(39, 50, 43, 50)
    arrow(81, 50, 85, 50)

    # RQ block at bottom
    ax.text(50, 28, "Research Questions",
            ha="center", fontsize=12, weight="bold", color="#222")
    rqs = [
        ("RQ1", "Closed: predict ambiguity\ngiven referent?"),
        ("RQ2", "Open: find referent +\npredict gender without it?"),
        ("RQ3", "Do contrastive examples\nhelp (ICL? FT?)"),
        ("RQ4", "Authentic + synthetic\nvs each alone?"),
    ]
    for i, (k, v) in enumerate(rqs):
        x = 1 + i * 24.75
        box(x, 8, 23, 16, f"{k}\n{v}", color="#f5f5f5", edge="#777",
            fontsize=9, weight="normal")

    fig.suptitle("Pipeline overview: data → augmentation → benchmark + FT → eval",
                 y=0.98, fontsize=13, weight="bold")
    fig.savefig(OUT / "fig_overview.png")
    plt.close(fig)
    print("wrote", OUT / "fig_overview.png")


# ---------------------------------------------------------------------------
# Figure 2 - Closed vs open task contrast (one sentence)
# ---------------------------------------------------------------------------
def fig_tasks():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    plt.subplots_adjust(wspace=0.05)

    # Shared user-sentence example, pre-wrapped to fit the panel width.
    sentence_lines_closed = (
        "USER:\n"
        "Sentence:  The author, Mr. Chen, follows 8 Milwaukee\n"
        "           families through their daily struggle to\n"
        "           find and keep a shelter.\n"
        "Referent:  author"
    )
    sentence_lines_open = (
        "USER:\n"
        "Sentence:  The author, Mr. Chen, follows 8 Milwaukee\n"
        "           families through their daily struggle to\n"
        "           find and keep a shelter."
    )

    # Left panel: closed task
    ax = axes[0]
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(50, 95, "Task 1: closed", ha="center", fontsize=14, weight="bold",
            color="#3b6aa1")
    ax.text(50, 88, "input: sentence + given referent", ha="center", fontsize=9.5,
            style="italic", color="#444")

    # User input
    box1 = mpatches.FancyBboxPatch((4, 56), 92, 26, boxstyle="round,pad=0.4",
                                     facecolor="#e7eef7", edgecolor="#3b6aa1", lw=1.2)
    ax.add_patch(box1)
    ax.text(8, 69, sentence_lines_closed,
            ha="left", va="center", fontsize=8.5, family="monospace")

    ax.annotate("", xy=(50, 50), xytext=(50, 56),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.5))

    # Output
    box2 = mpatches.FancyBboxPatch((4, 20), 92, 28, boxstyle="round,pad=0.4",
                                     facecolor="#e7f4e7", edgecolor="#3a7a3a", lw=1.2)
    ax.add_patch(box2)
    ax.text(8, 34, ('ASSISTANT:\n'
                    '{"gender": "masculine",\n'
                    ' "confidence": 5,\n'
                    ' "reasoning": "The title \'Mr.\' bound to the\n'
                    '               referent is a male-marking signal."}'),
            ha="left", va="center", fontsize=8.5, family="monospace")

    ax.text(50, 10, "→ scoring: predicted_label == expected_label?",
            ha="center", fontsize=9, color="#444", style="italic")

    # Right panel: open task
    ax = axes[1]
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(50, 95, "Task 2: open", ha="center", fontsize=14, weight="bold",
            color="#a52a2a")
    ax.text(50, 88, "input: sentence only; model must surface referents",
            ha="center", fontsize=9.5, style="italic", color="#444")

    box1 = mpatches.FancyBboxPatch((4, 60), 92, 22, boxstyle="round,pad=0.4",
                                     facecolor="#e7eef7", edgecolor="#3b6aa1", lw=1.2)
    ax.add_patch(box1)
    ax.text(8, 71, sentence_lines_open,
            ha="left", va="center", fontsize=8.5, family="monospace")

    ax.annotate("", xy=(50, 54), xytext=(50, 60),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.5))

    box2 = mpatches.FancyBboxPatch((4, 16), 92, 36, boxstyle="round,pad=0.4",
                                     facecolor="#fde7e7", edgecolor="#a52a2a", lw=1.2)
    ax.add_patch(box2)
    ax.text(8, 34, ('ASSISTANT:\n'
                    '{"referents": [\n'
                    '   {"referent": "author", "gender": "masculine",\n'
                    '    "confidence": 5,\n'
                    '    "reasoning": "Mr. Chen is masculine and\n'
                    '                  refers to author"},\n'
                    '   {"referent": "families", "gender": "ambiguous",\n'
                    '    "confidence": 5,\n'
                    '    "reasoning": "no gendered cue"}\n'
                    ']}'),
            ha="left", va="center", fontsize=8, family="monospace")

    ax.text(50, 8, "→ scoring: look up annotated referent in response;\n"
                   "   bucket = its gender label, or 'not_found' if absent",
            ha="center", fontsize=9, color="#444", style="italic")

    fig.suptitle("Closed vs open task: same sentence, different I/O contract",
                 y=1.02, fontsize=13, weight="bold")
    fig.savefig(OUT / "fig_tasks.png")
    plt.close(fig)
    print("wrote", OUT / "fig_tasks.png")


# ---------------------------------------------------------------------------
# Figure 3 - Closed -> open gap, all 7 models, ICL zs
# ---------------------------------------------------------------------------
def fig_closed_open_gap():
    df = pd.read_csv(ROOT / "results/eval" / "headline_thinking_final.csv")
    # Keep only the 7 ICL-tested models in our preferred display order
    order = [
        ("Gemma-4-31B-it",                "Gemma-31B"),
        ("Qwen3.6-27B",                   "Qwen-27B"),
        ("gpt-oss-20b",                   "gpt-oss-20B"),
        ("Magistral-Small-2506",          "Magistral-24B"),
        ("Gemma-4-E4B-it",                "Gemma-E4B"),
        ("Qwen3.5-4B",                    "Qwen-4B"),
        ("Ministral-3-3B-Instruct-2512",  "Ministral-3B"),
    ]
    rows = []
    for full, short in order:
        r = df[df["model"] == full].iloc[0]
        rows.append({"model": short, "closed": r["closed_accuracy"], "open": r["open_accuracy"]})
    d = pd.DataFrame(rows)
    d["gap"] = d["closed"] - d["open"]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(d))
    w = 0.36
    ax.bar(x - w/2, d["closed"], width=w, color="#3b6aa1", label="Closed task accuracy")
    ax.bar(x + w/2, d["open"],   width=w, color="#a52a2a", label="Open task accuracy")
    for i, gap in enumerate(d["gap"]):
        y = max(d["closed"][i], d["open"][i]) + 0.015
        ax.annotate(f"−{gap*100:.1f} pp",
                    xy=(i, y), ha="center", fontsize=8.5, color="#444", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(d["model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Accuracy on test (n=1,395)")
    ax.set_title("Closed → open accuracy gap (ICL zero-shot, thinking on)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, framealpha=0.9, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(OUT / "fig_closed_open_gap.png")
    plt.close(fig)
    print("wrote", OUT / "fig_closed_open_gap.png")


# ---------------------------------------------------------------------------
# Figure 4 - Qwen3.5-4B FT lift summary across conditions
# ---------------------------------------------------------------------------
def fig_ft_summary():
    closed_zs = pd.read_csv(ROOT / "results/eval" / "headline_thinking_final.csv")
    closed_zs = closed_zs.set_index("model").loc["Qwen3.5-4B"]
    closed_fs = pd.read_csv(ROOT / "results/eval" / "headline_fewshot_final.csv")
    closed_fs = closed_fs.set_index("model").loc["Qwen3.5-4B"]
    closed_ft = pd.read_csv(ROOT / "results/eval" / "headline_finetune_final.csv").set_index("model")
    open_ft = pd.read_csv(ROOT / "results/eval" / "headline_finetune_open_final.csv").set_index("model")

    # Closed conditions: ICL-zs, ICL-fs, FT-DAUTH, FT-DA
    closed_labels = ["ICL\nzero-shot", "ICL\nfew-shot",
                     "FT\nD-AUTH", "FT\nD-A"]
    closed_vals = [
        float(closed_zs["closed_accuracy"]),
        float(closed_fs["closed_accuracy"]),
        float(closed_ft.loc["Qwen3.5-4B / FT D-AUTH", "accuracy"]),
        float(closed_ft.loc["Qwen3.5-4B / FT D-A", "accuracy"]),
    ]
    closed_colors = [C_BASE, C_BASE, C_FT_DAUTH, C_FT_DA]

    # Open conditions: ICL-zs, ICL-fs, FT-DAUTH-open, FT-DA-open, FT-DMULTI-open
    open_labels = ["ICL\nzero-shot", "ICL\nfew-shot",
                   "FT\nD-AUTH", "FT\nD-A", "FT\nD-MULTI"]
    open_vals = [
        float(closed_zs["open_accuracy"]),
        float(closed_fs["open_accuracy"]),
        float(open_ft.loc["Qwen3.5-4B / FT D-AUTH-open", "accuracy"]),
        float(open_ft.loc["Qwen3.5-4B / FT D-A-open", "accuracy"]),
        float(open_ft.loc["Qwen3.5-4B / FT D-MULTI-open", "accuracy"]),
    ]
    open_colors = [C_BASE, C_BASE, C_FT_DAUTH, C_FT_DA, C_FT_DMULTI]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)

    for ax, labels, vals, colors, title, accent in [
        (axes[0], closed_labels, closed_vals, closed_colors, "Closed task (predict ambiguity given referent)",  "#3b6aa1"),
        (axes[1], open_labels,   open_vals,   open_colors,   "Open task (find referent + predict gender)",       "#a52a2a"),
    ]:
        x = np.arange(len(vals))
        bars = ax.bar(x, vals, color=colors, edgecolor="#333", linewidth=0.7)
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v*100:.1f}%", xy=(bar.get_x()+bar.get_width()/2, v+0.01),
                        ha="center", fontsize=9, color="#222")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_title(title, color=accent)
        ax.grid(axis="y", alpha=0.3)
        ax.axvline(1.5, color="#bbb", linestyle="--", linewidth=0.8)
        ax.text(0.5, 1.05, "ICL", ha="center", fontsize=9, color=accent, weight="bold",
                transform=ax.get_xaxis_transform())
        ax.text((len(vals)+0.5)/2, 1.05, "Fine-tuning", ha="center", fontsize=9, color=accent, weight="bold",
                transform=ax.get_xaxis_transform())
    axes[0].set_ylabel("Accuracy on test (n=1,395)")

    fig.suptitle("Qwen3.5-4B: ICL → FT lift across conditions", y=1.02,
                 fontsize=13, weight="bold")
    fig.savefig(OUT / "fig_ft_summary.png")
    plt.close(fig)
    print("wrote", OUT / "fig_ft_summary.png")


# ---------------------------------------------------------------------------
# Figure 5 - Authentic vs synthetic isolation (per-class recall)
# ---------------------------------------------------------------------------
def fig_synthetic_vs_authentic():
    """Show that DAUTH collapses to 'all-ambiguous' while DA recovers per-class recall."""
    closed_ft = pd.read_csv(ROOT / "results/eval" / "headline_finetune_final.csv").set_index("model")
    open_ft   = pd.read_csv(ROOT / "results/eval" / "headline_finetune_open_final.csv").set_index("model")

    conditions_closed = [
        ("D-AUTH (authentic only)",  "Qwen3.5-4B / FT D-AUTH",   "#cccccc"),
        ("D-A (full contrastive)",   "Qwen3.5-4B / FT D-A",      "#2ca02c"),
    ]
    conditions_open = [
        ("D-AUTH-open",   "Qwen3.5-4B / FT D-AUTH-open",  "#cccccc"),
        ("D-A-open",      "Qwen3.5-4B / FT D-A-open",     "#2ca02c"),
        ("D-MULTI-open",  "Qwen3.5-4B / FT D-MULTI-open", "#17becf"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    classes = ["masc_recall", "fem_recall", "ambig_recall"]
    cls_labels = ["masc R", "fem R", "amb R"]

    for ax, conds, df_src, title, accent in [
        (axes[0], conditions_closed, closed_ft, "Closed task (Qwen3.5-4B FT)", "#3b6aa1"),
        (axes[1], conditions_open,   open_ft,   "Open task (Qwen3.5-4B FT)",   "#a52a2a"),
    ]:
        n = len(conds); w = 0.8 / n
        x = np.arange(len(classes))
        for i, (label, model_key, color) in enumerate(conds):
            row = df_src.loc[model_key]
            vals = [float(row[c]) for c in classes]
            bars = ax.bar(x + i*w - (n-1)*w/2, vals, width=w, color=color,
                          edgecolor="#333", linewidth=0.6, label=label)
            for bar, v in zip(bars, vals):
                ax.annotate(f"{v*100:.0f}%", xy=(bar.get_x()+bar.get_width()/2, v+0.01),
                            ha="center", fontsize=8, color="#222")
        ax.set_xticks(x)
        ax.set_xticklabels(cls_labels, fontsize=10)
        ax.set_ylim(0, 1.10)
        ax.set_title(title, color=accent)
        ax.legend(loc="lower right", fontsize=8.5)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Per-class recall (test)")

    fig.suptitle("Synthetic contrastive data is the entire FT signal "
                 "(D-AUTH = authentic only; D-A = + synthetic)",
                 y=1.02, fontsize=12, weight="bold")
    fig.savefig(OUT / "fig_synthetic_vs_authentic.png")
    plt.close(fig)
    print("wrote", OUT / "fig_synthetic_vs_authentic.png")


if __name__ == "__main__":
    fig_overview()
    fig_tasks()
    fig_closed_open_gap()
    fig_ft_summary()
    fig_synthetic_vs_authentic()
