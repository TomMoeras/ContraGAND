from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "figures" / "data"
FIGS = ROOT / "figures"
MODELS = ["Qwen3.6-27B", "Gemma-4-31B-it"]
PALETTE = {"Qwen3.6-27B": "#0072B2", "Gemma-4-31B-it": "#E69F00"}
DPI = 150
FIGW = 6.5


def save(fig, name):
    path = FIGS / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved " + str(path))


def make_fig_01_heatmap():
    df = pd.read_csv(DATA / "01_headline.csv")
    col_defs = [(False, "closed", "Closed zs"), (True, "closed", "Closed think"),
                (False, "open", "Open zs"),   (True, "open", "Open think")]
    mat = np.zeros((2, 4))
    for j, (think, task, _) in enumerate(col_defs):
        for i, model in enumerate(MODELS):
            row = df[(df.model == model) & (df.task == task) & (df.thinking == think)]
            mat[i, j] = float(row.accuracy)
    fig, ax = plt.subplots(figsize=(FIGW, 2.4))
    im = ax.imshow(mat, cmap="YlGn", vmin=0.68, vmax=1.0, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels([c[2] for c in col_defs], fontsize=10)
    ax.set_yticks(range(2))
    ax.set_yticklabels(MODELS, fontsize=10)
    for i in range(2):
        for j in range(4):
            color = "black" if mat[i, j] < 0.94 else "white"
            ax.text(j, i, "%.3f" % mat[i, j], ha="center", va="center",
                    fontsize=10, color=color)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Accuracy")
    ax.set_title("Figure 1 - Overall accuracy (8-cell matrix)", fontsize=11, pad=8)
    fig.tight_layout()
    save(fig, "fig_01_heatmap.png")


def make_fig_02_per_strategy():
    df = pd.read_csv(DATA / "02_per_strategy.csv")
    df = df.sort_values("qwen_closed_zs")
    strategies = df.strategy.tolist()
    x = np.arange(len(strategies))
    width = 0.35
    fig, ax = plt.subplots(figsize=(FIGW, 4.5))
    ax.barh(x - width / 2, df.qwen_closed_zs, width, label="Qwen3.6-27B", color=PALETTE["Qwen3.6-27B"])
    ax.barh(x + width / 2, df.gemma_closed_zs, width, label="Gemma-4-31B-it", color=PALETTE["Gemma-4-31B-it"])
    ax.set_yticks(x)
    ax.set_yticklabels(strategies, fontsize=9)
    ax.set_xlabel("Accuracy (masculine variant, closed zs)")
    ax.set_xlim(0.4, 1.05)
    ax.axvline(1.0, color="grey", linewidth=0.6, linestyle="--")
    ax.legend(fontsize=9)
    ax.set_title("Figure 2 - Per-strategy accuracy (masculine, closed zero-shot)", fontsize=11)
    fig.tight_layout()
    save(fig, "fig_02_per_strategy.png")


def make_fig_03_open_confusion():
    df = pd.read_csv(DATA / "03_open_confusion.csv")
    true_labels = ["masculine", "feminine", "ambiguous"]
    pred_labels = ["masculine", "feminine", "ambiguous", "not_found"]
    colors = {"masculine": "#0072B2", "feminine": "#CC79A7",
              "ambiguous": "#999999", "not_found": "#F0E442"}
    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 3.0), sharey=True)
    for ax, model in zip(axes, MODELS):
        sub = df[(df.model == model) &
                 (df.true_label.isin(true_labels)) &
                 (df.pred_label.isin(pred_labels))]
        row_totals = sub.groupby("true_label")["count"].transform("sum")
        sub = sub.copy()
        sub["frac"] = sub["count"] / row_totals
        lefts = np.zeros(len(true_labels))
        for pred in pred_labels:
            fracs = []
            for tl in true_labels:
                r = sub[(sub.true_label == tl) & (sub.pred_label == pred)]
                fracs.append(float(r.frac) if len(r) == 1 else 0.0)
            ax.barh(true_labels, fracs, left=lefts, color=colors[pred], label=pred)
            lefts += np.array(fracs)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction")
        ax.set_title(model, fontsize=10)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[0].set_ylabel("True label")
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.12), fontsize=9, title="Predicted")
    fig.suptitle("Figure 3 - Row-normalised confusion (open task, zero-shot)", fontsize=11, y=1.02)
    fig.tight_layout()
    save(fig, "fig_03_open_confusion.png")


def make_fig_04_thinking_delta():
    df = pd.read_csv(DATA / "04_thinking_delta.csv")
    tasks = ["closed", "open"]
    x = np.arange(2)
    width = 0.35
    fig, ax = plt.subplots(figsize=(FIGW * 0.6, 3.2))
    for i, model in enumerate(MODELS):
        deltas = [float(df[(df.model == model) & (df.task == t)].delta) for t in tasks]
        ax.bar(x + (i - 0.5) * width, deltas, width,
               label=model, color=PALETTE[model])
    ax.set_xticks(x)
    ax.set_xticklabels(["Closed task", "Open task"], fontsize=10)
    ax.set_ylabel("Accuracy delta (think - base)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.legend(fontsize=9)
    ax.set_title("Figure 4 - Thinking-mode accuracy gain", fontsize=11)
    fig.tight_layout()
    save(fig, "fig_04_thinking_delta.png")


def make_fig_05_stereotype_overlap():
    df = pd.read_csv(DATA / "05_stereotype_overlap.csv")
    df_op = df[df.task == "open"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 4.0))
    for ax, model in zip(axes, MODELS):
        sub = df_op[df_op.model == model].sort_values("male_share")
        referents = sub.referent.tolist()
        x = np.arange(len(referents))
        masc_frac = sub.male_share.values
        fem_frac = 1 - masc_frac
        ax.barh(x, masc_frac, color="#0072B2", label="masculine")
        ax.barh(x, -fem_frac, color="#CC79A7", label="feminine")
        for xi, is_overlap in enumerate(sub.cross_model_overlap):
            if str(is_overlap).lower() == "true":
                ax.text(0.01, xi, "*", va="center", fontsize=10, fontweight="bold")
        ax.set_yticks(x)
        ax.set_yticklabels(referents, fontsize=8)
        ax.set_xlim(-1.1, 1.1)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: "%.0f%%" % (abs(v)*100)))
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Confident wrong guesses")
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.06), fontsize=9)
    fig.suptitle(
        "Figure 5 - Stereotype bias: ambiguous referents guessed with confidence",
        fontsize=10, y=1.03)
    fig.tight_layout()
    save(fig, "fig_05_stereotype_overlap.png")


if __name__ == "__main__":
    print("Generating figures ...")
    make_fig_01_heatmap()
    make_fig_02_per_strategy()
    make_fig_03_open_confusion()
    make_fig_04_thinking_delta()
    make_fig_05_stereotype_overlap()
    print("Done.")
