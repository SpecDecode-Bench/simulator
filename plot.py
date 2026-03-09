import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
from predictor import AccLenPredictMethod

MARKERS_COLORS = {
    AccLenPredictMethod.FIXED_1.name: ["o", "#1f77b4"],
    AccLenPredictMethod.FIXED_3.name: ["s", "#ff7f0e"],
    AccLenPredictMethod.FIXED_5.name: ["D", "#2ca02c"],
    AccLenPredictMethod.ORACLE.name: ["*", "#d62728"],
}

PMNAMES = {
    AccLenPredictMethod.FIXED_1.name: "k=1",
    AccLenPredictMethod.FIXED_3.name: "k=3",
    AccLenPredictMethod.FIXED_5.name: "k=5",
    AccLenPredictMethod.ORACLE.name: "Oracle",
}

AMNAMES = {
    "NGRAM": "N-gram",
    "EAGLE": "EAGLE3",
    "COMBINE_NGRAM_EAGLE": "Combine",
}

MARKERS_COLORS_AM = {
    "NGRAM": "#2ca02c",
    "EAGLE": "#ff7f0e",
    "COMBINE_NGRAM_EAGLE": "#d62728",
}


def _get_style(predict_method):
    if predict_method == AccLenPredictMethod.ORACLE.name:
        return 'dotted', 15, 1
    else:
        return '-', 8, 0.75


def save_legend_only(df, output_filepath, ncol=None, fontsize=16, combined=False):
    fig, ax = plt.subplots(figsize=(1, 1))

    predict_methods = df['predict_method'].unique()
    acc_methods = df['acc_method'].unique()
    handles = []
    labels = []

    for pm in predict_methods:
        for am in acc_methods:
            if am == "COMBINE_NGRAM_EAGLE" and pm != AccLenPredictMethod.ORACLE.name:
                continue
            subset = df[(df['predict_method'] == pm) & (df['acc_method'] == am)]
            if len(subset) > 0:
                line_style, marker_size, alpha = _get_style(pm)
                color = MARKERS_COLORS_AM[am] if combined else MARKERS_COLORS[pm][1]
                marker = MARKERS_COLORS[pm][0]
                line, = ax.plot([], [],
                        color=color,
                        marker=marker,
                        linestyle=line_style, linewidth=3,
                        alpha=alpha,
                        markersize=marker_size)
                handles.append(line)
                if combined:
                    labels.append(f"{PMNAMES[pm]}_{AMNAMES[am]}")
                else:
                    labels.append(PMNAMES[pm])

    if ncol is None:
        ncol = len(handles)

    fig_legend = plt.figure(figsize=(12, 0.5))
    fig_legend.legend(handles, labels, loc='center', ncol=ncol,
                      fontsize=fontsize, frameon=True)

    fig_legend.savefig(output_filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    plt.close(fig_legend)


def plot_speedup(df, output_filepath, combined=False):
    plt.figure(figsize=(4.5, 5))

    predict_methods = df['predict_method'].unique()
    acc_methods = df['acc_method'].unique()
    for pm in predict_methods:
        for am in acc_methods:
            if am == "COMBINE_NGRAM_EAGLE" and pm != AccLenPredictMethod.ORACLE.name:
                continue
            subset = df[(df['predict_method'] == pm) & (df['acc_method'] == am)]
            line_style, marker_size, alpha = _get_style(pm)
            color = MARKERS_COLORS_AM[am] if combined else MARKERS_COLORS[pm][1]
            label = f"{PMNAMES[pm]}_{AMNAMES[am]}"
            plt.plot(subset['batch_size'], subset['speedup'],
                    color=color,
                    marker=MARKERS_COLORS[pm][0],
                    linestyle=line_style, linewidth=3,
                    alpha=alpha,
                    label=label,
                    markersize=marker_size)

    plt.xlabel('Batch Size', fontsize=16)
    plt.ylabel('Speedup', fontsize=16)
    plt.xticks(df['batch_size'].unique())
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(output_filepath, bbox_inches='tight')
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot speculative decoding simulation results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing simulation CSV files")
    parser.add_argument("--figures-dir", type=str, default="figures",
                        help="Directory to save output figures")
    parser.add_argument("--model", type=str, default="llama3.1-8B",
                        help="Model name")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["cnn", "gsm8k", "sharegpt", "instructcoder"],
                        choices=["cnn", "gsm8k", "sharegpt", "instructcoder"],
                        help="Datasets to plot")
    parser.add_argument("--proposers", type=str, nargs="+",
                        default=["eagle3", "ngram"],
                        choices=["eagle3", "ngram", "combined"],
                        help="Proposers to plot")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    figures_dir = os.path.join(args.figures_dir, args.model)
    os.makedirs(figures_dir, exist_ok=True)

    def load_df(proposer, dataset, combined=False):
        if combined:
            dfs = []
            for p in ["eagle3", "ngram", "combined"]:
                dfs.append(pd.read_csv(os.path.join(args.results_dir, f"{p}_{dataset}_speedup.csv")))
            return pd.concat(dfs, ignore_index=True)
        return pd.read_csv(os.path.join(args.results_dir, f"{proposer}_{dataset}_speedup.csv"))

    for dataset in args.datasets:
        for proposer in args.proposers:
            combined = proposer == "combined"
            df = load_df(proposer, dataset, combined)
            output_filepath = os.path.join(figures_dir, f"{proposer}_{dataset}_speedup.pdf")
            plot_speedup(df, output_filepath, combined=combined)

    # Save legend per proposer type (one for normal, one for combined)
    for proposer in args.proposers:
        combined = proposer == "combined"
        df = load_df(proposer, args.datasets[0], combined)
        legend_filepath = os.path.join(figures_dir, f"{proposer}_legend_only.pdf")
        save_legend_only(df, legend_filepath, ncol=5, fontsize=14, combined=combined)
