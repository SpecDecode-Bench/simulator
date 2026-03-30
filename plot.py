import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
from predictor import AccLenPredictMethod

MARKERS_COLORS = {
    AccLenPredictMethod.FIXED_1.name: ["o", "#1f77b4"],
    AccLenPredictMethod.FIXED_3.name: ["s", "#ff7f0e"],
    AccLenPredictMethod.FIXED_5.name: ["D", "#2ca02c"],
    AccLenPredictMethod.ADAPTIVE.name: ["^", "#9467bd"],
    AccLenPredictMethod.ORACLE.name: ["*", "#d62728"],
}

PMNAMES = {
    AccLenPredictMethod.FIXED_1.name: "k=1",
    AccLenPredictMethod.FIXED_3.name: "k=3",
    AccLenPredictMethod.FIXED_5.name: "k=5",
    AccLenPredictMethod.ADAPTIVE.name: "Adaptive",
    AccLenPredictMethod.ORACLE.name: "Oracle",
}

AMNAMES = {
    "NGRAM": "Oracle_N-gram",
    "EAGLE": "Oracle_EAGLE3",
    "COMBINE_NGRAM_EAGLE": "Oracle_Combined",
}

MARKERS_COLORS_AM = {
    "NGRAM": "#2ca02c",
    "EAGLE": "#ff7f0e",
    "COMBINE_NGRAM_EAGLE": "#d62728",
}


def _get_style(predict_method):
    if predict_method == AccLenPredictMethod.ORACLE.name:
        return 'dotted', 15, 1
    elif predict_method == AccLenPredictMethod.ADAPTIVE.name:
        return '-.', 10, 1
    else:
        return '-', 8, 0.75


def save_legend_only(df, output_filepath, ncol=None, fontsize=16, combined=False):
    fig, ax = plt.subplots(figsize=(1, 1))

    handles = []
    labels = []
    for subset, color, marker, linestyle, markersize, alpha, label in _get_series(df, combined):
        line, = ax.plot([], [], color=color, marker=marker,
                        linestyle=linestyle, linewidth=3,
                        alpha=alpha, markersize=markersize)
        handles.append(line)
        labels.append(label)

    # Add shaded region legend entry for combined overhead
    if combined and 'switching_overhead' in df.columns and df[df['acc_method'] == "COMBINE_NGRAM_EAGLE"]['switching_overhead'].nunique() > 1:
        patch = ax.fill_between([], [], [], color=MARKERS_COLORS_AM["COMBINE_NGRAM_EAGLE"], alpha=0.2)
        handles.append(patch)

    if ncol is None:
        ncol = len(handles)

    fig_legend = plt.figure(figsize=(12, 0.5))
    fig_legend.legend(handles, labels, loc='center', ncol=ncol,
                      fontsize=fontsize, frameon=True)

    fig_legend.savefig(output_filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    plt.close(fig_legend)


def _get_series(df, combined):
    """Yield (subset_df, color, marker, linestyle, markersize, alpha, label) for each line to plot."""
    if combined:
        pm = AccLenPredictMethod.ORACLE.name
        for am in df['acc_method'].unique():
            subset = df[(df['predict_method'] == pm) & (df['acc_method'] == am)]
            if len(subset) == 0:
                continue
            # For combined with multiple overhead values, use the no-overhead subset
            if am == "COMBINE_NGRAM_EAGLE" and 'switching_overhead' in df.columns and subset['switching_overhead'].nunique() > 1:
                subset = subset[subset['switching_overhead'] == 0.0]
            yield subset, MARKERS_COLORS_AM[am], MARKERS_COLORS[pm][0], 'dotted', 15, 1, AMNAMES[am]
    else:
        for pm in df['predict_method'].unique():
            for am in df['acc_method'].unique():
                subset = df[(df['predict_method'] == pm) & (df['acc_method'] == am)]
                if len(subset) == 0:
                    continue
                line_style, marker_size, alpha = _get_style(pm)
                yield subset, MARKERS_COLORS[pm][1], MARKERS_COLORS[pm][0], line_style, marker_size, alpha, PMNAMES[pm]


def _add_overhead_shading(df, ax):
    """Add shaded region between overhead=0 and overhead>0 for the combined method."""
    if 'switching_overhead' not in df.columns:
        return
    combined = df[df['acc_method'] == "COMBINE_NGRAM_EAGLE"]
    if combined.empty or combined['switching_overhead'].nunique() <= 1:
        return
    no_oh = combined[combined['switching_overhead'] == 0.0].sort_values('batch_size')
    with_oh = combined[combined['switching_overhead'] > 0.0].sort_values('batch_size')
    ax.fill_between(no_oh['batch_size'].values,
                     with_oh['speedup'].values,
                     no_oh['speedup'].values,
                     color=MARKERS_COLORS_AM["COMBINE_NGRAM_EAGLE"], alpha=0.2)


def plot_speedup(df, output_filepath, combined=False):
    fig, ax = plt.subplots(figsize=(4.5, 5))

    for subset, color, marker, linestyle, markersize, alpha, label in _get_series(df, combined):
        ax.plot(subset['batch_size'], subset['speedup'],
                color=color, marker=marker,
                linestyle=linestyle, linewidth=3,
                alpha=alpha, label=label,
                markersize=markersize)

    if combined:
        _add_overhead_shading(df, ax)

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
                df = pd.read_csv(os.path.join(args.results_dir, p, f"{dataset}_speedup.csv"))
                if 'switching_overhead' not in df.columns:
                    df['switching_overhead'] = 0.0
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)
        df = pd.read_csv(os.path.join(args.results_dir, proposer, f"{dataset}_speedup.csv"))
        if 'switching_overhead' not in df.columns:
            df['switching_overhead'] = 0.0
        return df

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
