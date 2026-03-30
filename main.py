import os
import random
import argparse
random.seed(42)
from simulator import SpecSimulator, AccLenGroundTruth
from predictor import AccLenPredictMethod

MODEL_NAMES = {
    "llama3.1-8B": "Meta-Llama-3.1-8B-Instruct"
}
PREDICT_METHOD_MAP = {
    "FIXED_1": AccLenPredictMethod.FIXED_1,
    "FIXED_3": AccLenPredictMethod.FIXED_3,
    "FIXED_5": AccLenPredictMethod.FIXED_5,
    "ADAPTIVE": AccLenPredictMethod.ADAPTIVE,
    "ORACLE": AccLenPredictMethod.ORACLE,
}
ACC_GT_METHOD_MAP = {
    "eagle3": AccLenGroundTruth.EAGLE,
    "ngram": AccLenGroundTruth.NGRAM,
    "combined": AccLenGroundTruth.COMBINE_NGRAM_EAGLE,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run speculative decoding benchmarks")

    parser.add_argument(
        "--proposer",
        type=str,
        default="eagle3",
        choices=list(ACC_GT_METHOD_MAP.keys()),
        help="Proposer method to use"
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["gsm8k", "instructcoder", "sharegpt", "cnn"],
        choices=["gsm8k", "instructcoder", "sharegpt", "cnn"],
        help="Datasets to run simulations on"
    )

    parser.add_argument(
        "--predict-methods",
        type=str,
        nargs="+",
        default=["FIXED_1", "FIXED_3", "FIXED_5", "ADAPTIVE", "ORACLE"],
        choices=list(PREDICT_METHOD_MAP.keys()),
        help="Prediction methods to use"
    )

    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 16, 32, 64, 128],
        help="Batch sizes to test"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save output CSV files"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    model = "llama3.1-8B"

    predict_methods = [PREDICT_METHOD_MAP[pm] for pm in args.predict_methods]
    acc_gt_method = ACC_GT_METHOD_MAP[args.proposer]

    if acc_gt_method == AccLenGroundTruth.COMBINE_NGRAM_EAGLE:
        predict_methods = [AccLenPredictMethod.ORACLE]
        print(f"Using combined NGRAM and EAGLE proposer with ORACLE predict method, ignoring other predict methods...")

    for dataset in args.datasets:
        print(f"\n{'='*80}")
        print(f"Processing Dataset: {dataset}")
        print(f"Model: {model}, Proposer: {args.proposer}")
        print(f"{'='*80}\n")

        input_filepath = f"data/{model}/combined_{dataset}.jsonl"
        proposer_dir = os.path.join(args.output_dir, args.proposer)
        os.makedirs(proposer_dir, exist_ok=True)
        simulation_output = os.path.join(proposer_dir, f"{dataset}_speedup.csv")

        if not os.path.exists(simulation_output):
            with open(simulation_output, "w") as f:
                f.write("batch_size,predict_method,acc_method,speedup\n")

        for pm in predict_methods:
            simulator = SpecSimulator(input_filepath, pm, acc_gt_method, dataset)
            for batch_size in args.batch_sizes:
                print(f"[Model: {model}, Proposer: {args.proposer}] Batch Size={batch_size}, Predict Method={pm.name}, Acc GT Method={acc_gt_method.name}")
                speedup = simulator.simulate_like_experiment(batch_size)
                with open(simulation_output, "a") as f:
                    f.write(f"{batch_size},{pm.name},{acc_gt_method.name},{speedup:.2f}\n")

if __name__ == "__main__":
    main()
