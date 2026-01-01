# SpecBench Simulator

A simulator for speculative decoding methods with different proposers and prediction strategies. Note that we only support Llama3.1-8B-Instruct with H100 for now, but it is easy to extend to other models or hardware.

## Installation

```bash
conda create -n specbench_simulator python=3.10
conda activate specbench_simulator
pip install -r requirements.txt
```

## Quick Start

Run the full simulation for all datasets, proposers, and methods:

```bash
./simulate.sh
```
The simulator will read in the data from `data/{model}/combined_{dataset}.jsonl` and save the results to `results/{proposer}_{dataset}_speedup.csv`.
In `data/`, we share our profiled acceptance data for each dataset of Llama3.1-8B-Instruct with H100.

## Usage

### Basic Command

```bash
python main.py --proposer <proposer> --datasets <datasets> --batch-sizes <sizes>
```

### Options

- `--proposer`: Proposer method (`eagle3`, `ngram`, `combined`)
- `--datasets`: One or more datasets (`gsm8k`, `instructcoder`, `sharegpt`, `cnn`)
- `--predict-methods`: Prediction methods (`FIXED_1`, `FIXED_3`, `FIXED_5`, `ORACLE`)
- `--batch-sizes`: Batch sizes to test (e.g., `1 8 16 32 64 128`)
- `--output-dir`: Directory for output CSV files (default: `.`)

### Examples

Run EAGLE3 on a single dataset:
```bash
python main.py --proposer eagle3 --datasets gsm8k --batch-sizes 1 8 16
```

Run NGRAM with specific prediction methods:
```bash
python main.py --proposer ngram --datasets sharegpt --predict-methods FIXED_1 ORACLE
```

Run combined proposer (automatically uses ORACLE):
```bash
python main.py --proposer combined --datasets cnn
```

### Output

Results are saved as CSV files with the format: `{proposer}_{dataset}_speedup.csv`

Each file contains columns:
- `batch_size`: The batch size used
- `predict_method`: The prediction method
- `acc_method`: The acceptance ground truth method
- `speedup`: The measured speedup value

## Add Other Models and Hardware
Coming soon.

## Citation
Coming soon.