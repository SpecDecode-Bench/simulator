# SpecBench Simulator

A simulator for speculative decoding methods with different proposers and prediction strategies. You can find more details in our [paper](https://arxiv.org/abs/2601.11580). Note that we only support Llama3.1-8B-Instruct with H100 for now, but it is easy to extend to other models or hardware.

## Installation

```bash
conda create -n specbench_simulator python=3.10
conda activate specbench_simulator
pip install -r requirements.txt
```

## Quick Start

Run the full simulation and generate plots for all datasets, proposers, and methods:

```bash
./simulate.sh
```

This will:
1. Run simulations for EAGLE3, NGRAM, and combined proposers across all datasets
2. Save results to `results/{proposer}_{dataset}_speedup.csv`
3. Generate speedup plots and legends to `figures/llama3.1-8B/`

Input data is read from `data/{model}/combined_{dataset}.jsonl`. We share our profiled acceptance data for Llama3.1-8B-Instruct on H100 in `data/`.

## Usage

### Simulation

```bash
python main.py --proposer <proposer> --datasets <datasets> --batch-sizes <sizes>
```

**Options**

- `--proposer`: Proposer method (`eagle3`, `ngram`, `combined`)
- `--datasets`: One or more datasets (`gsm8k`, `instructcoder`, `sharegpt`, `cnn`)
- `--predict-methods`: Prediction methods (`FIXED_1`, `FIXED_3`, `FIXED_5`, `ORACLE`). Ignored for `combined` (always uses `ORACLE`)
- `--batch-sizes`: Batch sizes to test (default: `1 8 16 32 64 128`)
- `--output-dir`: Directory for output CSV files (default: `results`)

**Examples**

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

**Output**

Results are saved as `results/{proposer}_{dataset}_speedup.csv` with columns:
- `batch_size`: The batch size used
- `predict_method`: The prediction method (`FIXED_1`, `FIXED_3`, `FIXED_5`, `ORACLE`)
- `acc_method`: The acceptance ground truth method (`EAGLE`, `NGRAM`, `COMBINE_NGRAM_EAGLE`)
- `speedup`: The measured speedup value

### Plotting

```bash
python plot.py --results-dir <dir> --figures-dir <dir> --proposers <proposers> --datasets <datasets>
```

**Options**

- `--results-dir`: Directory containing simulation CSV files (default: `results`)
- `--figures-dir`: Directory to save output figures (default: `figures`)
- `--model`: Model name used to organize output subdirectory (default: `llama3.1-8B`)
- `--proposers`: One or more proposers to plot (`eagle3`, `ngram`, `combined`; default: `eagle3 ngram`)
- `--datasets`: One or more datasets to plot (default: all four)

**Example**

```bash
python plot.py --proposers eagle3 ngram combined --datasets gsm8k sharegpt
```

Figures are saved as `figures/{model}/{proposer}_{dataset}_speedup.pdf`. A standalone legend file `{proposer}_legend_only.pdf` is also saved per proposer type. For `combined` plots, data from all three proposers is merged and lines are colored by acceptance method (EAGLE3, N-gram, Combine).

## Add Other Models and Hardware
Coming soon.

## Citation
Please cite our paper if you find the repository helpful. Thank you!
```
@misc{liu2025specdecode,
      title={Speculative Decoding: Performance or Illusion?}, 
      author={Xiaoxuan Liu and Jiaxiang Yu and Jongseok Park and Ion Stoica and Alvin Cheung},
      year={2025},
      eprint={2601.11580},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.11580}, 
}
```
