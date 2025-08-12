# Chaos Analysis on Neuronal ISI Sequences

This repository contains scripts to extract inter-spike interval (ISI) sequences from externally sourced NWB files and evaluate chaotic signatures using nonlinear time series analysis.

## 📁 Structure

```
chaos-isi-analysis/
├── isi_extraction/        # ISI extraction scripts
├── metrics/               # Chaos metric computations
├── docs/                  # Metric explanations
├── README.md              # Landing page
├── requirements.txt       # Python dependencies
├── benchmarking/        # benchmarking synthetic signals and metrics
```

## 🔍 Pipeline Overview

1. **Extract ISIs**
   - `extract_task_relevant.py`: extracts ISIs during task trials
   - `extract_task_irrelevant.py`: extracts ISIs from inter-trial intervals

2. **Analyze Chaos**
   - SE, CD, LE

3. **Statistical Testing**
   - Phase-randomized surrogates
   - Z-score and empirical p-value

## 📂 Data

This repository assumes access to NWB files from [https://doi.org/10.48324/dandi.000004/0.220126.1852](https://doi.org/10.48324/dandi.000004/0.220126.1852).
Place NWB files under:

```
./data/nwbs_all/
```

## 🧰 Usage

```bash
python isi_extraction/extract_task_relevant.py
python metrics/rqa_rel.py
```

## 🧪 Dependencies

```bash
pip install -r requirements.txt
```

## 📜 Citation

If this codebase contributes to your research, please cite or reference the GitHub repository:

> Chitale, S.A. (2025). *Chaos Analysis on Neuronal ISI Sequences*.  
> GitHub Repository: [https://github.com/saarth-chitale/Chaos-Detection-Pipeline](https://github.com/saarth-chitale/Chaos-Detection-Pipeline)
