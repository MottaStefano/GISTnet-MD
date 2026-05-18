# 🧬 GISTnet-MD: Graph-Informed Spatiotemporal Networks for Molecular Dynamics

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-12.1-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GISTnet-MD** is an advanced Deep Learning pipeline built on **PyTorch Geometric**. It is designed to train Spatiotemporal Graph Neural Networks (GNNs) capable of recognizing complex functional or conformational states directly from raw Molecular Dynamics (MD) trajectories.

Historically, analyzing MD simulations required the manual definition of collective variables or classical dimensionality reduction techniques. GISTnet-MD introduces a paradigm shift: **automated, physics-informed pattern extraction** using Deep Learning, followed by a rigorous **Explainable AI (xAI)** framework (Integrated Gradients) to physically interpret the network's decisions at the atomic level.

---

## 🎯 The Biophysical Problem

The pipeline isolates the true "signal"—the mechanistic movements that dictate biological function—from the overwhelming thermal "noise". By representing proteins as **dynamic spatiotemporal graphs** (where nodes are amino acids and edges denote spatial proximity), we ensure translational and rotational invariance, inherently mapping the physical reality of non-covalent interactions over time.

---

## 🚀 Installation & Environment Setup

We strongly recommend using `conda` to manage the environment, as it seamlessly handles the CUDA dependencies required for GPU acceleration.

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Antigravity_GISTnet-MD.git
cd Antigravity_GISTnet-MD

# 2. Create the conda environment from the provided file
conda env create -f environment.yml

# 3. Activate the environment
conda activate gistnet-md
```

---

## 📂 Repository Structure

```text
GISTnet-MD/
├── GISTnet-MD/                # Main Python package
│   ├── preprocess.py          # Phase 1: Converts MD trajectories to PyTorch Graphs (.pt)
│   ├── train.py               # Phase 2: Trains the Spatiotemporal GNN (HybridStSchnet)
│   ├── xai_ig_pipeline.py     # Phase 3: Applies Integrated Gradients for xAI
│   ├── xai_pymol_generator.py # Phase 4: Generates PyMOL consensus scripts
│   └── core/                  # Core Python modules (Architecture, Loss, Datasets)
├── docs/                      # Extensive User Manuals and Markdown guides
├── environment.yml            # Conda dependencies definition
└── README.md                  # This file
```

---

## 🛠️ End-to-End Pipeline Quickstart (WT vs L99A Mutant)

The GISTnet-MD workflow is strictly sequential. Below is a complete bash example demonstrating how to classify a Wild-Type (WT) protein against an L99A mutant, using 5 MD groups (replicas) for each state, and extracting the structural determinants.

### Phase 1: Topological Preprocessing
Convert standard `.xtc` and `.pdb` files into PyTorch Geometric graphs. Here, we use the Center of Mass (`com`) representation with a 10.0 Å cutoff.

```bash
mkdir -p preprocessed_graphs/WT preprocessed_graphs/L99A

# Preprocess the 5 WT groups (Class WT - label 0)
for i in {1..5}; do
  python GISTnet-MD/preprocess.py \
    --xtc ../MD_simulations/WT/rep_${i}.xtc \
    --pdb ../MD_simulations/WT/complex-dry.pdb \
    --resid 1-98,100-164 \
    --out_dir preprocessed_graphs/WT/WT_rep_${i} \
    --prefix WT_rep_${i} \
    --representation com \
    --cutoff 10.0
done

# Repeat the loop for L99A data into preprocessed_graphs/L99A/
```
📚 **Read the full guide:** [`docs/preprocess_guide.md`](docs/preprocess_guide.md)

### Phase 2: Spatiotemporal Training
Train the model using a Leave-One-Group-Out (LOGO) strategy to prevent data leakage. The model looks at sliding windows of 10 frames.

```bash
mkdir -p training

# Train 5 separate models, holding out one group for validation each time
for i in {1..5}; do
    python GISTnet-MD/train.py \
     --data_class_dirs preprocessed_graphs/WT/ preprocessed_graphs/L99A/ \
     --data_class_labels WT L99A \
     --out_dir training/valrep_${i}_results \
     --window 10 \
     --val_groups ${i}
done
```
📚 **Read the full guide:** [`docs/train_guide.md`](docs/train_guide.md)

### Phase 3: Explainable AI (Integrated Gradients)
Ask the network *why* it made its decisions. IG calculates an "importance score" (positive/negative) for every residue and interaction, extracting true structural determinants relative to the mean training baseline.

```bash
python GISTnet-MD/xai_ig_pipeline.py \
  --model_dirs training/valrep_* \
  --out_dir ./analysis_IG \
  --ig_steps 10 \
  --baseline thermodynamic_mean
```
📚 **Read the full guide:** [`docs/ig_pipeline_guide.md`](docs/ig_pipeline_guide.md)

### Phase 4: Structural Visualization (PyMOL)
Aggregate the raw xAI mathematical attributions into human-readable 3D structures.

```bash
python GISTnet-MD/xai_pymol_generator.py \
  --ig_results_dir ./analysis_IG \
  --pdb_template ../reference.pdb \
  --out_dir ./pymol_viz \
  --resid 1-98,100-164 \
  --confidence_cut 0.8 \
  --top_edges 50 100 250
```

> **To view:** Open PyMOL and run `@./pymol_viz/global_view_xai.pml` in the command line.

📚 **Read the full guide:** [`docs/pymol_generator_guide.md`](docs/pymol_generator_guide.md)

---

## 📚 Detailed Documentation

For a deep dive into hyperparameters, architectural details, and advanced optimization options, please consult the guides in the `docs/` directory:

1. [Preprocessing Guide (`preprocess_guide.md`)](docs/preprocess_guide.md)
2. [Training Guide (`train_guide.md`)](docs/train_guide.md)
3. [Integrated Gradients Pipeline Guide (`ig_pipeline_guide.md`)](docs/ig_pipeline_guide.md)
4. [PyMOL Visualization Generator Guide (`pymol_generator_guide.md`)](docs/pymol_generator_guide.md)
