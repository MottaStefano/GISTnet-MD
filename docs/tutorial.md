# GISTnet-MD: Complete Step-by-Step Tutorial

Welcome to the comprehensive tutorial for **GISTnet-MD** (Graph-Informed Spatiotemporal Networks for Molecular Dynamics). 

This guide will walk you through a complete, realistic experiment: distinguishing between the conformational dynamics of a Wild-Type (WT) protein and a mutant (e.g., L99A), and then using Explainable AI (xAI) to discover *why* they behave differently.

---

## 1. Preparation and Prerequisites

Before starting, ensure your conda environment is active:
```bash
conda activate gistnet-md
```

### Dataset Structure
For a rigorous experiment, you need multiple independent Molecular Dynamics replicas (groups) for each class. We will use a Leave-One-Group-Out (LOGO) cross-validation strategy.

Your data should be organized like this:
```text
MD_simulations/
├── WT/
│   ├── reference.pdb     <-- Topology (without water/ions)
│   ├── rep_1.xtc           <-- Replica 1 trajectory
│   ├── rep_2.xtc           
│   └── ... rep_5.xtc
└── L99A/
    ├── reference.pdb
    ├── rep_1.xtc
    └── ... rep_5.xtc
```

---

## Phase 1: Topological Preprocessing (`preprocess.py`)

The first step is to convert the physical coordinates of the `.xtc` trajectories into PyTorch Geometric graphs. In these graphs, nodes are amino acids, and edges represent spatial proximity.

We will use the **Center of Mass (`com`)** representation and a **10.0 Å cutoff**. We also select a specific range of residues (`1-98,100-164`) to ignore flexible, noisy tails.

```bash
mkdir -p preprocessed_graphs/WT preprocessed_graphs/L99A

# Preprocess WT
for i in {1..5}; do
  python GISTnet-MD/preprocess.py \
    --xtc MD_simulations/WT/rep_${i}.xtc \
    --pdb MD_simulations/WT/reference.pdb \
    --resid 1-98,100-164 \
    --out_dir preprocessed_graphs/WT/WT_rep_${i} \
    --prefix WT_rep_${i} \
    --representation com \
    --cutoff 10.0
done

# Preprocess L99A
for i in {1..5}; do
  python GISTnet-MD/preprocess.py \
    --xtc MD_simulations/L99A/rep_${i}.xtc \
    --pdb MD_simulations/L99A/reference.pdb \
    --resid 1-98,100-164 \
    --out_dir preprocessed_graphs/L99A/L99A_rep_${i} \
    --prefix L99A_rep_${i} \
    --representation com \
    --cutoff 10.0
done
```
**What happens here?** 
Each frame of your `.xtc` file is turned into a `.pt` tensor file. The spatial information is frozen into edges and distances.

---

## Phase 2: Spatiotemporal Training (`train.py`)

Now we train the GNN. To prevent data leakage (the network memorizing specific initial velocities of a replica), we use a **Leave-One-Group-Out** scheme. 
We train 5 separate models. For Model 1, we use Replica 1 as the validation set and Replicas 2-5 as the training set.

```bash
mkdir -p training

for i in {1..5}; do
    echo "Training Fold ${i}..."
    python GISTnet-MD/train.py \
     --data_class_dirs preprocessed_graphs/WT/ preprocessed_graphs/L99A/ \
     --data_class_labels WT L99A \
     --out_dir training/valrep_${i}_results \
     --window 10 \
     --val_groups ${i}
done
```

**What happens here?**
The network learns the structural features distinguishing WT from L99A. The outputs in `training/valrep_1_results/` will include training convergence plots and the frozen model weights (`best_linear_model.pt`).

---

## Phase 3: Explainable AI (`xai_ig_pipeline.py`)

Training an AI is not enough; we need to understand its physical reasoning. 
We will use **Integrated Gradients (IG)**. This algorithm calculates a positive/negative "importance score" for every amino acid and every interaction, identifying the structural determinants.

```bash
python GISTnet-MD/xai_ig_pipeline.py \
  --model_dirs training/valrep_* \
  --out_dir ./analysis_IG \
  --ig_steps 15 \
  --N_baseline_medoids 5
```

**What happens here?**
The script loops through all 5 cross-validation models. It calculates "Expected Gradients" by extracting 5 representative structural states (Medoids, via `--N_baseline_medoids 5`) from the training set, and uses them as a baseline to figure out which interactions are unique to the WT or the Mutant.
The results are saved as `.gml` graph files and `.dat/.csv` numerical matrices.

---

## Phase 4: Structural Visualization (`xai_pymol_generator.py`)

Looking at raw matrices is difficult. We will aggregate the mathematical saliency scores from all the frames and project them directly onto a 3D PDB structure using PyMOL.

```bash
python GISTnet-MD/xai_pymol_generator.py \
  --ig_results_dir ./analysis_IG \
  --pdb_template MD_simulations/WT/reference.pdb \
  --out_dir ./pymol_viz \
  --resid 1-98,100-164 \
  --confidence_cut 0.8 \
  --top_edges 50 100 250
```

**What happens here?**
1. We filter out any frames where the AI was less than 80% confident (`--confidence_cut 0.8`).
2. We aggregate the scores to find the Mean and the 95th Percentile structural drivers.
3. We generate a Master PyMOL Script.

### Viewing the Results in PyMOL
Open your terminal, launch PyMOL, and execute the generated script:
```bash
pymol
# Inside PyMOL's command line:
@./pymol_viz/global_view_xai.pml
```
You will see the protein colored from **Magenta (Negative/Penalty)** to **Green (Positive/Hallmark)**, with 3D cylinders connecting the most important residues.

---
**Congratulations!** You have successfully completed an end-to-end analysis using GISTnet-MD. For deeper dives into specific parameters, refer to the individual guides in the `docs/` folder.
