# xAI Integrated Gradients Pipeline - User Guide

The `xai_ig_pipeline.py` script executes the core of the explainable Artificial Intelligence (xAI) module. Its purpose is to crack open the "black box" of the previously trained GNN model to understand *why* it makes certain structural classifications (e.g., why a sequence is considered "WT" instead of "L99A").

It achieves this by applying the **Integrated Gradients (IG)** mathematical algorithm on the test/validation trajectory groups, calculating a numerical "importance/saliency score" for every single node and edge in the protein graph over time.

---

## 🧠 The Theory: Baseline, Interpolation, and Ghost Edges

Integrated Gradients works by continuously interpolating a sample from a "blank" neutral state (Baseline) up to its actual 100% real structural state, accumulating the gradients (the network's reaction) along the way.

1. **The Baselines**:
   - `thermodynamic_mean`: The script automatically sweeps your entire Training Set to compute a global "Mean Centroid Baseline". It calculates the average radial distance (RBF) for every edge that appears consistently across all the training simulations.
   - `real_background_windows`: Extracts a specific number (`--num_baselines`) of Medoid baselines from the training set, computing Expected Gradients over them.
   - `zero_edges`: A simple baseline where initial RBF distances are completely nullified.

2. **The Interpolation Steps (`--ig_steps`)**:
   During the analysis of a validation frame, the algorithm morphs the graph from the topological Baseline to the actual current geometry of the frame in *N* incremental discrete steps. Higher steps yield better integral approximations at the cost of computation time.

3. **Ghost Edges (Transient Interactions)**:
   Because the validation frame might be missing some edges (e.g., two amino acids are currently far apart) that represent a historical interaction present in the global Baseline, the script injects **"Ghost Edges"**. 
   These are edges with an initial neutral baseline distance that are mathematically faded to zero interaction. This allows the model to capture the topological "absence" of a contact as a crucial distinguishing feature.

---

## ⚙️ The Pipeline (Under the Hood)

When `xai_ig_pipeline.py` is launched, it performs these operations:

1. **Automatic Discovery**: It navigates into the provided `--model_dirs`, extracting the pre-trained weights (`best_linear_model.pt`) and automatically parsing the internal `config.json` to seamlessly reconstruct the original network's architecture.
2. **Global Centroid Extraction**: Sweeps the training dataset to build the chosen Baseline graph(s).
3. **IG Calculation**: Iterates over every frame of the sequestered validation group(s), applying the IG wrapper to extract both **Node Importance** and **Edge Importance** tensors. 
4. **Graph Generation**: Dumps the pure topological information + saliencies as standard `.gml` network files into intermediate folders.
5. **VMD Extraction**: Parses the `.gml` files and exports numerical `.csv`/`.dat` arrays to inject data into visualization dashboards like VMD or PyMOL.

---

## 🛠️ Execution Modes (Command Line vs Input File)

The script natively supports pure Command Line Interface (CLI) execution.

As soon as the script completes its initialization, it drops a human-readable text file named `ig_pipeline_config.txt` (or `.in`) directly into the `--out_dir`. This file contains the exact configuration snapshot used to trigger the Integrated Gradients algorithm.

You can modify this text file and re-feed it to the script to avoid manual typing:

```bash
# E.g., Pure CLI Execution:
python xai_ig_pipeline.py --model_dirs results_hybrid/valrep_1 results_hybrid/valrep_2 --ig_steps 15 --out_dir my_xai_folder

# E.g., Re-running via the generated file:
python xai_ig_pipeline.py --config "my_xai_folder/ig_pipeline_config.txt"

# Selectively overwriting loaded parameters directly from the terminal prompt:
python xai_ig_pipeline.py --config "my_xai_folder/ig_pipeline_config.txt" --ig_steps 50
```

---

## 📋 Keywords Summary Table

Below is the complete list of parameters parsed by the IG pipeline:

| Keyword | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| **Input and Output** | | | |
| `-c`, `--config` | `Str` | `None` | Optional text file to load configuration from, formatted as `key=value`. |
| `--model_dirs` | `Str list` | *Required* | Space-separated list of training directories containing `best_linear_model.pt` and `config.json` (e.g., `results_hybrid/valrep_1`). |
| `--out_dir` | `Str` | `./xai_ig_results` | Main output directory boundary. Final results will be automatically organized into subfolders matching their respective validation groups. |
| **Integrated Gradients Parameters** | | | |
| `--ig_steps` | `Int` | `10` | The number of discrete interpolation steps used to mathematically approximate the integral from the Baseline to the sample. |
| `--baseline` | `Str` | `thermodynamic_mean` | Baseline method. Options: `thermodynamic_mean` (mean states of classes), `zero_edges` (null initial RBFs), `real_background_windows` (Expected Gradients on Medoids). |
| `--num_baselines` | `Int` | `5` | Number of background windows (Medoids) to extract per class when using the `real_background_windows` baseline. |
| **Dashboard Export Options**| | | |
| `--remove_gmls` | `Flag` | `False` | Remove the intermediate `.gml` graph files after exporting to VMD formats. By default, they are kept as they are required by the PyMOL generator. |

---

## 📂 Output Directory Structure & VMD Data

The pipeline organizes its output automatically to respect the LORO (Leave-One-Group-Out) validation schema. If you launched the script analyzing the validation fold 1 (`valrep_1`), the structure inside your `--out_dir` will look like this:

```text
xai_ig_results/
└── valrep_1/
    ├── ig_pipeline_config.txt          <-- Reusable configuration text dump.
    ├── xai_ig_run.log                  <-- Human-readable execution log.
    │
    ├── gmls/                           <-- Raw topological extraction graphs (Required for PyMOL).
    │   ├── window_000000_WT_pred_WT_grp_3.gml
    │   ├── window_000001_WT_pred_WT_grp_3.gml
    │   └── ...
    │
    └── vmd_data/                       <-- Formatted numerical arrays for fast Dashboard injection.
        └── WT/
            ├── WT_WT_grp3_xai_spatial.dat
            └── WT_WT_grp3_xai_temporal.csv
```

### Understanding the VMD / Data Formats

The final translation phase of the script (VMD Extraction) converts the heavy, complex `.gml` networks into highly optimized mathematical arrays ready to be imported into molecular viewers (like VMD's Data Importer or custom Python notebooks).

1. **`*_xai_spatial.dat`**:
   This is a space-separated data file (matrix). 
   - **Rows**: Represent the simulation frames (Time).
   - **Columns**: Represent the internal biological residues (Amino Acids).
   - **Values**: The numeric Saliency / Importance extracted by the Integrated Gradients. If a specific index `[Frame 50, Residue 12]` has a high positive value, it means that residue was structurally crucial for the network *at that exact moment* in time. 
   - **Header**: Contains metadata to auto-initialize viewers (`# META [TotalFrames] [NumResidues] [GlobalMaxSaliency] 0`).

2. **`*_xai_temporal.csv`**:
   This tracks the global metrics of the system, collapsing spatial dimensions to focus purely on time.
   - **Window Tracks (`Window_Track_0`, `_1` etc.)**: Since the GNN analyzes physical frames within sliding "Time Windows", this tracks the overall topological saliency of the *entire window*. Overlapping windows will use different columns (tracks) to avoid data collision.
   - **Confidence**: The 0.0 to 1.0 probability that the network guessed the true class correctly in that particular frame.
   - **Anomaly Score**: Tracks severe structural outliers or energetic penalties identified by the GNN during that frame.

**Why keep the `gmls/` folder?**
The `.dat` matrix only saves *Node* importance (per-residue saliency). If you want to visualize dynamic 3D Arrows/Cylinders connecting residues (Edge Importance), that topological connection data is exclusively retained in the raw `.gml` files. The subsequent script (`xai_pymol_generator.py`) requires this directory to generate 3D PyMOL rendering scripts.
