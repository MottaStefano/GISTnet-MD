# Train.py - User Guide

The `train.py` script is the core engine of the SpatioTemporal GNN pipeline. Its primary role is to train a contrastive AI model on the spatial graphs generated during the preprocessing step. It identifies defining structural patterns to differentiate between dynamic states (e.g., Apo vs. Holo conformations) and concludes with a complete interpretability (xAI) phase to validate the results.

---

## 📁 Folder Organization (Crucial)

Before running the training script, your preprocessed PyG `.pt` data must be strictly organized. The script dynamically reads classes and groups based on folder morphology. 

To conduct independent validation (Leave-One-Group-Out or LORO), the script needs to know which simulation groups belong to which conformational class.

**Required Structure:**
```text
your_dataset/
├── class_1_folder/                 <-- E.g., Apo
│   ├── group_1/                  <-- E.g., MD1
│   │   ├── apo_frame_000000.pt
│   │   └── apo_frame_000001.pt
│   ├── group_2/                  <-- E.g., MD2
│   └── group_3/
│
└── class_2_folder/                 <-- E.g., Holo
    ├── group_1/
    ├── group_2/
    └── group_3/
```
When running the script, you will specify `--data_class_dirs your_dataset/class_1_folder your_dataset/class_2_folder` and give them explicit conceptual labels like `--data_class_labels "Apo" "Holo"`. 

Using the `--val_groups` keyword (e.g., `--val_groups 3`), you instruct the model to train exclusively on `group_1` and `group_2` for both classes, reserving `group_3` ENTIRELY unseen for the validation phase.

---

## ⚙️ The Pipeline (Under the Hood)

The script executes a complex, multi-stage pipeline:

1. **Auto-Detection**: The graph settings (Cutoff, Node Features) injected during `preprocess.py` are automatically read from the metadata in the `.pt` files. You don't need to specify them again.
2. **Windowing & Collation**: The network is "SpatioTemporal". Single frames are logically grouped into shifting time "windows" (governed by `--window`, `--stride`, `--skip`) to analyze structural evolution over time rather than static snapshots.
3. **Contrastive Training**: The network learns through a `GroupContrastiveLoss`. It groups windows from the same class together while pushing apart windows of different classes inside an abstract spatial embedding.
4. **Linear Probing**: Once the GNN learns the unsupervised "shape" of the classes, the weights are frozen. A standard simple Linear Classifier is then trained strictly on the extracted spatial embeddings to gauge how analytically separable the classes became.
5. **xAI Post-Analysis**: At the end of training, a suite of evaluation plots is immediately generated:
    * Training convergence graphs (Loss/Accuracy).
    * Global 2D UMAP projections to visually gauge class separations.
    * Statistical Confidence plots to identify outliers or uncertain frames in the validation groups.

---

## 🛠️ Execution Modes (Command Line vs Input File)

Like the preprocessing script, `train.py` can be launched purely via the command line interface (CLI).

However, as soon as the execution begins, the script dumps an exact logical snapshot of **ALL activated options** into a structured text file named `train.in` inside your requested `--out_dir`. This file contains clear visual sections and explanatory comments.

You can modify this generated text file and feed it directly back to the script for subsequent runs without writing long commands in the terminal:

```bash
# E.g., Pure CLI Execution:
python train.py --data_class_dirs data/Apo data/Holo --data_class_labels Apo Holo --val_groups 3 --epochs 50 --out_dir results_run1

# E.g., Re-running via the generated file:
python train.py --config "results_run1/train.in"

# Overwriting file settings via CLI on-the-fly:
python train.py --config "results_run1/train.in" --epochs 100
```

---

## 📋 Keywords Summary Table

Below are all the parameters accepted by `train.py`. 

**Note on Debug Flags**: The parameters marked under the "Debug Settings" category are exposed solely to verify the model architecture on a developmental level. **Users should NOT modify these flags**, as changing them alters the core topological mathematics the model is balanced upon.

| Keyword | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| **Files and Folders** | | | |
| `-c`, `--config` | `Str` | `None` | Optional text file to load configuration from formatted as `key=value`. Ex: `train.in`. |
| `--data_class_dirs` | `Str list` | `None` | Folder(s) containing processed .pt files. Specify one root folder per class. |
| `--data_class_labels` | `Str list` | `None` | Human-readable names of the classes corresponding to `data_class_dirs` (e.g., "Apo" "Holo"). |
| `--out_dir` | `Str` | `./results_hybrid` | Target folder to save model weights, `.in` config files, logs, and plots. |
| **Data Configuration** | | | |
| `--window` | `Int` | `10` | How many contiguous frames make up a sequence (time window). |
| `--window_offset` | `Int` | `None` | Spacing between two windows. Defaults to window size if omitted. |
| `--stride` | `Int` | `1` | Frame subsampling within the window itself (e.g., 2 skips every other frame). |
| `--skip` | `Int` | `0` | How many initial frames to skip entirely before starting window creation. |
| `--val_groups` | `Int list` | `[]` | Indices of the groups to sequester for Validation (Leave-One-Group-Out validation). |
| **Architecture and Basic Training** | | | |
| `--hidden_dim` | `Int` | `128` | Hidden layer dimension in message passing. |
| `--embedding_dim` | `Int` | `64` | Final pooled global embedding dimension produced by the model. |
| `--seed` | `Int` | `42` | Random seed lock for total environment reproducibility. |
| `--epochs` | `Int` | `50` | Maximum hardware training epochs for the heavy Contrastive phase. |
| `--batch_size` | `Int` | `256` | Logical global batch size simulated over accumulation. |
| `--micro_batch_size` | `Int` | `32` | Effective hardware batch size loaded into GPU VRAM per step natively. |
| `--contrastive_learning_rate` | `Float` | `1e-4` | Optimizer learning rate purely for the base Contrastive Phase. |
| `--linear_learning_rate` | `Float` | `1e-3` | Optimizer learning rate for the independent Linear Probe classifier evaluation. |
| `--linear_epochs` | `Int` | `50` | Maximum training epochs dedicated strictly to the Linear Probe classifier. |
| `--patience` | `Int` | `10` | Tolerance epochs experiencing no validation loss improvement before triggering Early Stopping. |
| **Advanced Optimization** *(Requires --advanced-help)* | | | |
| `--map_negative_name_to_exclude` | `Flag` | `False` | Exclude same-group negative examples in contrastive learning. |
| `--disable_preload_ram` | `Flag` | `False` | Disable loading files into RAM (useful for low memory nodes). |
| `--n_layers` | `Int` | `3` | Number of GNN layers (iterations over the message passing mechanism). |
| `--dropout` | `Float` | `0.0` | Dropout rate in GNN (0.0 means unused). |
| `--attn_temperature` | `Float` | `1.0` | Attention temperature for temporal pooling. |
| `--margin` | `Float` | `1.0` | Outer repulsive margin allocated inside the Contrastive Loss math function. |
| `--use_scheduler` | `Flag` | `False` | Use Cosine Annealing learning rate scheduler. |
| `--scheduler_min_lr` | `Float` | `1e-6` | Minimum LR for scheduler. |
| `--balance_classes` | `Flag` | `False` | Use WeightedRandomSampler to balance classes randomly if datasets are imbalanced. |
| `--use_checkpointing` | `Flag` | `False` | Enable gradient checkpointing to save memory. |
| `--save_embeddings` | `Flag` | `False` | Save numpy embeddings of every frame in train and val sets during post-analysis. |
| **Debug Settings** *(Requires --debug-help)* | | | *(DO NOT MODIFY IN PRODUCTION)* |
| `--shuffling` | `Str` | `none` | Level of shuffling applied to the dataset. Options: `none`, `windows`, `global`. |
| `--shuffling_set` | `Str` | `all` | Which dataset split to shuffle. Options: `all`, `training`, `validation`. |
| `--noLORO` | `Flag` | `False` | Disable Leave-One-Group-Out cross-validation and use random split. |
| `--pooling_type` | `Str` | `attention` | Temporal pooling strategy. Options: `attention`, `mean`. |
| `--temporal_setup` | `Str` | `cnn` | Module used to process the temporal sequence. Options: `cnn`, `attention`. |
| `--pooling_activation` | `Str` | `softmax` | Activation function for temporal attention. Options: `softmax`, `sigmoid`. |
| `--contrastive` | `Str` | `group_aware` | Type of contrastive learning loss. Options: `group_aware`, `base`, `hard_negative`, `none`. |

---

## 📂 Output Directory Structure

Once the execution finishes successfully, your designated `--out_dir` (default: `./results_hybrid`) will be populated. 

Inside the output folder, you will find:

```text
results_hybrid/
│
├── train.in                          <-- Reusable configuration textual file dump.
├── config.json                       <-- Internal JSON configuration dump for xAI scripts.
├── train_run.log                     <-- Human-readable text log of the full command.
│
├── best_model.pt                     <-- The frozen weights of the SpatioTemporal GNN.
├── best_linear_model.pt              <-- The weights of the final Linear Probe evaluator.
│
├── training_history_contrastive.csv  <-- Epoch-by-epoch loss metrics for the GNN.
├── training_history_linear.csv       <-- Epoch-by-epoch loss metrics for the Probe.
├── training_convergence.png          <-- A 3-panel plot visually graphing the CSV histories.
│
├── umap_train_val_split_all_...png   <-- 2D topological mapping of the GNN's final spatial embeddings.
│
└── confidence_analysis/
    ├── training_{class}_{MD}_conf.png 
    ├── training_{class}_{MD}_data.csv 
    ├── validation_{class}_{MD}_conf.png 
    └── validation_{class}_{MD}_data.csv 
```

**Understanding `confidence_analysis/`:**
This subfolder contains the definitive evaluation metric produced by the Linear Probe.
- `*_data.csv`: A frame-by-frame breakdown containing the true label and the *Confidence Score* (Probability 0.0 to 1.0) the network assigned to that trajectory frame.
- `*_conf.png`: A scatter plot visual representation over time of the CSV probabilities. A steady flat line at index 1.0 means the network flawlessly recognized the protein's conformation at that moment. Plunging spikes towards 0.0 indicate instances where the protein severely lost its shape or the network became highly uncertain.
