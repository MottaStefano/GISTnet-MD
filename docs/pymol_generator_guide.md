# xAI PyMOL Visualization Generator - User Guide

The `xai_pymol_generator.py` script is the final, automated visualization component of the SpatioTemporal GNN pipeline. 

Its primary purpose is to aggregate the massive amounts of graph data (`.gml` files) generated during the Integrated Gradients step and project them back into the 3D space. It creates ready-to-use PyMOL scripts that map the Artificial Intelligence's learned "importance" directly onto the physical structure of your protein.

---

## ⚙️ The Pipeline & Statistical Aggregation

Because a molecular dynamics trajectory contains thousands of frames, and you might have analyzed multiple groups, looking at a single frame's saliency is often too noisy. This script performs a **Global Consensus Aggregation**.

1. **Filtering**: The script sweeps through your `--ig_results_dir` hunting for `.gml` files. It automatically discards frames where the network was "uncertain" (e.g., if you set `--confidence_cut 0.8`, it throws away any frame where the network was less than 80% sure).
   
2. **True Class Alignment**: It extracts the numerical saliency of every node (residues) and edge (interactions) from the surviving frames.

3. **Aggregation Strategies**: The script calculates two distinct statistical realities and writes them to your output folder:
   * **The Mean (Average)**: It computes the mathematical average importance of a residue/edge across all valid frames. This highlights *stable, consistent* structural drivers that define the class.
   * **The Percentile (e.g., 95th)**: It calculates the peak importance values (ignoring the top 5% extreme outliers). This is incredibly useful for capturing *rare but highly critical* transient interactions that might get washed out by a flat mathematical average.

4. **3D Projection**: The aggregated metrics are then forcibly injected into the B-Factor column of a PDB template and drawn as 3D cylinders (edges) within PyMOL.

---

## 🎨 The Color Scale (Magenta - White - Green)

The generator uses a custom divergent color scale strictly clamped between `-100` and `+100` to represent the network's mathematical gradient.
- **Magenta (-100)**: Strong *Negative* Saliency. The network actively penalized this structure (e.g., an abnormal clash or an antagonistic shape).
- **White (0)**: Structural Neutrality. This region did not uniquely contribute to the network's decision.
- **Green (+100)**: Strong *Positive* Saliency. The network identified this geometric configuration as the absolute defining hallmark of that structural class.

---

## 📂 Output Directory Ecosystem & Files

When you run the script, it populates your `--out_dir` (default: `./pymol_global_viz`) with a synchronized ecosystem of files. It generates a global overview across all data, but also creates class-specific subdirectories so you can isolate the features belonging exclusively to the Wild-Type or the Mutant.

```text
pymol_global_viz/
│
├── pymol_generator_config.txt       <-- Reusable configuration dump.
│
├── global_edges_stats_mean.csv      <-- Raw tabular data of all aggregated edges.
│
├── global_nodes_mean.pdb            <-- PDB Template with Mean Saliency inside B-Factors.
├── global_nodes_p95.pdb             <-- PDB Template with 95th Percentile Saliency inside B-Factors.
│
├── global_draw_edges_top50.py       <-- PyMOL script drawing the Top 50 strongest Mean edges.
├── global_draw_edges_top100.py      <-- PyMOL script drawing the Top 100 strongest Mean edges.
├── global_draw_edges_...            <-- (And so on depending on --top_edges).
│
└── global_view_xai.pml              <-- THE MASTER SCRIPT (Double click or run this in PyMOL).
```

### Understanding the Multi-Class Outputs
The Global Level (`global_*`): These files aggregate the absolute importance values across all simulated states. This gives you an immediate bird's-eye view of where the network focused its attention globally during the cross-validation.

The Class-Specific Subfolders (`Val_WT/`, `Val_L99A/`, etc.): The script automatically detects your data classes and splits the results into dedicated subdirectories. Inside these, the saliency scores and edges are isolated and recalculated only for the frames belonging to that specific class.

### How to visualize the results
You only need to open PyMOL and run the Master Script: `@pymol_global_viz/global_view_xai.pml`

The Master Script is intelligent:
- It automatically loads the `Mean Saliency PDB`, applies the correct Magenta-White-Green spectrum, and generates a color legend.
- It loads the `Percentile PDB` in the background (hidden by default).
- It executes *all* the various `draw_edges_topN.py` scripts, but defaults to only turning on the visual layer for the cleanly cut `Top 50` edges to prevent immediate visual clutter. You can toggle the other denser top-edge layers directly from the PyMOL GUI panel.

---

## 🛠️ Execution Modes (Command Line vs Input File)

Like the rest of the pipeline, the script supports CLI arguments but also generates an exact snapshot file `pymol_generator_config.txt`.

By editing this generated text file, you can rapidly respawn PyMOL scenes with different settings without re-typing terminal commands.

```bash
# E.g., Pure CLI Execution:
python xai_pymol_generator.py --ig_results_dir xai_ig_results/valrep_1 --pdb_template data/protein.pdb

# E.g., Re-running via the generated file:
python xai_pymol_generator.py --config "pymol_global_viz/pymol_generator_config.txt"

# Selectively overwriting loaded parameters (e.g., increasing the confidence filter):
python xai_pymol_generator.py --config "pymol_global_viz/pymol_generator_config.txt" --confidence_cut 0.95
```

---

## 📋 Keywords Summary Table

Below is the complete list of parameters parsed by the PyMOL Generator:

| Keyword | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| **Input and Output** | | | |
| `-c`, `--config` | `Str` | `None` | Optional text file to load configuration from, formatted as `key=value`. |
| `--ig_results_dir` | `Str` | `None` | Main directory containing the IG results. The script performs a recursive search inside it specifically looking for the raw `.gml` files. |
| `--pdb_template` | `Str` | `None` | A true biological PDB file serving as a physical 3D skeleton. The script extracts the structural coordinates from this template to anchor the AI gradients upon. |
| `--out_dir` | `Str` | `./pymol_global_viz` | Export folder where the aggregated PDBs and PyMOL Master scripts will be dumped. |
| **Data Filtering & Modes** | | | |
| `--confidence_cut` | `Float` | `0.0` | Minimum prediction confidence (0.0 to 1.0) required to include a graph. Drops any sample where the AI's internal certainty probability fell below this threshold. |
| `--resid` | `Str` | `None` | Selection by ordinal index (0-based). Ex: `1-50, 70-100`. Only computes visualization statistics for these specific amino acid positions. |
| `--resnum` | `Str` | `None` | Selection by Sequence ID (PDB). Ex: `15, 20-30`. |
| `--percentile` | `Float` | `95.0` | Value for the percentile calculation (e.g. 95, 99). Protects the visualization from rare single-frame extreme value spikes. |
| **Edge Visualization**| | | |
| `--top_edges` | `Int list` | `50 100 250 500 1000` | List of top N edges to visualize. The script generates an independent Python-PyMOL layer (3D connecting cylinders) for every single numeric value provided here. |
