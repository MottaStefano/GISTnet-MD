# Preprocess.py - User Guide

The `preprocess.py` script is the fundamental first step of the pipeline. Its purpose is to take a raw Molecular Dynamics trajectory (`.xtc` file) along with its original structure or topology (`.pdb` file) and convert them into a series of **Spatial Graphs** compatible with PyTorch Geometric (`.pt`) Artificial Intelligence models.

Each individual time frame of the simulation is transformed into a dedicated graph where the **nodes** represent the residues (amino acids) and the **edges** represent physical proximity interactions within a certain threshold radius.

---

## ⚙️ The Pipeline (Under the Hood)

When `preprocess.py` is launched, it sequentially performs the following operations:

1. **Topology Loading and Filtering**: It loads the provided PDB file and isolates only the protein (ignoring ions, solvent, etc.). If a manual `selection` string is provided, the script will physically isolate only the requested amino acid blocks. Residues are automatically mapped to standardized scalar indices.

2. **Structural Node Construction**: Depending on the user's choice (Atomic Representation), a node will spatially coincide with its C-Alpha (`ca`), or C-Beta (`cb`), or fall exactly in the mass-weighted Center of Mass (`com`) of the individual residue.

3. **Geometric Features Injection**: If requested, the script computes in parallel the dihedral angles of the entire chain (Only backbone angles `Phi/Psi`, or all four `Chi1/Chi2` etc.) and stores them as features associated with their respective nodes.

4. **Parallel Multi-Core Processing (Chunking)**: To prevent processing the entire trajectory from saturating the local machine's RAM, the script iteratively loads the files into predefined chunks (e.g., `1000 frames` at a time). A process pool spawns several operational processes distributing the computational load:
    * It builds the **Edges**: Dynamically connects two physical nodes via an edge only if their spatial distance frame by frame is less than the `cutoff` (geometric threshold in Ångströms).
    * It builds the **Distances**: Calculates the absolute angstrom distance to be deposited on the edge as an essential feature for the Artificial Intelligence package.

5. **Saving Outcomes and Configurations**: The script packages the outputs converting them into the `PyG Data` tensor format. Besides the `.pt` files, it exports the exact configuration of the keywords used into the **`preprocess.in`** file along with a summary log.

---

## 🛠️ Execution Modes (Command Line vs Input File)

You can launch the script originally by manually passing all arguments via the command line (CLI).

However, once the execution starts, the script will visually generate a file named `preprocess.in` inside the selected output directory. This file represents the exact logical dump of **ALL activated options** (both those provided by you and the intrinsic defaults set upstream by the developer), organized with clear dividing text comments.

You can freely edit this generated pure text file (e.g., changing the `cutoff` parameter from 10.0 to 12.0) and pass it directly to the script without having to re-enter every single option in the terminal shell:

```bash
# Example of pure execution (CLI):
python preprocess.py --xtc "traj.xtc" --pdb "topo.pdb" --out_dir "data/apo" --prefix "apo_sim" --cutoff 10.0

# Example of execution cloned from the previously generated file:
python preprocess.py --config "data/apo/preprocess.in"

# You can test cascaded overwrites from the terminal (CLI has the highest overwrite priority)
python preprocess.py --config "data/apo/preprocess.in" --cutoff 15.0
```

---

## 📋 Keywords Summary Table

Below are all the parameters accepted by the system. If you wish to consult or verify the "under the hood" features from the terminal, you can also print the hidden fraction of the commands by running `python preprocess.py --debug-help`

| Keyword | Type | Default | Description (Function) |
| :--- | :---: | :---: | :--- |
| **Input and Basic File Management** | | | |
| `-c`, `--config` | `Str` | `None` | Optional text file to load configuration from formatted as `key=value`. Ex: `preprocess.in` |
| `--xtc` | `Str` | *Required* | Path to the MD trajectory file (.xtc). |
| `--pdb` | `Str` | *Required* | Path to the topology/structure file (.pdb). |
| `--out_dir` | `Str` | *Required* | Folder where to save the generated `.pt` files. |
| `--prefix` | `Str` | *Required* | Prefix for naming output files (e.g., 'apo' will originate `apo_frame_000000.pt`). |
| **Residue Selection Mask** | | | *Filtering Modes: Mutually exclusive, you can only pick 1 at a time.* |
| `--resid` | `Str` | `None` | Selection by ordinal index (0-based internal PDB index). Masking example: "1-100, 150". By default, it globally packages the entire PDB system. |
| `--resnum` | `Str` | `None` | Selection by Sequence ID (PDB). Logical string example "15, 20-30". By default, it considers the whole protein. |
| **Primary GNN Topology** | | | |
| `--representation` | `Str` | `ca` | Which coordinate to use for the node: `ca` (C-alpha), `cb` (C-beta), or `com` (Center of mass). |
| `--cutoff` | `Float` | `10.0` | Cutoff radius (in Angstroms) to connect two nodes via an edge. If the 3D distance between the two residues exceeds or equals the cutoff, there will be no contact. |
| `--max_neighbors` | `Int` | `100` | Maximum number of edges per node to prevent computational bottlenecks in dense structures. |
| **Advanced System Debug** | | | *(Visible only with the --debug-help flag enabled on CLI)* |
| `--node_dihe` | `Str` | `none` | Calculation of dihedral angles as features. Options: `none`, `backbone` (phi/psi only), and `full` (also injecting chi1/chi2 external ramification rotations). |
| `--workers` | `Int` | `None` | Force the number of multiprocessing workers. Default applied if omitted: Total Virtual CPUs - 1. |
| `--chunk_size` | `Int` | `1000` | How many frames to read into memory at once from the XTC file. Delimits the logical number of frames per single wave of block reads. |
