import os
import sys
import argparse
import datetime

# --- CLI PARSER DEFINITION ---
# Placed here to avoid heavy imports when just requesting help
def get_parser(show_debug=None):
    # Detects if --debug-help was called to show hidden arguments
    if show_debug is None:
        show_debug = '--debug-help' in sys.argv

    parser = argparse.ArgumentParser(
        description="MD Trajectory Preprocessing: Converts .xtc/.pdb files into PyTorch Geometric graphs for GNN models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False # Disable default help to insert our custom flags
    )

    help_group = parser.add_argument_group('Help')
    help_group.add_argument('-h', '--help', action='help', help="Show this help message and exit.")
    help_group.add_argument('--debug-help', action='store_true', help=argparse.SUPPRESS)

    io_group = parser.add_argument_group('Files and Directories', 'Input management and output destination.')
    io_group.add_argument("-c", "--config", type=str, help="Optional text file to load configuration from.")
    io_group.add_argument("--xtc", required=False, help="Path to the MD trajectory file (.xtc).")
    io_group.add_argument("--pdb", required=False, help="Path to the topology/structure file (.pdb).")
    io_group.add_argument("--out_dir", required=False, help="Folder where to save the generated .pt files.")
    io_group.add_argument("--prefix", required=False, help="Prefix for naming output files (e.g., 'apo').")

    sel_group = parser.add_argument_group('Residue Selection', 'Choose which residues to include in the graph.')
    sel_mutex = sel_group.add_mutually_exclusive_group()
    sel_mutex.add_argument("--resid", default=None, help="Selection by ordinal index (0-based). Ex: '1-100, 150'. By default, it uses the whole protein.")
    sel_mutex.add_argument("--resnum", default=None, help="Selection by Sequence ID (PDB). Ex: '15, 20-30'. By default, it uses the whole protein.")

    gnn_group = parser.add_argument_group('Topological Parameters', 'Basic graph settings.')
    gnn_group.add_argument("--representation", choices=['ca', 'cb', 'com'], default='ca',
                           help="Which coordinate to use for the node: 'ca' (C-alpha), 'cb' (C-beta), or 'com' (Center of mass).")
    gnn_group.add_argument("--cutoff", type=float, default=10.0,
                           help="Cutoff radius (in Angstroms) to connect two nodes via an edge.")
    gnn_group.add_argument("--max_neighbors", type=int, default=100,
                           help="Maximum number of edges per node to prevent computational bottlenecks in dense structures.")

    # --- HIDDEN ARGUMENTS (Visible only with --debug-help) ---
    adv_help_node_dihe = "Calculation of dihedral angles as features. Options: 'none', 'backbone', 'full'." if show_debug else argparse.SUPPRESS
    adv_help_workers = "Force the number of multiprocessing workers. Default: N_cores - 1." if show_debug else argparse.SUPPRESS
    adv_help_chunk = "How many frames to read into memory at once from the XTC file." if show_debug else argparse.SUPPRESS

    adv_group = parser.add_argument_group('Advanced Options and Debug', 'Modify only if strictly necessary.') if show_debug else parser
    adv_group.add_argument("--node_dihe", choices=['none', 'backbone', 'full'], default='none', help=adv_help_node_dihe)
    adv_group.add_argument("--workers", type=int, default=None, help=adv_help_workers)
    adv_group.add_argument("--chunk_size", type=int, default=1000, help=adv_help_chunk)

    return parser, show_debug

# --- FAST HELP INTERCEPTOR ---
if __name__ == "__main__":
    if any(arg in sys.argv for arg in ['-h', '--help', '--debug-help']):
        parser, show_debug = get_parser()
        if show_debug:
            parser.print_help()
        else:
            parser.parse_args()
        sys.exit(0)


# ==============================================================================
# --- HEAVY IMPORTS ---
# These are only loaded if the user actually wants to run the script.
# ==============================================================================
import torch
import mdtraj as md
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch.multiprocessing import Pool, set_start_method
from tqdm import tqdm
from core.utils import parse_with_config, save_and_reload_config
from typing import List, Tuple, Dict, Optional, Set

# --- CONSTANTS ---
STANDARD_RESIDUES = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
RESIDUE_TO_ID = {res: i for i, res in enumerate(STANDARD_RESIDUES)}

NON_STANDARD_RESIDUE_MAP = {
    'HID': 'HIS', 'HIE': 'HIS', 'HIP': 'HIS',
    'HSD': 'HIS', 'HSE': 'HIS', 'HSP': 'HIS',
    'CYX': 'CYS', 'CYM': 'CYS',
    'ASH': 'ASP', 'GLH': 'GLU',
    'LYN': 'LYS'
}

# 1-Letter Code Dictionary for printing the sequence
AA_3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    # Map non-standard to standard 1-letter codes
    'HID': 'H', 'HIE': 'H', 'HIP': 'H', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H',
    'CYX': 'C', 'CYM': 'C', 'ASH': 'D', 'GLH': 'E', 'LYN': 'K'
}

# --- UTILITY FUNCTIONS (STATIC/HELPER) ---

def parse_selection_string(selection_str: str) -> List[int]:
    numbers = set()
    parts = selection_str.split(',')
    for part in parts:
        part = part.strip()
        if not part: continue
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                numbers.update(range(start, end + 1))
            except ValueError:
                raise ValueError(f"Invalid range format: '{part}'. Use 'start-end' format.")
        else:
            try:
                numbers.add(int(part))
            except ValueError:
                raise ValueError(f"Invalid residue number: '{part}'.")
    return sorted(list(numbers))

def get_static_node_indices(residues: list) -> torch.Tensor:
    num_residues = len(residues)
    unknown_id = len(STANDARD_RESIDUES)
    indices = torch.full((num_residues,), fill_value=unknown_id, dtype=torch.long)
    for i, residue in enumerate(residues):
        res_name = residue.name.upper()
        if res_name in NON_STANDARD_RESIDUE_MAP:
            res_name = NON_STANDARD_RESIDUE_MAP[res_name]
        idx = RESIDUE_TO_ID.get(res_name, unknown_id)
        indices[i] = idx
    return indices

def compute_dihedrals_chunk(traj: md.Trajectory, selected_res_indices: List[int], mode: str) -> np.ndarray:
    if mode == 'none': return None
    n_frames = traj.n_frames
    n_res = len(selected_res_indices)
    topo_idx_to_graph_idx = {r_idx: i for i, r_idx in enumerate(selected_res_indices)}

    tasks = []
    if mode in ['backbone', 'full']:
        tasks.append((md.compute_phi, 'phi'))
        tasks.append((md.compute_psi, 'psi'))
    if mode == 'full':
        tasks.append((md.compute_chi1, 'chi1'))
        tasks.append((md.compute_chi2, 'chi2'))

    n_feats_per_angle = 2
    total_feats = len(tasks) * n_feats_per_angle
    dihedral_features = np.zeros((n_frames, n_res, total_feats), dtype=np.float32)

    feat_offset = 0
    for compute_func, name in tasks:
        try:
            indices, angles = compute_func(traj)
            atom_idx_for_res = indices[:, 1]
            for i, atom_idx in enumerate(atom_idx_for_res):
                res_obj = traj.topology.atom(atom_idx).residue
                res_idx = res_obj.index
                if res_idx in topo_idx_to_graph_idx:
                    graph_idx = topo_idx_to_graph_idx[res_idx]
                    raw_angle = angles[:, i]
                    dihedral_features[:, graph_idx, feat_offset] = np.sin(raw_angle)
                    dihedral_features[:, graph_idx, feat_offset + 1] = np.cos(raw_angle)
        except Exception:
            pass
        feat_offset += n_feats_per_angle
    return dihedral_features

def compute_com_chunk(chunk: md.Trajectory, res_indices_map: List[int]) -> np.ndarray:
    n_frames = chunk.n_frames
    n_residues = len(res_indices_map)
    coms = np.zeros((n_frames, n_residues, 3), dtype=np.float32)
    masses = np.array([a.element.mass if a.element else 12.0 for a in chunk.topology.atoms])
    xyz = chunk.xyz
    for i, res in enumerate(chunk.topology.residues):
        atom_indices = [atom.index for atom in res.atoms]
        res_coords = xyz[:, atom_indices, :]
        res_masses = masses[atom_indices][None, :, None]
        total_mass = np.sum(masses[atom_indices])
        if total_mass == 0: continue
        weighted_coords = np.sum(res_coords * res_masses, axis=1)
        coms[:, i, :] = weighted_coords / total_mass
    return coms

# --- WORKER FUNCTION (Must remain Top-Level for Pickling) ---
def process_frame_task(args):
    coords_nm, static_indices, dihe_feats, frame_idx, out_dir, prefix, cutoff, max_k, representation, node_dihe = args
    try:
        pos = torch.tensor(coords_nm * 10.0, dtype=torch.float) # Angstroms
        edge_index = radius_graph(pos, r=cutoff, batch=None, loop=False, max_num_neighbors=max_k)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)

        data = Data(
            x=static_indices,
            pos=pos,
            edge_index=edge_index,
            edge_attr=dist,
            frame_index=frame_idx
        )
        if dihe_feats is not None:
            data.x_dihe = torch.tensor(dihe_feats, dtype=torch.float)

        # METADATA EMBEDDING: Save these global attributes inside the Graph object
        data.cutoff = cutoff
        data.representation = representation
        data.node_dihe = node_dihe

        filename = f"{prefix}_frame_{frame_idx:06d}.pt"
        torch.save(data, os.path.join(out_dir, filename))
        return None
    except Exception as e:
        return f"Error frame {frame_idx}: {e}"

# --- PREPROCESSOR CLASS ---

class MDGraphPreprocessor:
    """
    Class managing the conversion logic from Trajectory to PyG Graphs.
    """
    def __init__(self, xtc_path, pdb_path, out_dir, prefix,
                 selection_mode='all', selection_str=None,
                 representation='ca', dihedral_mode='none',
                 cutoff=10.0, max_neighbors=100,
                 chunk_size=1000, workers=None):

        self.xtc_path = xtc_path
        self.pdb_path = pdb_path
        self.out_dir = out_dir
        self.prefix = os.path.basename(prefix) # <-- FIX: Previene errori di percorso estraendo solo il nome finale
        self.selection_mode = selection_mode
        self.selection_str = selection_str
        self.representation = representation
        self.dihedral_mode = dihedral_mode
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.chunk_size = chunk_size
        self.workers = workers if workers else max(1, os.cpu_count() - 1)

        os.makedirs(self.out_dir, exist_ok=True)

    def _get_selection_indices(self, topology):
        total_residues = topology.n_residues
        sel_indices = []

        if self.selection_mode == 'resid':
            requested_ordinals = parse_selection_string(self.selection_str)
            for ordinal in requested_ordinals:
                idx = ordinal - 1
                if 0 <= idx < total_residues:
                    sel_indices.append(idx)
        elif self.selection_mode == 'resnum':
            requested_resnums = set(parse_selection_string(self.selection_str))
            # Only append if the resSeq ACTUALLY exists in the topology
            for res in topology.residues:
                if res.resSeq in requested_resnums:
                    sel_indices.append(res.index)
            sel_indices.sort()
        else: # 'all'
            sel_indices = [r.index for r in topology.residues if r.is_protein]

        if not sel_indices:
            raise ValueError("Empty residue selection. Check your input or the topology file.")
        return sel_indices

    def run(self):
        print(f"\n--- MD Graph Preprocessor ---")
        print(f"Traj: {self.xtc_path} | Top: {self.pdb_path}")

        # 1. Load Topology & Selection
        full_pdb = md.load_pdb(self.pdb_path)
        sel_indices = self._get_selection_indices(full_pdb.topology)
        selected_residues_obj = [full_pdb.topology.residue(i) for i in sel_indices]

        # Extract extensive topology information
        num_nodes = len(sel_indices)
        first_res = selected_residues_obj[0]
        last_res = selected_residues_obj[-1]

        # Convert to 1-letter sequence. Use 'X' for unknown entities.
        seq_1_letter = "".join([AA_3_TO_1.get(r.name.upper(), 'X') for r in selected_residues_obj])

        print(f"\n--- Topology & Selection Info ---")
        print(f"Target Nodes (Actual found) : {num_nodes} residues")
        print(f"First Residue               : {first_res.name} {first_res.resSeq}")
        print(f"Last Residue                : {last_res.name} {last_res.resSeq}")
        print(f"Sequence Extract            : {seq_1_letter}")
        print(f"---------------------------------\n")

        # Append topology infos to the log file explicitly
        log_path = os.path.join(self.out_dir, "preprocess_run.log")
        if os.path.exists(log_path):
            with open(log_path, 'a') as f:
                f.write("\n--- TOPOLOGY & SELECTION INFO ---\n")
                f.write(f"Target Nodes (Actual found) : {num_nodes} residues\n")
                f.write(f"First Residue               : {first_res.name} {first_res.resSeq}\n")
                f.write(f"Last Residue                : {last_res.name} {last_res.resSeq}\n")
                f.write(f"Sequence Extract            : {seq_1_letter}\n")
                f.write("="*70 + "\n")

        # 2. Static Features
        static_node_indices = get_static_node_indices(selected_residues_obj)

        # 3. Define Atoms to Load
        atoms_to_load = []
        if self.dihedral_mode != 'none' or self.representation in ['com', 'cb']:
            for r in selected_residues_obj:
                atoms_to_load.extend([a.index for a in r.atoms])
        else: # Only CA
            for r in selected_residues_obj:
                ca = next((a for a in r.atoms if a.name == 'CA'), None)
                if ca: atoms_to_load.append(ca.index)
                else: atoms_to_load.append(r.atom(0).index)

        # 4. Processing Loop
        global_frame_counter = 0
        iterator = md.iterload(self.xtc_path, top=self.pdb_path, atom_indices=atoms_to_load, chunk=self.chunk_size)

        try: set_start_method('spawn')
        except RuntimeError: pass

        with Pool(processes=self.workers) as pool:
            for chunk_idx, chunk in enumerate(iterator):
                current_n_frames = chunk.n_frames

                # A. Coords
                if self.representation == 'com':
                    coords_chunk = compute_com_chunk(chunk, sel_indices)
                elif self.representation == 'ca':
                    if self.dihedral_mode == 'none' and chunk.n_atoms == len(sel_indices):
                         coords_chunk = chunk.xyz
                    else:
                        coords_chunk = np.zeros((current_n_frames, len(sel_indices), 3), dtype=np.float32)
                        for i, r in enumerate(chunk.topology.residues):
                            ca = next((a for a in r.atoms if a.name == 'CA'), r.atom(0))
                            coords_chunk[:, i, :] = chunk.xyz[:, ca.index, :]
                elif self.representation == 'cb':
                    coords_chunk = np.zeros((current_n_frames, len(sel_indices), 3), dtype=np.float32)
                    for i, r in enumerate(chunk.topology.residues):
                        atom = next((a for a in r.atoms if a.name == 'CB'), None)
                        if not atom: atom = next((a for a in r.atoms if a.name == 'CA'), r.atom(0))
                        coords_chunk[:, i, :] = chunk.xyz[:, atom.index, :]

                # B. Dihedrals
                dihedrals_chunk = None
                if self.dihedral_mode != 'none':
                    dihedrals_chunk = compute_dihedrals_chunk(chunk, sel_indices, self.dihedral_mode)

                # C. Tasks
                tasks = []
                for i in range(current_n_frames):
                    tasks.append((
                        coords_chunk[i],
                        static_node_indices,
                        dihedrals_chunk[i] if dihedrals_chunk is not None else None,
                        global_frame_counter + i,
                        self.out_dir,
                        self.prefix,
                        self.cutoff,
                        self.max_neighbors,
                        self.representation,
                        self.dihedral_mode
                    ))

                # D. Execute
                results = list(pool.imap_unordered(process_frame_task, tasks))

                # --- FIX: Controllo Errori Silenti del Multiprocessing ---
                errors = [res for res in results if res is not None]
                log_path = os.path.join(self.out_dir, "preprocess_run.log")
                with open(log_path, 'a') as f:
                    if errors:
                        err_msg = f"\n[WARNING] {len(errors)} frames generated an error in Chunk {chunk_idx+1}!\nError Example: {errors[0]}\n"
                        print(err_msg)
                        f.write(err_msg)
                        f.write("--- Detailed Error List ---\n")
                        for err in errors:
                            f.write(f"{err}\n")
                        f.write("---------------------------\n")
    
                    global_frame_counter += current_n_frames
                    succ_msg = f"Chunk {chunk_idx+1} completed. Total frames processed: {global_frame_counter}"
                    print(succ_msg)
                    f.write(f"{succ_msg}\n")

def create_log_table(args):
    """Generates and saves a detailed log file with all parameters."""
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "preprocess_run.log")

    with open(log_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(" MD-GNN PREPROCESSING LOG\n")
        f.write("="*70 + "\n")
        f.write(f"Execution Date  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command         : {' '.join(sys.argv)}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Argument':<25} | {'Set Value':<40}\n")
        f.write("-" * 70 + "\n")
        for arg, value in vars(args).items():
            if arg != 'debug_help':
                f.write(f"--{arg:<23} | {str(value):<40}\n")
        f.write("="*70 + "\n")
    print(f"Detailed log saved in: {log_path}")

def main():
    parser, _ = get_parser()
    args = parse_with_config(parser)

    # Validation per args ora svincolati dal required=True di argparse
    if not all([getattr(args, 'xtc', None), getattr(args, 'pdb', None), getattr(args, 'out_dir', None), getattr(args, 'prefix', None)]):
        parser.error("The following arguments are required: --xtc, --pdb, --out_dir, --prefix")

    os.makedirs(args.out_dir, exist_ok=True)

    config_out_path = os.path.join(args.out_dir, "preprocess.in")
    full_parser, _ = get_parser(show_debug=True)
    args = save_and_reload_config(args, config_out_path, parser=full_parser)

    # Initialize selection resolution
    mode = 'all'
    sel_str = None
    if args.resid:
        mode = 'resid'
        sel_str = args.resid
    elif args.resnum:
        mode = 'resnum'
        sel_str = args.resnum

    # Generate the log table before starting heavy computations
    create_log_table(args)

    processor = MDGraphPreprocessor(
        xtc_path=args.xtc, pdb_path=args.pdb, out_dir=args.out_dir, prefix=args.prefix,
        selection_mode=mode, selection_str=sel_str,
        representation=args.representation, dihedral_mode=args.node_dihe,
        cutoff=args.cutoff, max_neighbors=args.max_neighbors,
        chunk_size=args.chunk_size, workers=args.workers
    )
    processor.run()

if __name__ == "__main__":
    main()
