import sys
import os
import argparse
import glob


# =============================================================================
# CLI PARSER
# =============================================================================

def get_parser(show_debug=None, show_advanced=None):
    if show_debug is None:
        show_debug = '--debug-help' in sys.argv
    if show_advanced is None:
        show_advanced = '--advanced-help' in sys.argv or show_debug

    parser = argparse.ArgumentParser(
        description="PyMOL Visualization Generator for Integrated Gradients (IG).\n"
                    "Supports two modes: 'Global Consensus' (aligns gradients to true class) "
                    "or 'Class-Specific' (filters only replicas of a specific target class).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False
    )

    help_group = parser.add_argument_group('Help')
    help_group.add_argument('-h', '--help', action='help', help="Show this help message and exit.")

    io_group = parser.add_argument_group('Input and Output', 'Define data sources and topology template.')
    io_group.add_argument("-c", "--config", type=str, help="Optional text file to load configuration from.")
    io_group.add_argument("--ig_results_dir", type=str, required=False,
                          help="Main directory containing the IG results. Recursive search for .gml files.")
    io_group.add_argument("--pdb_template", type=str, required=False,
                          help="PDB file used as a template to extract 3D coordinates for nodes (CA atoms).")
    io_group.add_argument("--out_dir", type=str, default="./pymol_global_viz",
                          help="Folder where the aggregated PDBs and PyMOL scripts will be saved.")

    filter_group = parser.add_argument_group('Data Filtering & Modes', 'Options to filter and align the analyzed graphs.')

    filter_group.add_argument("--confidence_cut", type=float, default=0.0,
                              help="Minimum prediction confidence (0.0 to 1.0) required to include a graph.")
    sel_mutex = filter_group.add_mutually_exclusive_group()
    sel_mutex.add_argument("--resid", type=str, default=None,
                              help="Selection by ordinal index (0-based). Ex: '1-50, 70-100'.")
    sel_mutex.add_argument("--resnum", type=str, default=None,
                              help="Selection by Sequence ID (PDB). Ex: '15, 20-30'.")
    filter_group.add_argument("--percentile", type=float, default=95.0,
                              help="Value for the percentile calculation (e.g. 95, 99).")

    edge_group = parser.add_argument_group('Edge Visualization', 'Settings for drawing 3D interactions.')
    edge_group.add_argument("--top_edges", nargs='+', type=int, default=[50, 100, 250, 500, 1000],
                            help="List of top N edges to visualize. A separate script is generated for each value.")

    return parser, show_debug, show_advanced

# --- EARLY HELP CHECK ---
if __name__ == "__main__":
    if any(arg in sys.argv for arg in ['-h', '--help']):
        parser, _, _ = get_parser()
        parser.print_help()
        sys.exit(0)

import numpy as np
import networkx as nx
import mdtraj as md
import pandas as pd
from tqdm import tqdm
from core.utils import parse_with_config, save_and_reload_config

# =============================================================================
# 1. MATHEMATICAL AND COLOR UTILITIES
# =============================================================================

def parse_selection_string(selection_str):
    numbers = set()
    parts = selection_str.split(',')
    for part in parts:
        part = part.strip()
        if not part: continue
        if '-' in part:
            start, end = map(int, part.split('-'))
            numbers.update(range(start, end + 1))
        else:
            numbers.add(int(part))
    return sorted(list(numbers))

def get_selection_indices(mode, selection_str, topology):
    if not selection_str: return None
    total_residues = topology.n_residues
    sel_indices = []

    if mode == 'resid':
        requested_ordinals = parse_selection_string(selection_str)
        for ordinal in requested_ordinals:
            idx = ordinal - 1
            if 0 <= idx < total_residues:
                sel_indices.append(idx)
    elif mode == 'resnum':
        requested_resnums = set(parse_selection_string(selection_str))
        for res in topology.residues:
            if res.resSeq in requested_resnums:
                sel_indices.append(res.index)
        sel_indices.sort()

    if not sel_indices:
        print("Empty residue selection. Check your input or the topology file.")
        sys.exit(1)
    return sel_indices

def normalize_min_max(value, v_min, v_max):
    if v_max - v_min < 1e-9: return 0.0
    return (value - v_min) / (v_max - v_min)

def get_color_gradient(val, max_val, low_color=(1.0, 0.0, 1.0), mid_color=(0.6, 0.6, 0.6), high_color=(0.0, 1.0, 0.0)):
    """Divergent color map: Magenta (negative) -> Grey60 (near zero) -> Green (positive)."""
    # Normalize linearly into [-1, 1] range assuming symmetric max_val
    norm_val = val / (max_val + 1e-9)
    norm_val = np.clip(norm_val, -1.0, 1.0)
    
    if norm_val < 0:
        # Interpolate between low_color and mid_color
        frac = norm_val + 1.0  # from 0 (at -1) to 1 (at 0)
        r = low_color[0] + frac * (mid_color[0] - low_color[0])
        g = low_color[1] + frac * (mid_color[1] - low_color[1])
        b = low_color[2] + frac * (mid_color[2] - low_color[2])
    else:
        # Interpolate between mid_color and high_color
        frac = norm_val  # from 0 (at 0) to 1 (at +1)
        r = mid_color[0] + frac * (high_color[0] - mid_color[0])
        g = mid_color[1] + frac * (high_color[1] - mid_color[1])
        b = mid_color[2] + frac * (high_color[2] - mid_color[2])
        
    return (r, g, b)

# =============================================================================
# 2. PARSING & TRUE CLASS ALIGNMENT
# =============================================================================

import re
import os
from tqdm import tqdm

# Pre-compiliamo le regex fuori dal ciclo per una velocità estrema (livello C)
RE_TRUE_LABEL = re.compile(r'true_label\s+["\']?(\d+)["\']?')
RE_PRED_LABEL = re.compile(r'predicted_label\s+["\']?(\d+)["\']?')
RE_CONF_TRUE  = re.compile(r'confidence_true_class\s+["\']?([-\d\.e]+)["\']?')
RE_CONF_PRED  = re.compile(r'prediction_confidence\s+["\']?([-\d\.e]+)["\']?')
RE_CONF_INF   = re.compile(r'confidence_pred_class\s+["\']?([-\d\.e]+)["\']?')

# [^\]]* assicura di cercare 'importance' solo all'interno di quello specifico blocco nodo/arco
RE_NODE = re.compile(r'node\s*\[\s*id\s+(\d+)[^\]]*?importance\s+["\']?([-\d\.e]+)["\']?')
RE_EDGE = re.compile(r'edge\s*\[\s*source\s+(\d+)\s+target\s+(\d+)[^\]]*?importance\s+["\']?([-\d\.e]+)["\']?')
RE_FILENAME = re.compile(r'window_\d+_([A-Za-z0-9_]+)_pred_([A-Za-z0-9_]+)_grp_')

def parse_gml_files(file_list, confidence_cutoff=0.0):
    """
    Ultra-Fast GML parser using C-level regex engines.
    """
    grouped_nodes = {}
    grouped_edges = {}
    valid_samples = {}
    skipped_samples = 0

    print("\nAnalyzing graphs and applying Inference/Validation grouping logic...")

    for fpath in tqdm(file_list, desc="Parsing GMLs"):
        try:
            fname = os.path.basename(fpath)
            is_inference = '_unknown_pred_' in fname

            # Legge tutto il file in RAM in un solo colpo (rapidissimo per file da pochi MB)
            with open(fpath, 'r') as f:
                content = f.read()

            # Estrarre i nomi delle classi direttamente dal nome del file
            m_fname = RE_FILENAME.search(fname)
            true_cls_str = m_fname.group(1) if m_fname else None
            pred_cls_str = m_fname.group(2) if m_fname else None

            # --- 1. METADATA PARSING ---
            if not is_inference:
                m_true = RE_TRUE_LABEL.search(content)
                m_pred = RE_PRED_LABEL.search(content)

                if not m_true or not m_pred:
                    continue
                true_label, pred_label = int(m_true.group(1)), int(m_pred.group(1))

                if true_label != pred_label:
                    skipped_samples += 1
                    continue

                m_conf = RE_CONF_TRUE.search(content) or RE_CONF_PRED.search(content)
                conf = float(m_conf.group(1)) if m_conf else 0.0
                
                # Usa il nome della classe estratto dal file se disponibile
                group_key = f"Val_{true_cls_str}" if true_cls_str else f"Val_TrueClass_{true_label}"
            else:
                m_pred = RE_PRED_LABEL.search(content)
                if not m_pred:
                    continue
                pred_label = int(m_pred.group(1))

                m_conf = RE_CONF_INF.search(content)
                conf = float(m_conf.group(1)) if m_conf else 0.0
                
                # Usa il nome della predizione estratto dal file se disponibile
                group_key = f"Inf_{pred_cls_str}" if pred_cls_str else f"Inf_PredClass_{pred_label}"

            if conf < confidence_cutoff:
                skipped_samples += 1
                continue

            global_key = "Global"
            if group_key not in grouped_nodes:
                grouped_nodes[group_key] = {}
                grouped_edges[group_key] = {}
                valid_samples[group_key] = 0

            if global_key not in grouped_nodes:
                grouped_nodes[global_key] = {}
                grouped_edges[global_key] = {}
                valid_samples[global_key] = 0

            valid_samples[group_key] += 1
            valid_samples[global_key] += 1
            multiplier = 1.0

            # --- 2. NODES PARSING ---
            # Trova tutti i nodi nel file alla velocità del C
            for match in RE_NODE.finditer(content):
                idx = int(match.group(1))
                val = float(match.group(2)) * multiplier

                if abs(val) > 1e-6:
                    if idx not in grouped_nodes[group_key]:
                        grouped_nodes[group_key][idx] = []
                    grouped_nodes[group_key][idx].append(val)
                    
                    if idx not in grouped_nodes[global_key]:
                        grouped_nodes[global_key][idx] = []
                    grouped_nodes[global_key][idx].append(val)

            # --- 3. EDGES PARSING ---
            # Trova tutti gli archi nel file alla velocità del C
            for match in RE_EDGE.finditer(content):
                u = int(match.group(1))
                v = int(match.group(2))
                val = float(match.group(3)) * multiplier

                if abs(val) > 1e-6:
                    if u > v:
                        u, v = v, u

                    edge_tuple = (u, v)
                    if edge_tuple not in grouped_edges[group_key]:
                        grouped_edges[group_key][edge_tuple] = []
                    grouped_edges[group_key][edge_tuple].append(val)

                    if edge_tuple not in grouped_edges[global_key]:
                        grouped_edges[global_key][edge_tuple] = []
                    grouped_edges[global_key][edge_tuple].append(val)

        except Exception as e:
            continue

    print(f"Parsing completed: {sum(valid_samples.values())} valid graphs processed across {len(valid_samples)} groups, {skipped_samples} discarded/filtered.")
    return grouped_nodes, grouped_edges, valid_samples

# =============================================================================
# 3. STATISTICAL AGGREGATION
# =============================================================================

def aggregate_nodes(node_accumulator, valid_graphs, perc_val=95.0):
    stats_mean = {}
    stats_perc = {}

    all_indices = sorted(node_accumulator.keys())
    if not all_indices: return {}, {}, 1.0, 1.0

    max_idx = max(all_indices)
    all_mean_vals = []
    all_perc_vals = []

    # Fill gaps with zeros
    for i in range(max_idx + 1):
        vals = node_accumulator.get(i, [])

        if not vals:
            m, p = 0.0, 0.0
        else:
            # Assicuriamoci che la media sia spalmata uniformemente su tutti i grafi validi processati.
            # Se ci sono 'buchi' (nodi non processati per qualche motivo astruso), valgono 0.
            m = np.sum(vals) / valid_graphs if valid_graphs > 0 else 0.0
            
            # Il percentile può restare come np.percentile sui valori non-zero,
            # oppure potremmo fare un pad a zero, ma il percentile 'sui valori attivi' è di prassi per ora,
            # lo paddiamo comunque per coerenza se valid_graphs > len(vals)
            padded_vals = vals + [0.0] * (valid_graphs - len(vals)) if valid_graphs > len(vals) else vals
            p = np.percentile(padded_vals, perc_val)

        stats_mean[i] = m
        stats_perc[i] = p
        all_mean_vals.append(m)
        all_perc_vals.append(p)

    max_mean = np.max(all_mean_vals) if all_mean_vals else 1.0
    max_perc = np.max(all_perc_vals) if all_perc_vals else 1.0

    return stats_mean, stats_perc, max_mean, max_perc

def aggregate_edges(edge_accumulator, valid_graphs, perc_val=95.0):
    edges_mean = []
    edges_perc = []

    for (u, v), vals in edge_accumulator.items():
        if valid_graphs > 0:
            m = np.sum(vals) / valid_graphs
        else:
            m = 0.0

        padded_vals = vals + [0.0] * (valid_graphs - len(vals)) if valid_graphs > len(vals) else vals
        p = np.percentile(padded_vals, perc_val) if padded_vals else 0.0
        freq = len(vals)
        edges_mean.append({'u': u, 'v': v, 'importance': m, 'frequency': freq})
        edges_perc.append({'u': u, 'v': v, 'importance': p, 'frequency': freq})

    # Sort by magnitude (values are already absolute saliency)
    edges_mean.sort(key=lambda x: x['importance'], reverse=True)
    edges_perc.sort(key=lambda x: x['importance'], reverse=True)

    # Use absolute magnitude for ranking, but keep signs inside
    max_mean = max([abs(e['importance']) for e in edges_mean]) if edges_mean else 1.0
    max_perc = max([abs(e['importance']) for e in edges_perc]) if edges_perc else 1.0

    return edges_mean, edges_perc, max_mean, max_perc

# =============================================================================
# 4. OUTPUT GENERATION
# =============================================================================

def write_colored_pdb(pdb_template_path, out_path, importance_map, max_val, metric_name, valid_res_indices=None):
    if not os.path.exists(pdb_template_path): return

    scale_factor = 100.0 / max_val if max_val > 0 else 1.0
    reverse_map = {res_idx: gml_idx for gml_idx, res_idx in enumerate(valid_res_indices)} if valid_res_indices else None

    with open(pdb_template_path, 'r') as f_in, open(out_path, 'w') as f_out:
        node_idx = -1
        last_res_seq = None

        for line in f_in:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                res_seq = line[22:26].strip()
                if res_seq != last_res_seq:
                    node_idx += 1
                    last_res_seq = res_seq

                if reverse_map is not None:
                    raw_val = importance_map.get(reverse_map.get(node_idx, -1), 0.0)
                else:
                    raw_val = importance_map.get(node_idx, 0.0)

                scaled_val = raw_val * scale_factor
                scaled_val = max(min(scaled_val, 99.99), -99.99) # PDB B-factor format constraint

                new_line = line[:60] + f"{scaled_val:6.2f}" + line[66:]
                f_out.write(new_line)
            else:
                f_out.write(line)

    print(f"[{metric_name}] PDB exported to: {out_path}")

def generate_cgo_script(edge_list, pdb_path, out_script_path, max_val, valid_res_indices=None, edge_top_n=100, obj_name='xAI_Edges'):
    try:
        traj = md.load(pdb_path)
    except OSError: return

    coords = {}
    xyz = traj.xyz[0] * 10.0
    for i, res in enumerate(traj.topology.residues):
        atom_idx = next((a.index for a in res.atoms if a.name == 'CA'), res.atom(0).index)
        coords[i] = xyz[atom_idx]

    viz_edges = edge_list[:edge_top_n]

    with open(out_script_path, 'w') as f:
        f.write(f"from pymol.cgo import *\nfrom pymol import cmd\n\ndef load_edges_{edge_top_n}():\n    obj = []\n")

        for edge in viz_edges:
            u, v, imp = edge['u'], edge['v'], edge['importance']

            if valid_res_indices is not None:
                if u >= len(valid_res_indices) or v >= len(valid_res_indices): continue
                abs_u, abs_v = valid_res_indices[u], valid_res_indices[v]
            else:
                abs_u, abs_v = u, v

            if abs_u not in coords or abs_v not in coords: continue

            p1 = coords[abs_u]
            p2 = coords[abs_v]

            # Divergent mapping for Cylinders
            r, g, b = get_color_gradient(imp, max_val)

            # Thickness based on absolute strength
            abs_strength = abs(imp) / (max_val + 1e-9)
            radius = 0.1 + (abs_strength * 0.3)

            f.write(f"    obj.extend([CYLINDER, {p1[0]:.3f}, {p1[1]:.3f}, {p1[2]:.3f}, {p2[0]:.3f}, {p2[1]:.3f}, {p2[2]:.3f}, {radius:.3f}, {r:.2f}, {g:.2f}, {b:.2f}, {r:.2f}, {g:.2f}, {b:.2f}])\n")

        f.write(f"\n    cmd.load_cgo(obj, '{obj_name}')\n    cmd.set('cgo_transparency', 0.2, '{obj_name}')\n\nload_edges_{edge_top_n}()\n")

    print(f"CGO Cylinders script exported to: {out_script_path}")

def generate_master_pml(out_dir, pdb_mean, pdb_perc, cgo_scripts_info, perc_val, prefix):
    pml_path = os.path.join(out_dir, f"{prefix}view_xai.pml")

    # Adapt legend based on the used mode
    legend_text = "Ramp: Magenta (Negative) -> Grey60 (Neutral) -> Green (Positive/High Saliency)"

    with open(pml_path, 'w') as f:
        f.write("# PyMOL Visualization Script generated by HybridStSchnet xAI\n")
        f.write("reinitialize\n")
        f.write(f"cd {os.path.abspath(out_dir)}\n\n")

        f.write("# Registering Dynamic Color Scale Limits command\n")
        f.write("python\n")
        f.write("from pymol import cmd\n")
        f.write("def colorscale_limits(limit):\n")
        f.write("    limit = float(limit)\n")
        f.write("    # Apply spectrum to both potential structures using grey60 palette\n")
        f.write("    cmd.spectrum('b', 'magenta_grey60_green', 'Mean_Importance', minimum=-limit, maximum=limit)\n")
        f.write("    cmd.spectrum('b', 'magenta_grey60_green', 'Percentile_Structure', minimum=-limit, maximum=limit)\n")
        f.write("    # Recreate the color bar ramp\n")
        f.write("    cmd.delete('color_bar')\n")
        f.write("    cmd.ramp_new('color_bar', 'Mean_Importance', [-limit, 0, limit], ['magenta', 'grey60', 'green'])\n")
        f.write("cmd.extend('colorscale_limits', colorscale_limits)\n")
        f.write("python end\n\n")

        f.write("# 1. Load Mean Importance Structure\n")
        f.write(f"load {os.path.basename(pdb_mean)}, Mean_Importance\n")
        f.write("hide everything, Mean_Importance\n")
        f.write("show cartoon, Mean_Importance\n")
        f.write("spectrum b, magenta_grey60_green, Mean_Importance, minimum=-100, maximum=100\n")
        f.write("ramp_new color_bar, Mean_Importance, [-100, 0, 100], [magenta, grey60, green]\n")
        f.write("set cartoon_gap_cutoff, 0\n\n")

        f.write(f"# 2. Load {perc_val}th Percentile Structure (Disabled by default)\n")
        f.write(f"load {os.path.basename(pdb_perc)}, Percentile_Structure\n")
        f.write("hide everything, Percentile_Structure\n")
        f.write("show cartoon, Percentile_Structure\n")
        f.write("spectrum b, magenta_grey60_green, Percentile_Structure, minimum=-100, maximum=100\n")
        f.write("disable Percentile_Structure\n\n")

        f.write("# 3. Load Edges Scripts\n")
        f.write("show cgo\n")
        for i, (script_path, obj_name) in enumerate(cgo_scripts_info):
            f.write(f"run {os.path.basename(script_path)}\n")
            if i == 0:
                f.write(f"enable {obj_name}\n")
            else:
                f.write(f"disable {obj_name}\n")
        f.write("\n")

        f.write("# 4. View Settings\n")
        f.write("bg_color white\n")
        f.write("set ray_shadows, 0\n")
        f.write("set orthoscopic, 1\n")
        f.write("orient\n")
        f.write("print '-' * 50\n")
        f.write(f"print '{legend_text}'\n")
        f.write("print 'You can resize color limits by typing: colorscale_limits <value>'\n")
        f.write("print 'Example: colorscale_limits 50'\n")
        f.write("print '-' * 50\n")

    print(f"-> PyMOL Master Script ready. To visualize, run this command in PyMOL: @{pml_path}\n")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser, _, _ = get_parser()
    args = parse_with_config(parser)
    
    if not getattr(args, 'ig_results_dir', None) or not getattr(args, 'pdb_template', None):
        parser.error("The following arguments are required: --ig_results_dir, --pdb_template")

    if not os.path.exists(args.ig_results_dir):
        print(f"Error: IG results directory '{args.ig_results_dir}' does not exist.")
        sys.exit(1)

    print(f"\n{'='*70}\n Advanced PyMOL xAI Generator\n{'='*70}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    config_out_path = os.path.join(args.out_dir, "pymol_generator_config.txt")
    full_parser, _, _ = get_parser(show_debug=True, show_advanced=True)
    args = save_and_reload_config(args, config_out_path, parser=full_parser)

    mode = 'all'
    sel_str = None
    if getattr(args, 'resid', None):
        mode = 'resid'
        sel_str = args.resid
    elif getattr(args, 'resnum', None):
        mode = 'resnum'
        sel_str = args.resnum

    valid_res_indices = None
    if sel_str:
        try:
            pdb_top = md.load(args.pdb_template).topology
            valid_res_indices = get_selection_indices(mode, sel_str, pdb_top)
            print(f"Residue mask active: Found {len(valid_res_indices)} selected positions.")
        except Exception as e:
            print(f"Error reading topology for residue selection: {e}")
            sys.exit(1)

    gml_files = glob.glob(os.path.join(args.ig_results_dir, "**", "*.gml"), recursive=True)
    if not gml_files:
        print(f"No GML files found in {args.ig_results_dir}.")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Parsing with new logic
    grouped_nodes, grouped_edges, valid_samples = parse_gml_files(gml_files, args.confidence_cut)
    if not valid_samples:
        print("No valid graph passed the filters (confidence or correctness).")
        sys.exit(1)

    for group_key in valid_samples.keys():
        print(f"\nProcessing Group: {group_key} (Graphs: {valid_samples[group_key]})")
        node_acc = grouped_nodes[group_key]
        edge_acc = grouped_edges[group_key]
        n_valid = valid_samples[group_key]

        # 2. Statistical Aggregation
        nodes_mean, nodes_perc, max_node_mean, max_node_perc = aggregate_nodes(node_acc, n_valid, args.percentile)
        edges_mean, edges_perc, max_edge_mean, max_edge_perc = aggregate_edges(edge_acc, n_valid, args.percentile)

        # 3. Output Generation
        if group_key == "Global":
            current_out_dir = args.out_dir
            prefix = "Global_"
        else:
            current_out_dir = os.path.join(args.out_dir, group_key)
            os.makedirs(current_out_dir, exist_ok=True)
            prefix = f"{group_key}_"

        pdb_mean_path = os.path.join(current_out_dir, f"{prefix}nodes_mean.pdb")
        pdb_perc_path = os.path.join(current_out_dir, f"{prefix}nodes_p{int(args.percentile)}.pdb")
        write_colored_pdb(args.pdb_template, pdb_mean_path, nodes_mean, max_node_mean, f"Mean ({group_key})", valid_res_indices)
        write_colored_pdb(args.pdb_template, pdb_perc_path, nodes_perc, max_node_perc, f"Percentile_{args.percentile} ({group_key})", valid_res_indices)

        cgo_scripts_info = []
        top_edges = sorted(args.top_edges)
        
        for top_n in top_edges:
            cgo_path = os.path.join(current_out_dir, f"{prefix}draw_edges_top{top_n}.py")
            obj_name = f"Edges_Top_{top_n}_{group_key}"
            generate_cgo_script(edges_mean, args.pdb_template, cgo_path, max_edge_mean, valid_res_indices, top_n, obj_name)
            cgo_scripts_info.append((cgo_path, obj_name))

        generate_master_pml(current_out_dir, pdb_mean_path, pdb_perc_path, cgo_scripts_info, args.percentile, prefix)

        # Save the CSV
        csv_path = os.path.join(current_out_dir, f"{prefix}edges_stats_mean.csv")
        pd.DataFrame(edges_mean).to_csv(csv_path, index=False)
        print(f"Edge statistics exported to: {csv_path}")

if __name__ == "__main__":
    main()
