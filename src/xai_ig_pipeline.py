import sys
import os
import argparse

# =============================================================================
# CLI PARSER (Caricato per primo per garantire un help istantaneo)
# =============================================================================

def get_parser(show_debug=None, show_advanced=None):
    if show_debug is None:
        show_debug = '--debug-help' in sys.argv
    if show_advanced is None:
        show_advanced = '--advanced-help' in sys.argv or show_debug

    parser = argparse.ArgumentParser(
        description="Integrated Gradients (IG) and VMD Export Pipeline for Spatiotemporal GNNs.\n"
                    "Automatically loads architectural and dataset parameters from the config.json of each trained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False
    )

    help_group = parser.add_argument_group('Help')
    help_group.add_argument('-h', '--help', action='help', help="Show this help message and exit.")
    help_group.add_argument('--restart', action='store_true', help="Skip already existing .gml files.")

    io_group = parser.add_argument_group('Input and Output', 'Define model sources and output destinations.')
    io_group.add_argument("-c", "--config", type=str, help="Optional text file to load configuration from.")
    io_group.add_argument("--model_dirs", nargs='+', required=False,
                          help="List of training directories containing 'best_linear_model.pt' and 'config.txt' (e.g., results_hybrid/valrep_1).")
    io_group.add_argument("--inference_dirs", nargs='+', required=False,
                          help="Optional paths to new simulation directories for pure inference. If provided, the script ignores validation groups and runs prediction+XAI on these new data.")
    io_group.add_argument("--out_dir", type=str, default="./xai_ig_results",
                          help="Main output directory. Results will be automatically organized into valrep subfolders.")

    ig_group = parser.add_argument_group('Integrated Gradients Parameters', 'Configure the interpretability algorithm.')
    ig_group.add_argument("--ig_steps", type=int, default=10,
                          help="Number of interpolation steps for the baseline approximation (higher = more accurate but slower).")
    ig_group.add_argument("--N_baseline_medoids", type=int, default=5,
                          help="Number of background windows (Medoids) to extract per class for the expected_gradients baseline calculation.")
    adv_group = parser.add_argument_group('Advanced Options', 'Advanced configuration flags (use --advanced-help to view).')
    adv_group.add_argument("--vram_mode", type=str, choices=['standard', 'memory_saving'], default='standard', help="Optimization mode for VRAM usage." if show_advanced else argparse.SUPPRESS)
    adv_group.add_argument("--baseline", type=str, default="expected_gradients", choices=["thermodynamic_mean", "zero_edges", "expected_gradients"],
                          help="Metodo di baseline. 'thermodynamic_mean': stati medi di altre classi; 'zero_edges': RBF iniziali nulli; 'expected_gradients': Expected Gradients su Medoidi reali." if show_advanced else argparse.SUPPRESS)

    vmd_group = parser.add_argument_group('Dashboard Export Options', 'Settings for spatial and temporal data generation.')
    vmd_group.add_argument("--remove_gmls", action="store_true",
                          help="Remove the intermediate .gml graph files after exporting to VMD formats. By default, they are kept as they are required by the PyMOL generator.")

    return parser, show_debug, show_advanced

# --- HELP CHECK PRECOCE ---
if __name__ == "__main__":
    if any(arg in sys.argv for arg in ['-h', '--help']):
        parser, _, _ = get_parser()
        parser.print_help()
        sys.exit(0)

# =============================================================================
# IMPORT PESANTI
# =============================================================================

import json
import glob
import re
import math
import numpy as np
import networkx as nx
import csv
from tqdm import tqdm
import torch

from core.utils import setup_logger, seed_everything, parse_with_config, save_and_reload_config
from core.dataset import MDFlexibleWindowDataset, collate_windows
from core.architecture import HybridStSchnet
from core.baseline_utils import compute_global_baseline, inject_ghost_edges, select_medoid_baselines, inject_multi_baseline_ghost_edges
from core.ig_utils import (
    LinearProbeClassifier,
    IGInteractionWrapper,
    integrated_gradients_rbf_directional,
    save_window_as_gml_directional,
    extract_node_importance_vector
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# FUNZIONI PIPELINE (Generazione -> Esportazione)
# =============================================================================

def run_ig_generation(config, model_dir, out_gml_dir, logger, ig_steps, baseline_type, N_baseline_medoids, restart=False, args=None):
    logger.info("Initializing Model and Dataset for IG Graph Generation...")

    val_groups = config.get('val_groups', [])
    num_classes = len(config.get('data_class_dirs', config.get('data_dir', [])))
    class_labels = config.get('data_class_labels', [f"Class_{i}" for i in range(num_classes)])

    backbone = HybridStSchnet(
        hidden_dim=config.get('hidden_dim', 128), embedding_dim=config.get('embedding_dim', 64),
        window_size=config.get('window', 10), n_layers=config.get('n_layers', 3),
        cutoff=config.get('cutoff', 10.0), dihedral_dim=config.get('dihedral_dim', 0),
        temporal_setup=config.get('temporal_setup', 'cnn'), pooling_type=config.get('pooling_type', 'attention'),
        pooling_activation=config.get('pooling_activation', 'softmax'), num_classes=None
    )

    full_model = LinearProbeClassifier(backbone, num_classes, config.get('embedding_dim', 64))
    linear_path = os.path.join(model_dir, "best_linear_model.pt")
    if not os.path.exists(linear_path): return False

    state_dict = torch.load(linear_path, map_location=DEVICE, weights_only=True)
    full_model.load_state_dict(state_dict)
    full_model.to(DEVICE).eval()

    ig_wrapper = IGInteractionWrapper(full_model).to(DEVICE)

    baselines_per_class = None
    medoids_per_class = None

    if baseline_type == "thermodynamic_mean":
        logger.info("Computing thermodynamic_mean baseline...")
        baselines_per_class = compute_global_baseline(config, ig_wrapper, DEVICE)
        if not baselines_per_class:
            logger.error("Failed to compute class-specific baselines from training set.")
            return False
    elif baseline_type == "expected_gradients":
        logger.info(f"Extracting {N_baseline_medoids} Medoid baselines per class from training set...")
        medoids_per_class = select_medoid_baselines(config, full_model, DEVICE, num_baselines=N_baseline_medoids)
        if not medoids_per_class:
            logger.error("Failed to extract Medoid baselines.")
            return False
    else:
        logger.info("Using zero_edges baseline. Skipping global distribution extraction.")

    if getattr(args, 'inference_dirs', None):
        logger.info(f"Inference Mode active. Explaining new data from: {args.inference_dirs}")
        ds = MDFlexibleWindowDataset(
            args.inference_dirs, split='all', val_groups=None,
            window_size=config.get('window', 10), window_offset=config.get('window_offset', None),
            stride=1, logger=None, skip=0,
            preload_ram=config.get('preload_ram', False), global_shuffle=False, window_shuffle=False,
            ignore_validation_logic=True
        )
    else:
        ds = MDFlexibleWindowDataset(
            config.get('data_class_dirs', config.get('data_dir')), split='val', val_groups=val_groups,
            window_size=config.get('window', 10), window_offset=config.get('window_offset', None),
            stride=1, logger=None, skip=0,
            preload_ram=config.get('preload_ram', False), global_shuffle=False, window_shuffle=False
        )

    if len(ds) == 0: return False

    # DataLoader sequenziale
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_windows)

    count = 0
    for i, (batched_data, labels, groups, _, batch_paths) in enumerate(tqdm(loader, desc="Generating GMLs")):
        if restart:
            existing_files = glob.glob(os.path.join(out_gml_dir, f"window_{i:06d}_*.gml"))
            if existing_files:
                count += 1
                continue

        batched_data = batched_data.to(DEVICE)
        true_label = labels[0].item()

        # Inietto l'autocast qui dentro per tutto il loop se richiesto
        autocast_dtype = torch.bfloat16 if getattr(args, 'vram_mode', 'standard') == 'memory_saving' else (torch.float16 if torch.cuda.is_available() else torch.float32)
        # Fix per fallback sicuro a float32 se non si usa memory_saving e si è in dubbio. Siccome il resto del codice andava in float32 nativo senza autocast:
        # Se memory_saving -> bfloat16, altrimenti -> si skippa autocast o si setta float32 (che fa nulla in autocast).
        
        ctx = torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16) if getattr(args, 'vram_mode', 'standard') == 'memory_saving' else torch.autocast(device_type=DEVICE.type, enabled=False)

        with ctx:
            with torch.no_grad():
                logits = full_model(batched_data)
                probs = torch.softmax(logits, dim=1)

            energy = -torch.logsumexp(logits, dim=1)
            energy_score = energy[0].item()

            conf_true = probs[0, true_label].item() if true_label < probs.shape[1] else 0.0
            pred_label = logits.argmax(dim=1).item()
            conf_pred = probs[0, pred_label].item()

        is_inference = getattr(args, 'inference_dirs', None) is not None
        target = pred_label if is_inference else true_label

        real_dist = batched_data.edge_attr.clone().detach()
        if real_dist.dim() > 1: real_dist = real_dist.squeeze()

        with torch.no_grad():
            rbf_val = ig_wrapper.spatial.rbf(real_dist)
            env_val = ig_wrapper.spatial.envelope(real_dist, ig_wrapper.spatial.cutoff).unsqueeze(-1)
            target_rbf_feat = (rbf_val * env_val).detach()

        # Selezione e Applicazione metodo baseline
        if baseline_type == "thermodynamic_mean":
            fallback_k = list(baselines_per_class.keys())[0]
            mean_baseline_edge_index, mean_baseline_rbf = baselines_per_class.get(target, baselines_per_class[fallback_k])

            batched_data, aligned_target_rbf = inject_ghost_edges(
                batched_data, target_rbf_feat, mean_baseline_edge_index, DEVICE
            )

            node_imp, edge_imp = integrated_gradients_rbf_directional(
                ig_wrapper, batched_data, target,
                target_rbf=aligned_target_rbf, baseline_rbf=mean_baseline_rbf,
                baseline_type=baseline_type, steps=ig_steps, device=DEVICE
            )

        elif baseline_type == "expected_gradients":
            # Contrasto contro i Medoidi di tutte le altre classi
            other_classes = [c for c in medoids_per_class.keys() if c != target]
            if not other_classes:
                other_classes = [target] # Fallback seesiste 1 sola classe

            baselines_to_use = []
            weights_to_use = []
            for c in other_classes:
                medoids, weights = medoids_per_class[c]
                baselines_to_use.extend(medoids)
                weights_to_use.extend(weights)

            # Normalizza i pesi a 1.0 per la media pesata
            total_weight = sum(weights_to_use)
            if total_weight > 0:
                weights_to_use = [w / total_weight for w in weights_to_use]
            else:
                weights_to_use = [1.0 / len(weights_to_use) for _ in weights_to_use]

            batched_data, aligned_target_rbf, aligned_b_rbfs = inject_multi_baseline_ghost_edges(
                batched_data, target_rbf_feat, baselines_to_use, ig_wrapper, DEVICE
            )

            n_imp_acc, e_imp_acc = 0, 0
            for b_rbf, w_b in zip(aligned_b_rbfs, weights_to_use):
                n_imp, e_imp = integrated_gradients_rbf_directional(
                    ig_wrapper, batched_data, target,
                    target_rbf=aligned_target_rbf, baseline_rbf=b_rbf,
                    baseline_type=baseline_type, steps=ig_steps, device=DEVICE
                )
                n_imp_acc += n_imp * w_b
                e_imp_acc += e_imp * w_b

            # Expected Gradients: Media pesata
            node_imp = n_imp_acc
            edge_imp = e_imp_acc

        else: # zero_edges
            aligned_target_rbf = target_rbf_feat
            node_imp, edge_imp = integrated_gradients_rbf_directional(
                ig_wrapper, batched_data, target,
                target_rbf=aligned_target_rbf, baseline_rbf=None,
                baseline_type=baseline_type, steps=ig_steps, device=DEVICE
            )

        info = {
            'sample_idx': i, 'true_label': true_label, 'predicted_label': pred_label,
            'confidence_true_class': conf_true, 'confidence_pred_class': conf_pred,
            'energy_score': energy_score,
            'target_class_analyzed': target, 'ig_method': f'rbf_directional_{baseline_type}'
        }

        if is_inference:
            class_name_true = "unknown"
        else:
            class_name_true = class_labels[true_label] if true_label < len(class_labels) else f"class{true_label}"
            
        class_name_pred = class_labels[pred_label] if pred_label < len(class_labels) else f"class{pred_label}"
        group_name = os.path.basename(os.path.dirname(batch_paths[0][0]))

        fname = f"window_{i:06d}_{class_name_true}_pred_{class_name_pred}_grp_{group_name}.gml"
        save_window_as_gml_directional(
            sample_idx=i, num_nodes_per_frame=batched_data.num_nodes // config.get('window', 10),
            window_size=config.get('window', 10), edge_index_full=batched_data.edge_index,
            edge_imp_full=edge_imp, node_imp_full=node_imp, labels_full=batched_data.x,
            out_path=os.path.join(out_gml_dir, fname), prediction_info=info
        )
        count += 1
    return ds

def run_vmd_export(config, ds, gml_dir, out_vmd_dir, logger):
    logger.info("Aggregating GMLs into VMD Formats (Spatial DAT + Temporal CSV)...")

    gml_files = glob.glob(os.path.join(gml_dir, "**/*.gml"), recursive=True)
    if not gml_files: return

    class_labels = config.get('data_class_labels', [])
    if not class_labels:
        class_labels = [f"class{i}" for i in range(10)]

    pattern = re.compile(r"window_(\d+)_([A-Za-z0-9_]+)_pred_([A-Za-z0-9_]+)_grp_([A-Za-z0-9_\-]+)")
    valid_files = []
    for fpath in gml_files:
        match = pattern.search(os.path.basename(fpath))
        if match:
            w_idx = int(match.group(1))
            true_cls_str = match.group(2)
            group_str = match.group(4)

            true_cls_idx = -1
            if true_cls_str in class_labels:
                true_cls_idx = class_labels.index(true_cls_str)
            elif true_cls_str.startswith("class"):
                true_cls_idx = int(true_cls_str.replace("class", ""))
            else:
                true_cls_idx = sum(ord(c) for c in true_cls_str)

            if w_idx < len(ds): valid_files.append({
                'w_idx': w_idx, 'path': fpath,
                'true_cls': true_cls_idx, 'true_cls_str': true_cls_str,
                'group_str': group_str
            })

    valid_files.sort(key=lambda x: x['w_idx'])

    window_size = config.get('window', 10)
    window_offset = config.get('window_offset')
    if window_offset is None: window_offset = window_size

    overlap_factor = window_size / float(window_offset)
    MAX_TRACKS = 8

    track_step = max(1, math.ceil(overlap_factor / MAX_TRACKS))
    logger.info(f"Overlap Factor is {overlap_factor:.1f}. Assigning 1 window every {track_step} to the {MAX_TRACKS} tracks.")

    track_allocator = {}
    window_track_map = {}
    window_counter = {}
    group_max_frames = {}
    prefix_map = {}

    for item in valid_files:
        w_idx, true_cls = item['w_idx'], item['true_cls']
        paths = ds.samples[w_idx][0]

        f_match = ds.filename_re.search(os.path.basename(paths[0]))
        l_match = ds.filename_re.search(os.path.basename(paths[-1]))
        if not f_match or not l_match: continue

        prefix, start_fid = f_match.group(1), int(f_match.group(2))
        end_fid = int(l_match.group(2))
        true_cls_str = item.get('true_cls_str', f"class_{true_cls}")
        group_str = item.get('group_str', os.path.basename(os.path.dirname(paths[0])))
        dict_key = f"{prefix}_{true_cls_str}_{group_str}"
        prefix_map[dict_key] = prefix

        if dict_key not in group_max_frames: group_max_frames[dict_key] = 0
        if end_fid > group_max_frames[dict_key]: group_max_frames[dict_key] = end_fid

        if dict_key not in window_counter: window_counter[dict_key] = 0
        current_w_idx = window_counter[dict_key]
        window_counter[dict_key] += 1

        if dict_key not in track_allocator: track_allocator[dict_key] = [-1] * MAX_TRACKS

        assigned_track = -1
        if current_w_idx % track_step == 0:
            for t in range(MAX_TRACKS):
                if track_allocator[dict_key][t] < start_fid:
                    assigned_track = t
                    track_allocator[dict_key][t] = end_fid
                    break

        window_track_map[w_idx] = assigned_track

    G_test = nx.read_gml(valid_files[0]['path'])
    num_residues = len(extract_node_importance_vector(G_test))

    logger.info("Converting ig structures to pure CSV formats for VMD Visualization...")
    spatial_sum, temporal_other, frame_count, temporal_sals = {}, {}, {}, {}
    for dict_key, max_fid in group_max_frames.items():
        total_frames = max_fid + 1
        spatial_sum[dict_key] = np.zeros((total_frames, num_residues), dtype=np.float32)
        temporal_other[dict_key] = np.zeros((total_frames, 2), dtype=np.float32)
        frame_count[dict_key] = np.zeros((total_frames, 1), dtype=np.float32)
        temporal_sals[dict_key] = np.full((total_frames, MAX_TRACKS), np.nan, dtype=np.float32)

    for item in tqdm(valid_files, desc="Exporting to VMD"):
        w_idx, true_cls, fpath = item['w_idx'], item['true_cls'], item['path']
        try:
            G = nx.read_gml(fpath)
            vec = extract_node_importance_vector(G, num_residues)
            conf = float(G.graph.get('confidence_true_class', 0.0))
            energy = float(G.graph.get('energy_score', 0.0))
        except Exception: continue

        paths = ds.samples[w_idx][0]
        track_id = window_track_map.get(w_idx, -1)

        for f_idx, p in enumerate(paths):
            match = ds.filename_re.search(os.path.basename(p))
            if match:
                prefix, fid = match.group(1), int(match.group(2))
                true_cls_str = item.get('true_cls_str', f"class_{true_cls}")
                group_str = item.get('group_str', os.path.basename(os.path.dirname(paths[0])))
                dict_key = f"{prefix}_{true_cls_str}_{group_str}"
                if fid >= len(spatial_sum[dict_key]): continue

                spatial_sum[dict_key][fid] += vec
                temporal_other[dict_key][fid, 0] += conf
                temporal_other[dict_key][fid, 1] += energy
                frame_count[dict_key][fid] += 1.0

                if track_id != -1:
                    temporal_sals[dict_key][fid, track_id] = float(G.graph.get(f'frame_{f_idx}_saliency', 0.0))

    for dict_key in spatial_sum.keys():
        s_sum, t_other, count = spatial_sum[dict_key], temporal_other[dict_key], frame_count[dict_key]
        mask_nz = count[:, 0] > 0

        s_avg = np.zeros_like(s_sum)
        s_avg[mask_nz, :] = s_sum[mask_nz, :] / count[mask_nz, :]
        t_avg_other = np.zeros_like(t_other)
        t_avg_other[mask_nz, :] = t_other[mask_nz, :] / count[mask_nz, :]

        global_max = np.max(np.abs(s_avg))
        total_frames = s_avg.shape[0]

        prefix = prefix_map[dict_key]
        sim_dir = os.path.join(out_vmd_dir, prefix)
        os.makedirs(sim_dir, exist_ok=True)

        dat_file = os.path.join(sim_dir, f"{dict_key}_xai_spatial.dat")
        with open(dat_file, 'w') as f:
            f.write(f"# META {total_frames} {num_residues} {global_max} 0\n")
            for i in range(total_frames):
                f.write(" ".join([f"{x:.4f}" for x in s_avg[i]]) + "\n")

        csv_file = os.path.join(sim_dir, f"{dict_key}_xai_temporal.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = ["Frame"] + [f"Window_Track_{b}" for b in range(MAX_TRACKS)] + ["Confidence", "Anomaly_Score"]
            writer.writerow(headers)

            for i in range(total_frames):
                row_data = [i]
                for b in range(MAX_TRACKS):
                    val = temporal_sals[dict_key][i, b]
                    row_data.append("NaN" if np.isnan(val) else f"{val:.4f}")

                if mask_nz[i]:
                    row_data.extend([f"{t_avg_other[i, 0]:.4f}", f"{t_avg_other[i, 1]:.4f}"])
                else:
                    row_data.extend(["NaN", "NaN"])
                writer.writerow(row_data)

# =============================================================================
# ESECUZIONE PRINCIPALE
# =============================================================================

def main():
    parser, _, _ = get_parser()
    args = parse_with_config(parser)

    if getattr(args, 'model_dirs', None) is None:
        parser.error("The following arguments are required: --model_dirs")

    seed_everything(42)

    os.makedirs(args.out_dir, exist_ok=True)

    config_out_path = os.path.join(args.out_dir, "ig_pipeline_config.txt")
    full_parser, _, _ = get_parser(show_debug=True, show_advanced=True)
    args = save_and_reload_config(args, config_out_path, parser=full_parser)

    logger = setup_logger(args.out_dir)
    logger.info(f"--- Starting xAI Integrated Gradients & VMD Pipeline on {DEVICE} ---")
    logger.info(f"Using baseline method: {args.baseline}")

    for model_dir in args.model_dirs:
        model_name = os.path.basename(os.path.normpath(model_dir))
        logger.info(f"\n>> Processing Model Directory: {model_name}")

        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            logger.error(f"Cannot find config.json in {model_dir}. Skipping.")
            continue

        with open(config_path, 'r') as f:
            config = json.load(f)

        valrep_name = f"valrep_{config.get('val_groups', [0])[0]}"
        fold_out_dir = os.path.join(args.out_dir, valrep_name)
        gml_dir = os.path.join(fold_out_dir, "gmls")
        vmd_dir = os.path.join(fold_out_dir, "vmd_data")

        os.makedirs(gml_dir, exist_ok=True)
        os.makedirs(vmd_dir, exist_ok=True)

        # 1. GENERATE GMLS
        ds = run_ig_generation(config, model_dir, gml_dir, logger, args.ig_steps, args.baseline, args.N_baseline_medoids, getattr(args, 'restart', False), args)

        # 2. VMD EXPORT
        if ds:
            run_vmd_export(config, ds, gml_dir, vmd_dir, logger)

        # 3. CLEANUP
        if args.remove_gmls:
            logger.info("Cleaning up heavy GML files...")
            for f in glob.glob(os.path.join(gml_dir, "*.gml")):
                os.remove(f)
            os.rmdir(gml_dir)
            logger.info("Cleanup complete.")
        else:
            logger.info(f"Intermediate GML files preserved in {gml_dir} for PyMOL generator.")

if __name__ == "__main__":
    main()
