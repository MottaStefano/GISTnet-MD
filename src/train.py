import os
import sys
import argparse
import copy
import datetime
import json


def get_parser(show_debug=None, show_advanced=None):
    if show_debug is None:
        show_debug = '--debug-help' in sys.argv
    if show_advanced is None:
        show_advanced = '--advanced-help' in sys.argv or show_debug

    parser = argparse.ArgumentParser(
        description="Training Spatiotemporal GNN (Contrastive + Linear Probe) and xAI Post-Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    
    help_group = parser.add_argument_group('Help')
    help_group.add_argument('-h', '--help', action='help', help="Show basic options and exit.")
    help_group.add_argument('--advanced-help', action='store_true', help="Show advanced optimization options.")
    help_group.add_argument('--debug-help', action='store_true', help=argparse.SUPPRESS)

    io_group = parser.add_argument_group('Files and Folders', 'Input/output configuration.')
    io_group.add_argument("-c", "--config", type=str, help="Optional text file to load configuration from.")
    io_group.add_argument("--data_class_dirs", nargs='+', required=False, help="Folder(s) containing processed .pt files. One folder per class.")
    io_group.add_argument("--data_class_labels", nargs='+', required=False, help="Names of the classes corresponding to data_class_dirs.")
    io_group.add_argument("--out_dir", type=str, default="./results_hybrid", help="Folder to save weights and logs.")
    
    data_group = parser.add_argument_group('Data Configuration', 'Options on how frames are organized.')
    data_group.add_argument("--window", type=int, default=10, help="How many frames make up a sequence (time window).")
    data_group.add_argument("--window_offset", type=int, default=None, help="Spacing between two windows.")
    data_group.add_argument("--stride", type=int, default=1, help="Frame subsampling.")
    data_group.add_argument("--skip", type=int, default=0, help="How many initial frames to skip.")
    data_group.add_argument("--val_groups", type=int, nargs='+', default=[], help="Indices of groups to use as Validation.")

    arch_group = parser.add_argument_group('Architecture and Basic Training', 'Fundamental parameters of the GNN model.')
    arch_group.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer dimension in message passing.")
    arch_group.add_argument("--embedding_dim", type=int, default=64, help="Final embedding dimension produced by the model.")
    arch_group.add_argument("--seed", type=int, default=42, help="Seed for total reproducibility.")
    arch_group.add_argument("--epochs", type=int, default=50, help="Maximum training epochs for the Contrastive phase.")
    arch_group.add_argument("--batch_size", type=int, default=256, help="Logical global batch size.")
    arch_group.add_argument("--micro_batch_size", type=int, default=32, help="Effective batch size for gradient accumulation.")
    arch_group.add_argument("--contrastive_learning_rate", type=float, default=1e-4, help="Learning Rate for contrastive phase.")
    arch_group.add_argument("--linear_learning_rate", type=float, default=1e-3, help="Learning Rate for Linear Probe.")
    arch_group.add_argument("--linear_epochs", type=int, default=50, help="Training epochs for the Linear Probe.")
    arch_group.add_argument("--patience", type=int, default=10, help="Tolerance epochs before Early Stopping.")

    adv_group = parser.add_argument_group('Advanced Optimization and Regularization', 'Modify only if necessary.') if show_advanced else parser
    adv_group.add_argument("--map_negative_name_to_exclude", action='store_true', help="Exclude same-group negative examples in contrastive learning." if show_advanced else argparse.SUPPRESS)
    adv_group.add_argument("--disable_preload_ram", action='store_true', help="Disable loading files into RAM." if show_advanced else argparse.SUPPRESS)
    adv_group.add_argument("--n_layers", type=int, default=3, help="Number of GNN layers." if show_advanced else argparse.SUPPRESS)
    adv_group.add_argument("--dropout", type=float, default=0.0, help="Dropout rate in GNN." if show_advanced else argparse.SUPPRESS)
    adv_group.add_argument("--attn_temperature", type=float, default=1.0, help="Attention temperature for temporal pooling." if show_advanced else argparse.SUPPRESS)
    adv_group.add_argument("--margin", type=float, default=1.0, help="Margin for contrastive loss." if show_advanced else argparse.SUPPRESS)
    adv_group.add_argument("--use_scheduler", action='store_true', help="Use Cosine Annealing learning rate scheduler." if show_advanced else argparse.SUPPRESS)
    adv_group.add_argument("--scheduler_min_lr", type=float, default=1e-6, help="Minimum LR for scheduler." if show_advanced else argparse.SUPPRESS)
    adv_group.add_argument("--balance_classes", action='store_true', help="Use WeightedRandomSampler to balance classes." if show_advanced else argparse.SUPPRESS)
    adv_group.add_argument("--use_checkpointing", action='store_true', help="Enable gradient checkpointing to save memory." if show_advanced else argparse.SUPPRESS)
    
    adv_group.add_argument("--save_embeddings", action='store_true', help="Save numpy embeddings of every frame in train and val sets.")

    dbg_group = parser.add_argument_group('Debug Settings', 'Exclusive for debugging.') if show_debug else parser
    dbg_group.add_argument("--shuffling", type=str, choices=['none', 'windows', 'global'], default='none', help="Level of shuffling applied to the dataset." if show_debug else argparse.SUPPRESS)
    dbg_group.add_argument("--shuffling_set", type=str, choices=['all', 'training', 'validation'], default='all', help="Which dataset split to shuffle." if show_debug else argparse.SUPPRESS)
    dbg_group.add_argument("--noLORO", action='store_true', help="Disable Leave-One-Group-Out cross-validation and use random split." if show_debug else argparse.SUPPRESS)
    dbg_group.add_argument("--pooling_type", type=str, choices=['attention', 'mean'], default='attention', help="Temporal pooling strategy." if show_debug else argparse.SUPPRESS)
    dbg_group.add_argument("--temporal_setup", type=str, choices=['cnn', 'attention'], default='cnn', help="Module used to process the temporal sequence." if show_debug else argparse.SUPPRESS)
    dbg_group.add_argument("--pooling_activation", type=str, choices=['softmax', 'sigmoid'], default='softmax', help="Activation function for temporal attention." if show_debug else argparse.SUPPRESS)
    dbg_group.add_argument("--contrastive", type=str, choices=['group_aware', 'base', 'hard_negative', 'none'], default='group_aware', help="Type of contrastive learning loss." if show_debug else argparse.SUPPRESS)

    return parser, show_debug, show_advanced

if __name__ == "__main__":
    if any(arg in sys.argv for arg in ['-h', '--help', '--advanced-help', '--debug-help']):
        parser, show_debug, show_advanced = get_parser()
        # Stampa l'help direttamente a prescindere dal tipo di richiesta
        parser.print_help()
        sys.exit(0)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import relativi
from core.utils import setup_logger, seed_everything, parse_with_config, save_and_reload_config
from core.dataset import MDFlexibleWindowDataset, collate_windows
from core.architecture import HybridStSchnet
from core.loss import GroupContrastiveLoss
from core.train_linear import run_linear_probe, LinearProbeClassifier
from core.train_contrastive import HybridTrainer
from core.xAI_analysis import run_post_training_analysis

def resolve_shuffling_params(args, current_split_role):
    if args.shuffling == 'none':
        return False, False
        
    target_set = args.shuffling_set
    should_shuffle = False
    if target_set == 'all': should_shuffle = True
    elif target_set == 'training' and current_split_role == 'train': should_shuffle = True
    elif target_set == 'validation' and current_split_role == 'val': should_shuffle = True
        
    if not should_shuffle: return False, False
        
    return (args.shuffling == 'global'), (args.shuffling == 'windows')

def verify_group_counts(data_class_dirs, logger):
    group_counts = []
    for d in data_class_dirs:
        try:
            subdirs = [f.path for f in os.scandir(d) if f.is_dir()]
            group_counts.append(len(subdirs))
        except Exception as e:
            logger.error(f"Error reading directory {d}: {e}")
            sys.exit(1)
            
    if len(set(group_counts)) > 1:
        logger.error(f"CRITICAL ERROR: The class directories must have the SAME number of subfolders (groups). Found counts: {group_counts} for dirs {data_class_dirs}.")
        sys.exit(1)
    
    if len(group_counts) > 0 and group_counts[0] < 2:
        logger.error(f"CRITICAL ERROR: Each class must have AT LEAST 2 groups (subfolders) to allow for proper separation between training and validation. Found: {group_counts[0]} groups.")
        sys.exit(1)

def check_node_alignment(data_class_dirs, data_class_labels, logger):
    STANDARD_RESIDUES = [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
    ]
    
    reference_nodes = None
    ref_class_name = None
    ref_group_name = None
    
    mismatches_found = 0
    max_warnings = 50
    
    for class_idx, class_dir in enumerate(data_class_dirs):
        class_name = data_class_labels[class_idx]
        if not os.path.isdir(class_dir):
            continue
            
        subdirs = sorted([f.path for f in os.scandir(class_dir) if f.is_dir()])
        for group_path in subdirs:
            group_name = os.path.basename(group_path)
            
            # find first .pt file
            first_pt = None
            try:
                for p in os.listdir(group_path):
                    if p.endswith('.pt'):
                        first_pt = os.path.join(group_path, p)
                        break
            except Exception:
                continue
                
            if first_pt:
                data = torch.load(first_pt, weights_only=False)
                if getattr(data, 'x', None) is None:
                    continue
                node_ids = data.x.cpu().numpy()
                
                if reference_nodes is None:
                    reference_nodes = node_ids
                    ref_class_name = class_name
                    ref_group_name = group_name
                else:
                    if len(node_ids) != len(reference_nodes):
                        logger.warning(f"WARNING: Node length mismatch! Class '{ref_class_name}' group '{ref_group_name}' has {len(reference_nodes)} nodes, but Class '{class_name}' group '{group_name}' has {len(node_ids)} nodes.")
                        mismatches_found += 1
                        if mismatches_found >= max_warnings:
                            return
                    else:
                        for i, (n1, n2) in enumerate(zip(reference_nodes, node_ids)):
                            if n1 != n2:
                                res1 = STANDARD_RESIDUES[n1] if n1 < len(STANDARD_RESIDUES) else "UNK"
                                res2 = STANDARD_RESIDUES[n2] if n2 < len(STANDARD_RESIDUES) else "UNK"
                                logger.warning(f"WARNING: Node alignment mismatch at index {i}! Node {i} of class '{ref_class_name}' group '{ref_group_name}' is '{res1}', while for class '{class_name}' group '{group_name}' it is '{res2}'.")
                                mismatches_found += 1
                                if mismatches_found >= max_warnings:
                                    logger.warning("WARNING: Node alignment mismatch limit reached (50 warnings). Stopping alignment check.")
                                    return

def auto_detect_graph_params(data_dirs, logger):
    sample_pt = None
    for d in data_dirs:
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith('.pt'):
                    sample_pt = os.path.join(root, f)
                    break
            if sample_pt: break
        if sample_pt: break
    
    if not sample_pt:
        logger.error("ERROR: No .pt files found in the specified folders!")
        sys.exit(1)
        
    data_sample = torch.load(sample_pt, weights_only=False)
    cutoff = getattr(data_sample, 'cutoff', 10.0)
    node_dihe = getattr(data_sample, 'node_dihe', 'none')
    
    if node_dihe == 'none': dihedral_dim = 0
    elif node_dihe == 'backbone': dihedral_dim = 4
    elif node_dihe == 'full': dihedral_dim = 8
    else: dihedral_dim = 0
    
    logger.info(f"Auto-Detected Metadata from Graph: Cutoff = {cutoff}A | Node_Dihe = '{node_dihe}' (Dim: {dihedral_dim})")
    return cutoff, dihedral_dim

def create_log_table(args, out_dir, parser=None):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. SALVA IL JSON CON I PARAMETRI
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    # 1b. SALVA IL FILE .IN DI INPUT COME FATTO IN XAI
    config_in_path = os.path.join(out_dir, "train.in")
    full_parser, _, _ = get_parser(show_debug=True, show_advanced=True)
    args = save_and_reload_config(args, config_in_path, parser=full_parser)

    # 2. SALVA IL LOG LEGGIBILE
    log_path = os.path.join(out_dir, "train_run.log")
    with open(log_path, 'w') as f:
        f.write("="*70 + "\n MD-GNN TRAINING LOG\n" + "="*70 + "\n")
        f.write(f"Execution Date  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 70 + "\n")
        for arg, value in vars(args).items():
            if arg not in ['debug_help', 'advanced_help']:
                f.write(f"--{arg:<28} | {str(value):<40}\n")
        f.write("="*70 + "\n")
    return log_path

def plot_training_convergence(c_history, l_history, out_dir):
    """Genera e salva un plot dei valori storici del training in un unico PNG e CSV."""
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs_c = range(1, len(c_history['train_loss']) + 1)
    axes[0].plot(epochs_c, c_history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(epochs_c, c_history['val_loss'], label='Val Loss', color='orange')
    axes[0].set_title('Contrastive Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].legend()

    if 'val_acc' in c_history and any(not np.isnan(x) for x in c_history['val_acc']):
        axes[1].plot(epochs_c, c_history['val_acc'], label='KNN Val Acc', color='green')
        axes[1].set_title('Contrastive Validation Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].legend()

    if l_history:
        epochs_l = range(1, len(l_history['train_loss']) + 1)
        axes[2].plot(epochs_l, l_history['train_loss'], label='Linear Train Loss', color='purple', linestyle='--')
        ax2 = axes[2].twinx()
        ax2.plot(epochs_l, l_history['val_acc'], label='Linear Val Acc', color='red')
        axes[2].set_title('Linear Probe Training')
        axes[2].set_xlabel('Epochs')
        axes[2].legend(loc='upper left')
        ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_convergence.png"))
    plt.close()

    # CSV
    df_c = pd.DataFrame(c_history)
    df_c.index.name = 'Contrastive_Epoch'
    df_c.to_csv(os.path.join(out_dir, "training_history_contrastive.csv"))
    
    if l_history:
        df_l = pd.DataFrame(l_history)
        df_l.index.name = 'Linear_Epoch'
        df_l.to_csv(os.path.join(out_dir, "training_history_linear.csv"))

def main():
    parser, _, _ = get_parser()
    args = parse_with_config(parser)
    
    # Resolution of negated flags
    args.preload_ram = not args.disable_preload_ram

    # FIX: Iniezione variabili mancanti per compatibilità con il modulo xAI_analysis
    args.global_shuffle = getattr(args, 'shuffling', 'none') == 'global'
    args.window_shuffle = getattr(args, 'shuffling', 'none') == 'windows'

    seed_everything(args.seed)
    
    # Initialize logger, device, and auto-detect parameters
    logger = setup_logger(args.out_dir)
    log_file_path = create_log_table(args, args.out_dir, parser=parser)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"--- Pipeline Started on {device} ---")
    
    if len(args.data_class_dirs) != len(args.data_class_labels):
        logger.error(f"Number of data directories ({len(args.data_class_dirs)}) does not match number of labels ({len(args.data_class_labels)})!")
        sys.exit(1)
    verify_group_counts(args.data_class_dirs, logger)

    cutoff_val, dihe_dim_val = auto_detect_graph_params(args.data_class_dirs, logger)
    args.cutoff = cutoff_val
    args.dihedral_dim = dihe_dim_val
    
    if args.micro_batch_size > args.batch_size: args.micro_batch_size = args.batch_size

    # --- 1. Dataset Initialization ---
    test_loader = None
    if args.noLORO:
        if len(args.val_groups) > 0:
            logger.info(f"[Hybrid Mode] Hold-out Groups {args.val_groups} reserved for TEST.")
            g_s_train, w_s_train = resolve_shuffling_params(args, 'train')
            pool_ds = MDFlexibleWindowDataset(args.data_class_dirs, split='train', val_groups=args.val_groups, 
                                              window_size=args.window, window_offset=args.window_offset,
                                              stride=args.stride, skip=args.skip, 
                                              preload_ram=args.preload_ram, 
                                              global_shuffle=g_s_train, window_shuffle=w_s_train,
                                              logger=logger, ignore_validation_logic=False)
            g_s_val, w_s_val = resolve_shuffling_params(args, 'val')
            test_ds = MDFlexibleWindowDataset(args.data_class_dirs, split='val', val_groups=args.val_groups, 
                                              window_size=args.window, window_offset=args.window_offset,
                                              stride=args.stride, skip=args.skip, preload_ram=args.preload_ram, 
                                              global_shuffle=g_s_val, window_shuffle=w_s_val,
                                              logger=logger, ignore_validation_logic=False)
            train_size = int(0.8 * len(pool_ds))
            val_size = len(pool_ds) - train_size
            generator = torch.Generator().manual_seed(args.seed)
            train_ds, val_ds = random_split(pool_ds, [train_size, val_size], generator=generator)
            workers = 0 if args.preload_ram else 4
            test_loader = DataLoader(test_ds, batch_size=args.micro_batch_size, shuffle=False, collate_fn=collate_windows, num_workers=workers)
        else:
            logger.info("Initializing FULL dataset (ignoring val_groups) for Random Split 80/20...")
            g_s, w_s = resolve_shuffling_params(args, 'train') 
            full_ds = MDFlexibleWindowDataset(args.data_class_dirs, split='train', val_groups=[], 
                                              window_size=args.window, window_offset=args.window_offset,
                                              stride=args.stride, skip=args.skip, preload_ram=args.preload_ram,
                                              global_shuffle=g_s, window_shuffle=w_s,
                                              logger=logger, ignore_validation_logic=True)
            train_size = int(0.8 * len(full_ds))
            val_size = len(full_ds) - train_size
            generator = torch.Generator().manual_seed(args.seed)
            train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=generator)
    else:
        logger.info("Initializing Standard LORO splits based on val_groups...")
        g_s_train, w_s_train = resolve_shuffling_params(args, 'train')
        train_ds = MDFlexibleWindowDataset(args.data_class_dirs, split='train', val_groups=args.val_groups,
                                           window_size=args.window, window_offset=args.window_offset,
                                           stride=args.stride, skip=args.skip, preload_ram=args.preload_ram,
                                           global_shuffle=g_s_train, window_shuffle=w_s_train, logger=logger)
        g_s_val, w_s_val = resolve_shuffling_params(args, 'val')
        val_ds = MDFlexibleWindowDataset(args.data_class_dirs, split='val', val_groups=args.val_groups,
                                         window_size=args.window, window_offset=args.window_offset,
                                         stride=args.stride, skip=args.skip, preload_ram=args.preload_ram,
                                         global_shuffle=g_s_val, window_shuffle=w_s_val, logger=logger)

    if len(train_ds) == 0:
        logger.error("CRITICAL ERROR: Train dataset is empty! Please check your --val_groups selection.")
        sys.exit(1)

    if len(val_ds) == 0:
        logger.error("CRITICAL ERROR: Validation dataset is empty! You need at least one valid group for validation to enable early stopping and evaluation. Please check your --val_groups selection.")
        sys.exit(1)

    logger.info("Verifying node alignment across input graphs...")
    check_node_alignment(args.data_class_dirs, args.data_class_labels, logger)

    sampler, shuffle = None, True
    if args.balance_classes:
        weights = train_ds.dataset.get_weights()[train_ds.indices] if isinstance(train_ds, Subset) else train_ds.get_weights()
        if len(weights) > 0:
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            shuffle = False

    workers = 0 if args.preload_ram else 4
    train_loader = DataLoader(train_ds, batch_size=args.micro_batch_size, shuffle=shuffle, sampler=sampler,
                              collate_fn=collate_windows, num_workers=workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.micro_batch_size, shuffle=False,
                            collate_fn=collate_windows, num_workers=workers, pin_memory=True)

    # --- 2. Phase 1: Contrastive Training ---
    num_classes_model = len(args.data_class_dirs) if args.contrastive == 'none' else None
    model = HybridStSchnet(
        hidden_dim=args.hidden_dim, embedding_dim=args.embedding_dim, window_size=args.window, n_layers=args.n_layers,
        cutoff=args.cutoff, dihedral_dim=args.dihedral_dim, use_checkpointing=args.use_checkpointing,
        pooling_type=args.pooling_type, dropout=args.dropout, attn_temperature=args.attn_temperature, 
        temporal_setup=args.temporal_setup, pooling_activation=args.pooling_activation, num_classes=num_classes_model
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.contrastive_learning_rate)
    criterion = nn.CrossEntropyLoss() if args.contrastive == 'none' else GroupContrastiveLoss(margin=args.margin, map_negative_name_to_exclude=args.map_negative_name_to_exclude, mode=args.contrastive)
    trainer = HybridTrainer(model, optimizer, criterion, device, logger, args=args)
    contrastive_history = trainer.fit(train_loader, val_loader, test_loader=test_loader, train_ds=train_ds)

    # --- 3. Phase 2: Linear Probe ---
    linear_history = None
    if args.contrastive != 'none':
        logger.info("\n" + "="*50 + "\nPhase 2: Linear Probing\n" + "="*50)
        linear_args = copy.deepcopy(args)
        linear_args.pretrained_path = os.path.join(args.out_dir, "best_model.pt")
        linear_args.lr = args.linear_learning_rate
        linear_args.epochs = args.linear_epochs
        
        g_s_lin, w_s_lin = resolve_shuffling_params(args, 'train')
        linear_args.global_shuffle = g_s_lin
        linear_args.window_shuffle = w_s_lin
        
        ds_source = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
        shared_cache = ds_source.ram_cache
        
        del model, optimizer, trainer
        torch.cuda.empty_cache()
        
        linear_history = run_linear_probe(linear_args, external_logger=logger, shared_cache=shared_cache)

    # Salva il plot della convergenza
    plot_training_convergence(contrastive_history, linear_history, args.out_dir)

    # --- 4. Post-Training xAI Analysis ---
    if args.contrastive != 'none':
        # Reinizializza un loader "pulito" e in ordine temporale per i test (shuffle = False)
        logger.info("Setting up Loaders for Analysis...")
        ana_train_loader = DataLoader(train_ds, batch_size=args.micro_batch_size, shuffle=False, collate_fn=collate_windows, num_workers=workers)
        ana_val_loader = DataLoader(val_ds, batch_size=args.micro_batch_size, shuffle=False, collate_fn=collate_windows, num_workers=workers)

        # Ricrea il backbone
        backbone = HybridStSchnet(
            hidden_dim=args.hidden_dim, embedding_dim=args.embedding_dim, window_size=args.window, n_layers=args.n_layers,
            cutoff=args.cutoff, dihedral_dim=args.dihedral_dim, use_checkpointing=args.use_checkpointing,
            pooling_type=args.pooling_type, dropout=args.dropout, attn_temperature=args.attn_temperature, 
            temporal_setup=args.temporal_setup, pooling_activation=args.pooling_activation, num_classes=None
        )
        
        # Inizializza il Classifier Lineare Completo e carica i pesi allenati
        num_classes = len(args.data_class_dirs)
        full_model = LinearProbeClassifier(backbone, num_classes, args.embedding_dim)
        
        linear_model_path = os.path.join(args.out_dir, "best_linear_model.pt")
        if os.path.exists(linear_model_path):
            full_model.load_state_dict(torch.load(linear_model_path, map_location=device, weights_only=True))
        
        full_model.to(device)
        
        # Lancia l'analisi integrata
        run_post_training_analysis(full_model, ana_train_loader, ana_val_loader, args)

if __name__ == "__main__":
    main()
