import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
import umap

from .dataset import MDFlexibleWindowDataset, collate_windows

def get_sequential_inference_data(model, args, device, logger):
    """
    Carica TUTTI i dati (Train + Val) in ordine strettamente sequenziale
    senza shuffling, per poter tracciare le traiettorie temporali.
    """
    logger.info("  [Analysis] Loading full sequential dataset for inference...")
    
    # 1. Dataset Train (tutte le repliche tranne val_groups)
    train_ds = MDFlexibleWindowDataset(
        data_class_dirs=args.data_class_dirs, split='train', val_groups=args.val_groups,
        window_size=args.window, window_offset=args.window_offset, stride=args.stride,
        skip=args.skip, 
        global_shuffle=False, window_shuffle=False # CRITICO: No shuffle
    )
    
    # 2. Dataset Val (solo repliche in val_groups)
    loaders = [('train', DataLoader(train_ds, batch_size=args.micro_batch_size, shuffle=False, collate_fn=collate_windows, num_workers=4))]
    
    if len(args.val_groups) > 0:
        val_ds = MDFlexibleWindowDataset(
            data_class_dirs=args.data_class_dirs, split='val', val_groups=args.val_groups,
            window_size=args.window, window_offset=args.window_offset, stride=args.stride,
            skip=args.skip, 
            global_shuffle=False, window_shuffle=False # CRITICO: No shuffle
        )
        loaders.append(('validation', DataLoader(val_ds, batch_size=args.micro_batch_size, shuffle=False, collate_fn=collate_windows, num_workers=4)))
    
    data_store = {k: [] for k in ['embeddings', 'logits', 'probs', 'energy', 'labels', 'groups', 'replica_names', 'splits']}
    
    model.eval()
    with torch.no_grad():
        for split_name, loader in loaders:
            for batch_data, labels, groups, _, batch_paths in tqdm(loader, desc="[Analysis] Inference", leave=False):
                batch_data = batch_data.to(device)
                
                # Attenzione: il LinearProbeClassifier restituisce i logits. 
                # Per avere anche le features serve estrarle dal backbone.
                features = model.backbone(batch_data)
                logits = model.head(features)
                
                probs = F.softmax(logits, dim=1)
                
                # ENERGY SCORE (Liu et al. 2020)
                energy = -torch.logsumexp(logits, dim=1)
                
                data_store['embeddings'].append(features.cpu().numpy())
                data_store['logits'].append(logits.cpu().numpy())
                data_store['probs'].append(probs.cpu().numpy())
                data_store['energy'].append(energy.cpu().numpy())
                data_store['labels'].append(labels.numpy())
                data_store['groups'].append(groups.numpy())
                
                # Fetch replica names natively from folder names
                for paths in batch_paths:
                    first_frame_path = paths[0]
                    replica_name = os.path.basename(os.path.dirname(first_frame_path))
                    data_store['replica_names'].append(replica_name)
                    data_store['splits'].append(split_name)

    # Convert array keys
    for k in ['embeddings', 'logits', 'probs', 'energy', 'labels', 'groups']:
        data_store[k] = np.concatenate(data_store[k])
    
    # Replica names and splits are lists of strings
    data_store['replica_names'] = np.array(data_store['replica_names'])
    data_store['splits'] = np.array(data_store['splits'])
    
    return data_store

def get_global_to_local_map(labels, groups, class_labels=None):
    unique_groups = np.unique(groups)
    global_to_class = {}
    
    for g in unique_groups:
        mask = (groups == g)
        if np.any(mask):
            global_to_class[g] = labels[mask][0]
            
    global_to_local = {}
    unique_classes = np.unique(list(global_to_class.values()))
    
    for c in unique_classes:
        c_groups = sorted([g for g in unique_groups if global_to_class.get(g) == c])
        for i, g in enumerate(c_groups):
            global_to_local[g] = i + 1
            
    return global_to_local, global_to_class

def plot_complex_grid(proj_data, proj_data_dict, method_name, val_groups_str, output_dir, num_replicas, class_labels=None):
    labels = proj_data_dict['labels']
    groups = proj_data_dict['groups']
    """Genera la griglia N righe x M colonne dinamicamente in base alle classi."""
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    
    fig, axes = plt.subplots(num_classes, num_replicas, figsize=(5 * num_replicas, 4 * num_classes), dpi=150)
    global_to_local, global_to_class = get_global_to_local_map(labels, groups, class_labels)
    time_cmap = plt.get_cmap("coolwarm")
    bg_color = "#E0E0E0"
    
    # Gestione degli axes se 1D
    if num_classes == 1: axes = np.expand_dims(axes, axis=0)
    if num_replicas == 1: axes = np.expand_dims(axes, axis=1)

    for c_idx, class_idx in enumerate(unique_classes):
        for replica_idx in range(1, num_replicas + 1):
            ax = axes[c_idx, replica_idx - 1]
            ax.scatter(proj_data[:, 0], proj_data[:, 1], c=bg_color, s=2, alpha=0.3, rasterized=True)
            
            target_global_id = None
            for g, l in global_to_local.items():
                if l == replica_idx and global_to_class.get(g) == class_idx:
                    target_global_id = g
                    break
            
            if target_global_id is not None:
                mask = (groups == target_global_id)
                if np.sum(mask) > 0:
                    points = proj_data[mask]
                    num_points = len(points)
                    time_indices = np.linspace(0, 1, num_points)
                    ax.scatter(points[:, 0], points[:, 1], c=time_indices, cmap=time_cmap, s=5, alpha=0.8)
            
            class_name = class_labels[class_idx] if class_labels and class_idx < len(class_labels) else f"Class {class_idx}"
            
            # Fetch replica name
            mask = (groups == target_global_id) if target_global_id is not None else []
            if np.sum(mask) > 0:
                rep_name_arr = proj_data_dict.get('replica_names', None)
                rep_name = rep_name_arr[mask][0] if rep_name_arr is not None else f"Rep {replica_idx}"
                split_arr = proj_data_dict.get('splits', None)
                split_name = split_arr[mask][0].upper()[:3] if split_arr is not None else ""
                title_lbl = f"{class_name}\n({split_name}) {rep_name}"
            else:
                title_lbl = f"{class_name}\nRep {replica_idx}"

            ax.set_title(title_lbl, fontsize=10)
            ax.axis('off')

    plt.suptitle(f"{method_name} Space (Time: Blue->Red)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(output_dir, f"{method_name}_space_grid.png")
    plt.savefig(save_path)
    plt.close()

def save_and_plot_confidence(data, val_groups_str, output_dir, class_labels=None):
    groups = data['groups']
    labels = data['labels']
    all_energy = data['energy']
    replica_names = data.get('replica_names', np.array([f"rep{g}" for g in groups]))
    splits = data.get('splits', np.array(["" for _ in groups]))

    unique_groups = np.unique(groups)
    
    for grp in unique_groups:
        grp_mask = (groups == grp)
        grp_labels = labels[grp_mask]
        
        if len(np.unique(grp_labels)) > 1: continue
            
        cls = grp_labels[0]
        
        rep_logits = data['logits'][grp_mask]
        rep_probs = data['probs'][grp_mask]
        rep_embs = data['embeddings'][grp_mask]
        rep_energy = all_energy[grp_mask]
        
        rep_name = replica_names[grp_mask][0] if len(replica_names[grp_mask]) > 0 else f"rep{grp}"
        split_name = splits[grp_mask][0] if len(splits[grp_mask]) > 0 else "train"
        
        preds = np.argmax(rep_probs, axis=1)
        acc = (np.sum(preds == cls) / len(preds)) * 100.0 if len(preds) > 0 else 0.0
        
        # CSV Export
        confidence_trace = rep_probs[:, cls]
        df_replica = pd.DataFrame({
            'frame_idx': np.arange(len(rep_probs)),
            'true_label': cls,
            'predicted_label': preds,
            'confidence_score': confidence_trace,
            'energy_score': rep_energy
        })
        for c_idx in range(rep_probs.shape[1]):
            col_name = f'prob_{class_labels[c_idx]}' if class_labels and c_idx < len(class_labels) else f'prob_class_{c_idx}'
            df_replica[col_name] = rep_probs[:, c_idx]

        class_name = class_labels[cls] if class_labels and cls < len(class_labels) else f"class{cls}"
        
        base_name = f"{split_name}_{class_name}_{rep_name}"
        if not split_name: base_name = f"val{val_groups_str}_{class_name}_{rep_name}"
        
        df_replica.to_csv(os.path.join(output_dir, f"{base_name}_data.csv"), index=False)
        
        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(confidence_trace, color='#2ca02c', linewidth=0.8, alpha=0.9, label=f'Prob({class_name})')
        
        if len(confidence_trace) > 50:
            window = 50
            smooth = np.convolve(confidence_trace, np.ones(window)/window, mode='valid')
            plt.plot(np.arange(len(smooth)) + window//2, smooth, color='black', linewidth=1.5, alpha=0.6, label='Moving Avg')
        
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.5)
        plt.ylim(-0.05, 1.05)
        
        title_prefix = f"[{split_name.upper()}] " if split_name else f"Val Groups {val_groups_str} | "
        plt.title(f"{title_prefix}{class_name} | {rep_name} | Accuracy: {acc:.2f}%")
        plt.ylabel(f"Prob({class_name})")
        plt.xlabel("Time (Window Index)")
        plt.legend()
        plt.grid(True, alpha=0.2)
        
        plt.savefig(os.path.join(output_dir, f"{base_name}_conf.png"))
        plt.close()

def run_post_training_analysis(model, args, device, logger):
    """
    Entry point per eseguire l'analisi finale dopo il training.
    """
    logger.info("\n" + "="*50)
    logger.info("PHASE 3: POST-TRAINING ANALYSIS & LATENT SPACE")
    logger.info("="*50)
    
    # 1. Setup Cartelle
    latent_dir = os.path.join(args.out_dir, "analysis", "latent_space")
    conf_dir = os.path.join(args.out_dir, "analysis", "confidence")
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)
    
    val_str = "_".join(map(str, args.val_groups)) if args.val_groups else "none"
    
    # 2. Estrazione Dati Sequenziali
    data = get_sequential_inference_data(model, args, device, logger)
    
    # 3. PCA & UMAP
    logger.info("  [Analysis] Computing PCA & UMAP...")
    pca = PCA(n_components=2)
    pca_proj = pca.fit_transform(data['embeddings'])
    
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, metric='cosine', random_state=args.seed)
    umap_proj = reducer.fit_transform(data['embeddings'])
    
    # 4. Salvataggio Globale (NPZ e CSV) in latent_space
    logger.info("  [Analysis] Exporting Global Artifacts...")
    np.savez_compressed(
        os.path.join(latent_dir, f"val_{val_str}_global_data.npz"),
        **data, pca_proj=pca_proj, umap_proj=umap_proj
    )
    
    df_global = pd.DataFrame({
        'group_id': data['groups'],
        'label': data['labels'],
        'energy_score': data['energy'],
        'pca_x': pca_proj[:, 0], 'pca_y': pca_proj[:, 1],
        'umap_x': umap_proj[:, 0], 'umap_y': umap_proj[:, 1]
    })
    df_global.to_csv(os.path.join(latent_dir, f"val_{val_str}_global_coords.csv"), index=False)
    
    
    class_labels = getattr(args, 'data_class_labels', None)
    
    # 5. Plot Griglie
    logger.info("  [Analysis] Generating PCA & UMAP Grid Plots...")
    plot_complex_grid(pca_proj, data, "PCA", val_str, latent_dir, args.num_replicas, class_labels)
    plot_complex_grid(umap_proj, data, "UMAP", val_str, latent_dir, args.num_replicas, class_labels)
    
    # 6. Plot Confidence
    logger.info("  [Analysis] Generating Per-Replica Confidence Analysis...")
    save_and_plot_confidence(data, val_str, conf_dir, class_labels)

    logger.info(f"✅ Analysis Complete! Results saved in: {os.path.join(args.out_dir, 'analysis')}")
