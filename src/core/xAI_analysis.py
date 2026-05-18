import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import umap
from tqdm import tqdm
import re

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_meaningful_name(paths, global_shuffle=False):
    """Genera un nome file indicativo basato sui percorsi dei frame nella window."""
    if not paths:
        return f"embedding_window_{np.random.randint(100000)}"

    first_path = paths[0]
    parts = first_path.split(os.sep)

    # Se global_shuffle è attivo, la window è un mix. Mettiamo solo la cartella root o un hash
    if global_shuffle:
        cls_name = parts[-3] if len(parts) >= 3 else "mixed"
        rep_name = parts[-2] if len(parts) >= 2 else "mixed"
        return f"embedding_{cls_name}_{rep_name}_mixed_window_{hash(tuple(paths)) % 100000}"

    # Comportamento normale: estrai classe, gruppo e range di frame
    try:
        cls_name = parts[-3]
        grp_name = parts[-2]
        file_name = parts[-1]

        # Estrai il prefisso del file (es. md25) e l'ID del primo frame
        match = re.search(r"(.+)_frame_(\d+)\.pt", file_name)
        if match:
            prefix = match.group(1)
            start_frame = match.group(2)

            # ID dell'ultimo frame
            last_path = paths[-1]
            last_file_name = last_path.split(os.sep)[-1]
            last_match = re.search(r"(.+)_frame_(\d+)\.pt", last_file_name)
            end_frame = last_match.group(2) if last_match else "X"

            return f"embedding_{cls_name}_{grp_name}_{prefix}_{start_frame}-{end_frame}"
        else:
            return f"embedding_window_{hash(tuple(paths)) % 100000}"
    except:
        return f"embedding_window_{hash(tuple(paths)) % 100000}"

def extract_features(model, loader, is_shuffled, split_name):
    """Estrazione feature per un singolo loader (train o val), includendo il nome dello split."""
    data_store = {k: [] for k in ['embeddings', 'logits', 'probs', 'energy', 'labels', 'groups', 'names', 'splits', 'group_names']}

    with torch.no_grad():
        for batch_data, labels, groups, _, paths in tqdm(loader, desc="Inference", leave=False):
            batch_data = batch_data.to(DEVICE)
            # Chiamata al forward modificato di LinearProbeClassifier in train_linear.py
            # che nel file era "return logits". Modifichiamolo dinamicamente o assumiamo
            # che il LinearProbeClassifier esponga il backbone.

            embs = model.backbone(batch_data)
            logits = model.head(embs)
            probs = F.softmax(logits, dim=1)
            energy = -torch.logsumexp(logits, dim=1)

            data_store['embeddings'].append(embs.cpu().numpy())
            data_store['logits'].append(logits.cpu().numpy())
            data_store['probs'].append(probs.cpu().numpy())
            data_store['energy'].append(energy.cpu().numpy())
            data_store['labels'].append(labels.numpy())
            data_store['groups'].append(groups.numpy())

            # Nomi file per ogni window del batch
            for window_paths in paths:
                data_store['names'].append(extract_meaningful_name(window_paths, is_shuffled))
                
                # Salviamo Split e Group folder name
                data_store['splits'].append(split_name)
                if is_shuffled or not window_paths:
                    data_store['group_names'].append("mixed")
                else:
                    parts = window_paths[0].split(os.sep)
                    data_store['group_names'].append(parts[-2] if len(parts) >= 2 else "mixed")

    if len(data_store['embeddings']) == 0:
        return None

    return {
        'embeddings': np.concatenate(data_store['embeddings']),
        'logits': np.concatenate(data_store['logits']),
        'probs': np.concatenate(data_store['probs']),
        'energy': np.concatenate(data_store['energy']),
        'labels': np.concatenate(data_store['labels']),
        'groups': np.concatenate(data_store['groups']),
        'names': data_store['names'],
        'splits': data_store['splits'],
        'group_names': data_store['group_names']
    }

def get_global_to_local_map(labels, groups):
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

def plot_complex_grid(proj_data, data_store, class_labels, method_name, output_dir, num_groups=5):
    labels, groups = data_store['labels'], data_store['groups']
    group_names_arr = np.array(data_store['group_names'])
    
    unique_classes = np.unique(labels)
    fig, axes = plt.subplots(len(unique_classes), num_groups, figsize=(20, 4*len(unique_classes)), dpi=150)
    global_to_local, global_to_class = get_global_to_local_map(labels, groups)
    time_cmap = plt.get_cmap("coolwarm")
    bg_color = "#E0E0E0"

    if num_groups == 1:
        axes = np.expand_dims(axes, axis=1)
    if len(unique_classes) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, class_idx in enumerate(unique_classes):
        for group_idx in range(1, num_groups + 1):
            ax = axes[row_idx, group_idx - 1]
            ax.scatter(proj_data[:, 0], proj_data[:, 1], c=bg_color, s=2, alpha=0.3, rasterized=True)

            target_global_id = None
            for g, l in global_to_local.items():
                if l == group_idx and global_to_class.get(g) == class_idx:
                    target_global_id = g
                    break

            class_str = class_labels[class_idx] if class_idx < len(class_labels) else f"Class_{class_idx}"
            display_title = f"{class_str} - Group {group_idx}"

            if target_global_id is not None:
                mask = (groups == target_global_id)
                if np.sum(mask) > 0:
                    points = proj_data[mask]
                    num_points = len(points)
                    time_indices = np.linspace(0, 1, num_points)
                    ax.scatter(points[:, 0], points[:, 1], c=time_indices, cmap=time_cmap, s=5, alpha=0.8)
                    
                    grp_name = group_names_arr[mask][0]
                    display_title = f"{class_str} | {grp_name}"

            ax.set_title(display_title)
            ax.axis('off')

    plt.suptitle(f"{method_name} Space (Colored by Time: Blue->Red)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(output_dir, f"{method_name}_grid.png")
    plt.savefig(save_path)
    plt.close()

def save_and_plot_confidence(data, class_labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    groups, labels = data['groups'], data['labels']
    all_energy = data['energy']
    splits_arr = np.array(data['splits'])
    group_names_arr = np.array(data['group_names'])
    
    unique_groups = np.unique(groups)

    for grp in unique_groups:
        grp_mask = (groups == grp)
        grp_labels = labels[grp_mask]

        if len(np.unique(grp_labels)) > 1:
            continue

        cls = grp_labels[0]
        cls_str = class_labels[cls] if cls < len(class_labels) else f"class{cls}"
        
        grp_name = group_names_arr[grp_mask][0]
        split_name = splits_arr[grp_mask][0]

        grp_probs = data['probs'][grp_mask]
        grp_energy = all_energy[grp_mask]

        preds = np.argmax(grp_probs, axis=1)
        correct = (preds == cls).sum()
        total = len(preds)
        acc = (correct / total) * 100.0 if total > 0 else 0.0

        confidence_trace = grp_probs[:, cls]

        df_group = pd.DataFrame({
            'frame_idx': np.arange(len(grp_probs)),
            'true_label': cls,
            'predicted_label': preds,
            'confidence_score': confidence_trace,
            'energy_score': grp_energy
        })

        num_classes_in_probs = grp_probs.shape[1]
        for c_idx in range(num_classes_in_probs):
            df_group[f'prob_class_{c_idx}'] = grp_probs[:, c_idx]

        fname_csv = f"{split_name}_{cls_str}_{grp_name}_data.csv"
        df_group.to_csv(os.path.join(output_dir, fname_csv), index=False)

        plt.figure(figsize=(10, 4))
        plt.plot(confidence_trace, color='#2ca02c', linewidth=0.8, alpha=0.9)

        if len(confidence_trace) > 50:
            window = 50
            smooth = np.convolve(confidence_trace, np.ones(window)/window, mode='valid')
            plt.plot(np.arange(len(smooth)) + window//2, smooth, color='black', linewidth=1.5, alpha=0.6, label='Moving Avg')

        plt.axhline(0.5, color='red', linestyle='--', alpha=0.5)
        plt.ylim(-0.05, 1.05)
        plt.title(f"{cls_str} | {split_name.capitalize()} | {grp_name} | Accuracy: {acc:.2f}%")
        plt.ylabel(f"Prob(Class={cls_str})")
        plt.xlabel("Time (Window Index)")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.savefig(os.path.join(output_dir, f"{split_name}_{cls_str}_{grp_name}_conf.png"))
        plt.close()

def save_raw_embeddings(data, names, out_dir):
    """Salva i singoli embedding usando numpy."""
    os.makedirs(out_dir, exist_ok=True)
    for emb, name in zip(data['embeddings'], names):
        file_path = os.path.join(out_dir, f"{name}.npy")
        np.save(file_path, emb)

def run_post_training_analysis(model, train_loader, val_loader, args):
    """
    Esegue l'analisi xAI, la confidence analysis e opzionalmente salva gli embeddings.
    """
    print("\n" + "="*50)
    print("STARTING POST-TRAINING ANALYSIS")
    print("="*50)
    model.eval()

    # Crea cartelle di output
    xai_dir = os.path.join(args.out_dir, "xai_analysis")
    conf_dir = os.path.join(args.out_dir, "confidence_analysis")
    os.makedirs(xai_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)

    # 1. Estrazione Dati
    is_shuffled = args.shuffling in ['global', 'windows'] or args.global_shuffle or args.window_shuffle
    print("Extracting features from Train Set...")
    train_data = extract_features(model, train_loader, is_shuffled=is_shuffled, split_name='training')
    print("Extracting features from Validation Set...")
    val_data = extract_features(model, val_loader, is_shuffled=is_shuffled, split_name='validation')

    # Combina i dati per PCA/UMAP
    if train_data and val_data:
        full_data = {k: np.concatenate((train_data[k], val_data[k])) for k in train_data if k not in ['names', 'splits', 'group_names']}
        for string_key in ['names', 'splits', 'group_names']:
            full_data[string_key] = train_data[string_key] + val_data[string_key]
    elif train_data: full_data = train_data
    elif val_data: full_data = val_data
    else:
        print("Error: No data extracted.")
        return
        
    class_labels = getattr(args, 'data_class_labels', [f"Class_{i}" for i in range(len(np.unique(full_data['labels'])))])

    # 2. Confidence Plot (Sia train che val, tipicamente su val_groups)
    print("Generating Confidence Plots and CSVs...")
    save_and_plot_confidence(full_data, class_labels, conf_dir)

    # 3. PCA & UMAP (Globale)
    print("Computing PCA & UMAP Latent Spaces...")
    pca = PCA(n_components=2)
    pca_proj = pca.fit_transform(full_data['embeddings'])

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
    umap_proj = reducer.fit_transform(full_data['embeddings'])

    df_global = pd.DataFrame({
        'group_id': full_data['groups'],
        'label': full_data['labels'],
        'energy_score': full_data['energy'],
        'pca_x': pca_proj[:, 0],
        'pca_y': pca_proj[:, 1],
        'umap_x': umap_proj[:, 0],
        'umap_y': umap_proj[:, 1]
    })
    df_global.to_csv(os.path.join(xai_dir, "global_coords.csv"), index=False)

    # Quanti gruppi visualizzare sulla griglia? Diciamo max 5
    num_shown_groups = len(np.unique(full_data['groups'])) // len(np.unique(full_data['labels']))
    num_shown_groups = min(num_shown_groups, 5) if num_shown_groups > 0 else 5

    plot_complex_grid(pca_proj, full_data, class_labels, "PCA", xai_dir, num_shown_groups)
    plot_complex_grid(umap_proj, full_data, class_labels, "UMAP", xai_dir, num_shown_groups)

    # 4. Salvataggio Embeddings Individuali
    if getattr(args, 'save_embeddings', False):
        print("Saving Raw Embeddings (Train & Val)...")
        emb_train_dir = os.path.join(args.out_dir, "embeddings", "train")
        emb_val_dir = os.path.join(args.out_dir, "embeddings", "val")

        if train_data: save_raw_embeddings(train_data, train_data['names'], emb_train_dir)
        if val_data: save_raw_embeddings(val_data, val_data['names'], emb_val_dir)
        print(f"Embeddings saved in {os.path.join(args.out_dir, 'embeddings')}")

    print("Post-Training Analysis Complete. All artifacts saved.")
