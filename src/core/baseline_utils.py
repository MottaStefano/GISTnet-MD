import torch
import numpy as np
from tqdm import tqdm
from core.dataset import MDFlexibleWindowDataset, collate_windows
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from torch_geometric.data import Batch

def compute_global_baseline(config, wrapper, device):
    """
    Calcola una baseline termodinamica 'class-specific'.
    Per ogni classe C, la baseline è la media degli stati di tutte le altre classi (!= C).
    Utilizza uno stride di 5 per velocizzare l'estrazione.
    """
    ds_train = MDFlexibleWindowDataset(
        config.get('data_class_dirs', config.get('data_dir')), split='train',
        val_groups=config.get('val_groups', []), window_size=config.get('window', 10),
        window_offset=config.get('window_offset', None), stride=5, logger=None, skip=0, # <-- STRIDE = 5
        preload_ram=config.get('preload_ram', False), global_shuffle=False, window_shuffle=False
    )
    if len(ds_train) == 0:
        return None

    class_edge_sum = {}
    class_edge_count = {}
    class_graph_count = {}

    loader_train = torch.utils.data.DataLoader(ds_train, batch_size=8, shuffle=False, collate_fn=collate_windows)
    for batched_data, labels, groups, shared_names, batch_paths in tqdm(loader_train, desc="Computing Training Baselines (Stride 5)"):
        batched_data = batched_data.to(device)

        num_graphs_in_batch = batched_data.num_graphs
        num_nodes_per_frame = batched_data.num_nodes // num_graphs_in_batch
        logical_batch_size = labels.shape[0]
        frames_per_sample = num_graphs_in_batch // logical_batch_size

        real_dist = batched_data.edge_attr.clone().detach()
        if real_dist.dim() > 1: real_dist = real_dist.squeeze()

        with torch.no_grad():
            rbf_val = wrapper.spatial.rbf(real_dist)
            env_val = wrapper.spatial.envelope(real_dist, wrapper.spatial.cutoff).unsqueeze(-1)
            target_rbf_feat = (rbf_val * env_val).detach()

        edges = batched_data.edge_index.t().cpu().numpy()
        feats = target_rbf_feat.cpu().numpy()
        batch_classes = labels.cpu().numpy()

        # Inizializza dizionari se nuova classe trovata
        for c in np.unique(batch_classes):
            if c not in class_edge_sum:
                class_edge_sum[c] = {}
                class_edge_count[c] = {}
                class_graph_count[c] = 0

        # Mappatura archi e accumulo per classe
        for u, v, feat in zip(edges[:, 0], edges[:, 1], feats):
            graph_idx = u // num_nodes_per_frame
            sample_idx = graph_idx // frames_per_sample
            c = batch_classes[sample_idx]

            u_canon = int(u) % num_nodes_per_frame
            v_canon = int(v) % num_nodes_per_frame
            edge_tuple = (u_canon, v_canon)

            if edge_tuple not in class_edge_sum[c]:
                class_edge_sum[c][edge_tuple] = np.zeros_like(feat)
                class_edge_count[c][edge_tuple] = 0

            class_edge_sum[c][edge_tuple] += feat
            class_edge_count[c][edge_tuple] += 1

        # Aggiorna il conto dei grafi elaborati per classe
        for g_idx in range(logical_batch_size):
            c = batch_classes[g_idx]
            class_graph_count[c] += frames_per_sample

    if not class_graph_count: return None

    # Costruisci le baseline: per la classe target C, media le feature di tutte le classi != C
    baselines_per_class = {}
    all_classes = list(class_edge_sum.keys())

    for target_c in all_classes:
        other_classes = [c for c in all_classes if c != target_c]
        if not other_classes:
            other_classes = [target_c] # Fallback se c'è solo una classe (anomalia)

        target_edge_sum = {}
        target_edge_count = {}
        target_total_graphs = sum(class_graph_count[c] for c in other_classes)

        for c in other_classes:
            for edge, feat in class_edge_sum[c].items():
                if edge not in target_edge_sum:
                    target_edge_sum[edge] = np.zeros_like(feat)
                    target_edge_count[edge] = 0
                target_edge_sum[edge] += feat
                target_edge_count[edge] += class_edge_count[c][edge]

        # Filtra il rumore (frequenza < 5% dei frame totali analizzati)
        min_occurences = target_total_graphs * 0.05
        sorted_edges = sorted([k for k, v in target_edge_count.items() if v >= min_occurences])

        if sorted_edges:
            mean_baseline_edge_index = torch.tensor(sorted_edges, dtype=torch.long, device=device).t().contiguous()
            mean_baseline_rbf = []
            for edge in sorted_edges:
                mean_feat = target_edge_sum[edge] / target_total_graphs
                mean_baseline_rbf.append(torch.tensor(mean_feat, dtype=torch.float32, device=device))
            mean_baseline_rbf = torch.stack(mean_baseline_rbf)
        else:
            mean_baseline_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            mean_baseline_rbf = torch.empty((0, target_rbf_feat.shape[-1]), dtype=torch.float32, device=device)

        baselines_per_class[target_c] = (mean_baseline_edge_index, mean_baseline_rbf)

    return baselines_per_class

def inject_ghost_edges(batched_data, target_rbf_feat, mean_baseline_edge_index, device):
    """
    Expands the current temporal graph batch to include all edges from the baseline union
    across ALL frames in the sliding window, injecting zeros for missing RBF features.
    """
    num_graphs_in_batch = batched_data.num_graphs
    num_nodes_per_frame = batched_data.num_nodes // num_graphs_in_batch

    edges_np = batched_data.edge_index.t().cpu().numpy()
    current_edges = [(int(u), int(v)) for u, v in edges_np]
    current_feat_dict = {edge: feat for edge, feat in zip(current_edges, target_rbf_feat)}

    baseline_np = mean_baseline_edge_index.t().cpu().numpy()
    baseline_edges_canonical = [(int(u), int(v)) for u, v in baseline_np]

    full_aligned_rbf_feats = []
    full_baseline_edge_list = []

    for g_idx in range(num_graphs_in_batch):
        offset = g_idx * num_nodes_per_frame
        for u_canon, v_canon in baseline_edges_canonical:
            shifted_u = u_canon + offset
            shifted_v = v_canon + offset
            shifted_edge = (shifted_u, shifted_v)

            full_baseline_edge_list.append([shifted_u, shifted_v])

            if shifted_edge in current_feat_dict:
                full_aligned_rbf_feats.append(current_feat_dict[shifted_edge])
            else:
                full_aligned_rbf_feats.append(torch.zeros_like(target_rbf_feat[0]))

    aligned_rbf_tensor = torch.stack(full_aligned_rbf_feats)
    full_baseline_edge_index = torch.tensor(full_baseline_edge_list, dtype=torch.long, device=device).t().contiguous()

    batched_data.edge_index = full_baseline_edge_index
    batched_data.edge_attr = aligned_rbf_tensor

    return batched_data, aligned_rbf_tensor

def select_medoid_baselines(config, full_model, device, num_baselines=5, confidence_threshold=0.90):
    """
    Filtra il dataset di training per classe, assicurandosi che le window siano classificate
    correttamente e con elevata confidenza. Usa la backbone per estrarne gli embedding globali,
    esegue il K-Means e recupera le window (Batch PyG reali) più vicine ai centroidi (Medoidi).
    Questi K medoidi agiranno da real background baselines per calcolare gli Expected Gradients.
    """
    ds_train = MDFlexibleWindowDataset(
        config.get('data_class_dirs', config.get('data_dir')), split='train',
        val_groups=config.get('val_groups', []), window_size=config.get('window', 10),
        window_offset=config.get('window_offset', None), stride=5, logger=None, skip=0,
        preload_ram=config.get('preload_ram', False), global_shuffle=False, window_shuffle=False
    )
    if len(ds_train) == 0:
        return {}

    loader_train = torch.utils.data.DataLoader(ds_train, batch_size=16, shuffle=False, collate_fn=collate_windows)

    class_embeddings = {}
    class_indices = {}
    full_model.eval()
    idx_offset = 0

    for batched_data, labels, groups, shared_names, batch_paths in tqdm(loader_train, desc="Extracting Embeddings for Medoids"):
        batched_data = batched_data.to(device)
        with torch.no_grad():
            features = full_model.backbone(batched_data) # Global Graph Embeddings
            logits = full_model.head(features)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            confs = probs[torch.arange(logits.size(0)), preds]

        feats_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()
        confs_np = confs.cpu().numpy()

        for i, (feat, label, pred, conf) in enumerate(zip(feats_np, labels_np, preds_np, confs_np)):
            # Applica il filtro di purezza: predizione corretta e alta confidenza
            if pred == label and conf >= confidence_threshold:
                if label not in class_embeddings:
                    class_embeddings[label] = []
                    class_indices[label] = []
                class_embeddings[label].append(feat)
                class_indices[label].append(idx_offset + i)

        # L'offset deve avanzare dell'intera dimensione del batch processato per mantenere
        # allineati gli indici rispetto al dataset originale
        idx_offset += len(labels_np)

    medoids_per_class = {}
    for c, feats in class_embeddings.items():
        feats_array = np.array(feats)
        n_clusters = min(num_baselines, len(feats_array))
        if n_clusters == 0:
            continue

        # Identifica i K cluster latenti della classe C
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(feats_array)
        # Trova il sample reale più vicino a ciascun centroide (Il Medoide)
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, feats_array)
        
        labels = kmeans.labels_
        cluster_counts = np.bincount(labels)
        total_samples = len(feats_array)
        medoid_weights = [cluster_counts[i] / total_samples for i in range(len(closest))]

        medoid_idxs = [class_indices[c][i] for i in closest]

        medoid_batches = []
        for idx in medoid_idxs:
            # Recupera la riga dal dataset e la passa al collate function in modo nativo e sicuro
            batch_item = ds_train[idx]
            batched_data, _, _, _, _ = collate_windows([batch_item])
            medoid_batches.append(batched_data)

        medoids_per_class[c] = (medoid_batches, medoid_weights)

    return medoids_per_class

def inject_multi_baseline_ghost_edges(target_batch, target_rbf_feat, baselines, wrapper, device):
    """
    Produce un'Unione Topologica Globale tra la target window e TUTTE le K baseline.
    Restituisce un batched_data allineato per la target e una lista di tensori RBF per i medoidi.
    L'allineamento globale è essenziale per poter mediare i gradienti (vettori edge_imp avranno uguale shape).
    """
    # 1. Recupera gli archi della target window
    target_edges = list(map(tuple, target_batch.edge_index.t().cpu().numpy()))
    union_edges = set(target_edges)
    target_feat_dict = {edge: feat for edge, feat in zip(target_edges, target_rbf_feat)}

    # 2. Estrae RBF e archi per ciascun medoide, fondendo le topologie nell'unione
    baseline_feat_dicts = []
    for b_batch in baselines:
        b_batch = b_batch.to(device)
        real_dist = b_batch.edge_attr.clone().detach()
        if real_dist.dim() > 1: real_dist = real_dist.squeeze()

        with torch.no_grad():
            rbf_val = wrapper.spatial.rbf(real_dist)
            env_val = wrapper.spatial.envelope(real_dist, wrapper.spatial.cutoff).unsqueeze(-1)
            b_rbf_feat = (rbf_val * env_val).detach()

        b_edges = list(map(tuple, b_batch.edge_index.t().cpu().numpy()))
        union_edges.update(b_edges)
        baseline_feat_dicts.append({edge: feat for edge, feat in zip(b_edges, b_rbf_feat)})

    union_edges = sorted(list(union_edges))

    # 3. Costruisce i tensori allineati (Zero Injection per archi mancanti nel set locale)
    full_aligned_target_rbf = []
    full_aligned_baseline_rbfs = [[] for _ in range(len(baselines))]

    zero_feat = torch.zeros_like(target_rbf_feat[0])

    for edge in union_edges:
        full_aligned_target_rbf.append(target_feat_dict.get(edge, zero_feat))
        for i, b_dict in enumerate(baseline_feat_dicts):
            full_aligned_baseline_rbfs[i].append(b_dict.get(edge, zero_feat))

    aligned_target_rbf = torch.stack(full_aligned_target_rbf)
    aligned_baseline_rbfs = [torch.stack(b) for b in full_aligned_baseline_rbfs]

    union_edge_index = torch.tensor(union_edges, dtype=torch.long, device=device).t().contiguous()
    target_batch.edge_index = union_edge_index
    target_batch.edge_attr = aligned_target_rbf

    return target_batch, aligned_target_rbf, aligned_baseline_rbfs
