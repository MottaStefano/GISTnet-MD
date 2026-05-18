import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_add
from torch_geometric.data import Batch

# =============================================================================
# WRAPPERS E FUNZIONI CORE PER INTEGRATED GRADIENTS
# =============================================================================

class LinearProbeClassifier(nn.Module):
    def __init__(self, backbone, num_classes, embedding_dim):
        super().__init__()
        self.backbone = backbone
        # Congela la backbone durante IG per preservare i gradienti corretti
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = nn.Sequential(
            nn.BatchNorm1d(embedding_dim, affine=True),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, batched_data):
        self.backbone.eval()
        features = self.backbone(batched_data)
        logits = self.head(features)
        return logits

class IGInteractionWrapper(nn.Module):
    """
    Wrapper per calcolare i gradienti rispetto all'espansione RBF degli archi.
    Non altera le identità dei nodi né le distanze fisiche.
    """
    def __init__(self, full_model):
        super().__init__()
        self.model = full_model
        self.backbone = full_model.backbone
        self.spatial = self.backbone.spatial_encoder

    def forward(self, rbf_feat_tensor, data_batch):
        # 1. Calcolo embedding nodi statici
        h = self.spatial.embedding(data_batch.x)
        if self.spatial.dihedral_dim > 0 and hasattr(data_batch, 'x_dihe'):
            dihe_emb = self.spatial.dihedral_norm(self.spatial.dihedral_proj(data_batch.x_dihe))
            h = h + dihe_emb

        edge_index = data_batch.edge_index
        rbf_feat = rbf_feat_tensor

        # 2. Message passing con RBF modulate
        for layer in self.spatial.layers:
            h = layer(h, edge_index, rbf_feat)

        node_features = self.spatial.atom_norm(h)

        # 3. Aggregazione temporale
        x_dense, mask = to_dense_batch(node_features, data_batch.batch)
        total_graphs, max_nodes, _ = x_dense.size()
        batch_size_logical = total_graphs // self.backbone.window_size

        x = x_dense.view(batch_size_logical, self.backbone.window_size, max_nodes, self.backbone.hidden_dim)

        if self.backbone.temporal_setup == 'cnn':
            x = x.permute(0, 2, 3, 1)
            x_temporal_in = x.reshape(batch_size_logical * max_nodes, self.backbone.hidden_dim, self.backbone.window_size)
            temporal_out = self.backbone.temporal_net(x_temporal_in)
        else:
            x = x.permute(0, 2, 1, 3)
            x_atom_time = x.reshape(batch_size_logical * max_nodes, self.backbone.window_size, self.backbone.hidden_dim)
            temporal_out = self.backbone.temporal_net(x_atom_time)

        temporal_out = temporal_out.view(batch_size_logical, max_nodes, self.backbone.hidden_dim)
        mask_reshaped = mask.view(batch_size_logical, self.backbone.window_size, max_nodes)
        mask_final = mask_reshaped[:, 0, :]

        temporal_out = self.backbone.final_dropout(temporal_out)
        graph_embedding = self.backbone.pooling(temporal_out, mask=mask_final)

        features = self.backbone.projector(graph_embedding)
        logits = self.model.head(features)
        return logits

def aggregate_edge_importance_linear(edge_imp, edge_index, num_nodes):
    """Aggregazione lineare (con segno) dell'importanza dagli archi ai nodi."""
    row, col = edge_index
    node_imp_sum = scatter_add(edge_imp, row, dim=0, dim_size=num_nodes) + \
                   scatter_add(edge_imp, col, dim=0, dim_size=num_nodes)
    return node_imp_sum / 2.0

def integrated_gradients_rbf_directional(wrapper, data_batch, target_class_index, target_rbf=None, baseline_rbf=None, baseline_type='zero_edges', steps=20, ig_batch_size=5, device='cuda'):
    """
    Esegue IG interpolando l'intensità delle interazioni (RBF), calcolando variazioni direzionali rispetto a una media.
    I passi alpha vengono processati a blocchi (batches) per saturare al meglio la GPU e accelerare l'inferenza.
    """
    wrapper.eval()

    if target_rbf is None:
        real_dist = data_batch.edge_attr.clone().detach()
        if real_dist.dim() > 1: real_dist = real_dist.squeeze()

        with torch.no_grad():
            rbf_val = wrapper.spatial.rbf(real_dist)
            env_val = wrapper.spatial.envelope(real_dist, wrapper.spatial.cutoff).unsqueeze(-1)
            target_rbf_feat = (rbf_val * env_val).detach()
    else:
        target_rbf_feat = target_rbf.to(device)

    # Gestione esplicita del tipo di baseline
    if baseline_rbf is None or baseline_type == 'zero_edges':
        baseline_rbf_feat = torch.zeros_like(target_rbf_feat)
    else:
        baseline_rbf_feat = baseline_rbf.to(device)
        num_graphs = target_rbf_feat.size(0) // baseline_rbf_feat.size(0)
        if num_graphs > 1:
            baseline_rbf_feat = baseline_rbf_feat.repeat(num_graphs, 1)

    grads_rbf_accum = torch.zeros_like(target_rbf_feat)

    alphas = torch.linspace(0, 1, steps + 1, device=device)
    alpha_steps = alphas[1:]

    E_total = target_rbf_feat.shape[0]
    V_total = data_batch.num_nodes
    N_total = data_batch.num_graphs

    from types import SimpleNamespace

    # Processamento IG a Blocchi (Batches)
    for i in range(0, len(alpha_steps), ig_batch_size):
        batch_alphas = alpha_steps[i:i + ig_batch_size]
        B = len(batch_alphas)

        # 1. Costruiamo un "Mega-Batch" manualmente per evitare bug di `to_data_list()` su grafi modificati con i ghost_edges
        big_batch = SimpleNamespace()

        if data_batch.x.dim() == 1:
            big_batch.x = data_batch.x.repeat(B)
        else:
            big_batch.x = data_batch.x.repeat(B, 1)

        if hasattr(data_batch, 'x_dihe') and data_batch.x_dihe is not None:
            big_batch.x_dihe = data_batch.x_dihe.repeat(B, 1)

        edge_indices = []
        for b_idx in range(B):
            edge_indices.append(data_batch.edge_index + b_idx * V_total)
        big_batch.edge_index = torch.cat(edge_indices, dim=1)

        batches = []
        for b_idx in range(B):
            batches.append(data_batch.batch + b_idx * N_total)
        big_batch.batch = torch.cat(batches, dim=0)

        # 2. Re-espandiamo le RBF (Baseline e Target) in profondità (B volte)
        big_target = target_rbf_feat.repeat(B, 1)
        big_baseline = baseline_rbf_feat.repeat(B, 1)

        # 3. Adattiamo l'alpha in modo che diventi un vettore parallelo per ogni Edge [B*E_total, 1]
        alpha_expanded = batch_alphas.repeat_interleave(E_total).unsqueeze(-1)

        # 4. Interpolazione Lineare Multi-Step e Tracciamento dei Gradienti
        step_rbf = big_baseline + alpha_expanded * (big_target - big_baseline)
        step_rbf.requires_grad_(True)

        logits = wrapper(step_rbf, big_batch)

        # 5. Isoliamo il Logit Score richiesto per ognuno dei B passi calcolati
        logical_batch_size = logits.shape[0] // B
        logits_reshaped = logits.view(B, logical_batch_size, -1)

        target_scores = logits_reshaped[:, :, target_class_index]
        mean_scores = logits_reshaped.mean(dim=2)
        score = (target_scores - mean_scores).sum()

        wrapper.zero_grad()
        score.backward()

        # 6. Ricostruiamo la dimensionalità originale (E_total, F) e sommiamo allo stack globale
        grad_reshaped = step_rbf.grad.view(B, E_total, -1)
        grads_rbf_accum += grad_reshaped.sum(dim=0)

    avg_rbf_grads = grads_rbf_accum / steps
    delta_rbf = (target_rbf_feat - baseline_rbf_feat)

    edge_attr_raw = delta_rbf * avg_rbf_grads
    edge_importances = edge_attr_raw.sum(dim=1).detach()

    node_importances = aggregate_edge_importance_linear(
        edge_importances, data_batch.edge_index, data_batch.num_nodes
    )
    return node_importances, edge_importances

def save_window_as_gml_directional(sample_idx, num_nodes_per_frame, window_size,
                       edge_index_full, edge_imp_full, node_imp_full,
                       labels_full, out_path, prediction_info=None):
    """Salva il grafo di attribuzione Direzionale come file GML."""
    G = nx.Graph()
    if prediction_info:
        for k, v in prediction_info.items():
            G.graph[k] = v

    try:
        node_imp_matrix = node_imp_full.view(window_size, num_nodes_per_frame).cpu().numpy()

        # Metriche per-frame
        frame_saliency = np.sum(np.abs(node_imp_matrix), axis=1)
        frame_directional = np.sum(node_imp_matrix, axis=1)

        for f_idx in range(window_size):
            G.graph[f'frame_{f_idx}_saliency'] = float(frame_saliency[f_idx])
            G.graph[f'frame_{f_idx}_directional'] = float(frame_directional[f_idx])

        # Aggregazione temporale
        node_imp_agg = np.sum(node_imp_matrix, axis=0)
        node_types = labels_full[:num_nodes_per_frame].cpu().numpy()

        for i in range(num_nodes_per_frame):
            G.add_node(i, importance=float(node_imp_agg[i]), residue_type=int(node_types[i]))

    except Exception as e:
        print(f"Error Nodes {sample_idx}: {e}")
        return

    edge_index_cpu = edge_index_full.cpu().numpy()
    edge_imp_cpu = edge_imp_full.cpu().numpy()

    src, dst = edge_index_cpu[0], edge_index_cpu[1]
    edge_dict = {}

    for k in range(len(src)):
        u, v = src[k], dst[k]
        imp = float(edge_imp_cpu[k])

        u_canon, v_canon = u % num_nodes_per_frame, v % num_nodes_per_frame
        if u_canon == v_canon: continue
        if u_canon > v_canon: u_canon, v_canon = v_canon, u_canon

        key = (int(u_canon), int(v_canon))
        edge_dict[key] = edge_dict.get(key, 0.0) + imp

    for (u, v), total_imp in edge_dict.items():
        if abs(total_imp) > 1e-5: # Filtro per escludere rumore bianco ed Edge Ghost silenti
            G.add_edge(u, v, importance=total_imp)

    nx.write_gml(G, out_path)

def extract_node_importance_vector(G, num_residues=None):
    """Helper per estrarre le importance dai nodi di un GML letto."""
    node_ids = [int(n) for n in G.nodes()]
    if not node_ids: return np.array([])

    max_id = max(node_ids)
    actual_num_residues = max_id + 1
    if num_residues is None: num_residues = actual_num_residues

    vec = np.zeros(num_residues, dtype=np.float32)
    for node_id, attrs in G.nodes(data=True):
        idx = int(node_id)
        if idx < num_residues:
            vec[idx] = attrs.get('importance', 0.0)
    return vec
