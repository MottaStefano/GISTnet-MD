import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import to_dense_batch

# ===================================================================================
# 1. COMPONENTI DI SUPPORTO (INVARIATI)
# ===================================================================================

class Envelope(nn.Module):
    def __init__(self, exponent=5):
        super(Envelope, self).__init__()
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, dist, cutoff):
        p = dist / cutoff
        env_val = 1.0 + self.a * p**self.p + self.b * p**(self.p + 1) + self.c * p**(self.p + 2)
        return torch.where(dist < cutoff, env_val, torch.zeros_like(dist))

class RBFExpansion(nn.Module):
    def __init__(self, min_dist=0.0, max_dist=20.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(min_dist, max_dist, num_gaussians)
        self.register_buffer('offset', offset)
        self.coeff = -0.5 / ((max_dist - min_dist) / num_gaussians) ** 2

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class SparseInteractionBlock(nn.Module):
    def __init__(self, hidden_dim, num_gaussians):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.SiLU()
        )
        self.conv_out = nn.Linear(hidden_dim, hidden_dim)
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

    def forward(self, h, edge_index, rbf_feat):
        row, col = edge_index
        edge_weight = self.mlp(rbf_feat)
        h_trans = self.conv_out(h)
        messages = h_trans[col] * edge_weight
        messages = messages.to(dtype=h.dtype) 
        aggr_messages = torch.zeros_like(h)
        aggr_messages.index_add_(0, row, messages)
        h_new = h + self.update_net(aggr_messages)
        return h_new

# ===================================================================================
# 2. MODULI TEMPORALI
# ===================================================================================

class TemporalCNN(nn.Module):
    """
    Incapsula la logica esatta della CNN originale 'Old'.
    Input: (Batch, Channels, Time)
    Output: (Batch, Channels) -> Squeeze interno applicato
    """
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim), nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=2), 
            nn.BatchNorm1d(hidden_dim), nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class TemporalAttention(nn.Module):
    """
    Nuovo modulo di Attention.
    Input: (Batch, Time, Channels) -> Nota la differenza di dimensione rispetto a CNN
    Output: (Batch, Channels)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        scores = self.query_layer(x) # (B, T, 1)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(x * weights, dim=1)
        return context

# ===================================================================================
# 3. POOLING LAYERS
# ===================================================================================

class MaskedMeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask is None: return x.mean(dim=1)
        mask_expanded = mask.unsqueeze(-1).float()
        sum_features = (x * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_features / count

class ConfigurableGlobalPooling(nn.Module):
    """
    Gestisce sia 'softmax' (Old behavior) che 'sigmoid' (New behavior).
    """
    def __init__(self, input_dim, temperature=1.0, activation='softmax'):
        super().__init__()
        self.temperature = temperature
        self.activation = activation
        
        self.attn_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(), 
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x, mask=None):
        attn_scores = self.attn_net(x).squeeze(-1)
        
        if mask is not None:
            min_val = -1e4
            attn_scores = attn_scores.masked_fill(~mask, min_val)
        
        if self.activation == 'softmax':
            attn_weights = torch.softmax(attn_scores / self.temperature, dim=1).unsqueeze(-1)
        elif self.activation == 'sigmoid':
            attn_weights = torch.sigmoid(attn_scores).unsqueeze(-1)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        
        return (x * attn_weights).sum(dim=1)

# ===================================================================================
# 4. ARCHITETTURA IBRIDA
# ===================================================================================

class HybridSpatialEncoder(nn.Module):
    def __init__(self, hidden_dim=128, n_layers=3, cutoff=10.0, dihedral_dim=0, use_checkpointing=False):
        super().__init__()
        self.cutoff = cutoff
        self.use_checkpointing = use_checkpointing
        self.dihedral_dim = dihedral_dim
        
        self.embedding = nn.Embedding(21, hidden_dim)
        
        if dihedral_dim > 0:
            self.dihedral_proj = nn.Linear(dihedral_dim, hidden_dim)
            self.dihedral_norm = nn.LayerNorm(hidden_dim)
        
        self.rbf = RBFExpansion(max_dist=cutoff, num_gaussians=64)
        self.envelope = Envelope()
        self.layers = nn.ModuleList([
            SparseInteractionBlock(hidden_dim, num_gaussians=64) for _ in range(n_layers)
        ])
        self.atom_norm = nn.LayerNorm(hidden_dim)

    def forward(self, data):
        h = self.embedding(data.x)
        if self.dihedral_dim > 0 and hasattr(data, 'x_dihe'):
            dihe_emb = self.dihedral_norm(self.dihedral_proj(data.x_dihe))
            h = h + dihe_emb

        edge_index = data.edge_index
        dist = data.edge_attr
        if dist.dim() > 1: dist = dist.squeeze()
        
        rbf_feat = self.rbf(dist) * self.envelope(dist, self.cutoff).unsqueeze(-1)

        for layer in self.layers:
            if self.use_checkpointing and self.training:
                h = checkpoint(layer, h, edge_index, rbf_feat, use_reentrant=False)
            else:
                h = layer(h, edge_index, rbf_feat)
        
        h = self.atom_norm(h)
        return h

class HybridStSchnet(nn.Module):
    def __init__(self, hidden_dim=128, embedding_dim=64, window_size=10, 
                 n_layers=3, cutoff=10.0, dihedral_dim=0, use_checkpointing=False,
                 pooling_type='attention', dropout=0.0, attn_temperature=1.0,
                 temporal_setup='cnn',       
                 pooling_activation='softmax',
                 num_classes=None # UPDATED: Optional argument for classification head
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.temporal_setup = temporal_setup
        self.num_classes = num_classes # Store
        
        self.spatial_encoder = HybridSpatialEncoder(
            hidden_dim=hidden_dim, n_layers=n_layers, cutoff=cutoff, 
            dihedral_dim=dihedral_dim, use_checkpointing=use_checkpointing
        )
        
        # --- Modulo Temporale ---
        if self.temporal_setup == 'cnn':
            self.temporal_net = TemporalCNN(hidden_dim, dropout=dropout)
        elif self.temporal_setup == 'attention':
            self.temporal_net = TemporalAttention(hidden_dim)
        else:
            raise ValueError(f"Unknown temporal_setup: {temporal_setup}")
        
        # --- Pooling ---
        if pooling_type == 'mean':
            self.pooling = MaskedMeanPooling()
        else:
            self.pooling = ConfigurableGlobalPooling(
                hidden_dim, 
                temperature=attn_temperature,
                activation=pooling_activation
            )
        
        self.final_dropout = nn.Dropout(p=dropout)

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # UPDATED: Add classifier head if num_classes is set
        if self.num_classes is not None:
            self.classifier_head = nn.Sequential(
                nn.BatchNorm1d(embedding_dim, affine=True),
                nn.Linear(embedding_dim, num_classes)
            )

    def forward(self, batched_data):
        # 1. Spatial Encoder
        node_features = self.spatial_encoder(batched_data)
        
        # 2. Densify
        x_dense, mask = to_dense_batch(node_features, batched_data.batch)
        total_graphs, max_nodes, _ = x_dense.size() 
        batch_size_logical = total_graphs // self.window_size
        
        # 3. Reshape Temporale
        x = x_dense.view(batch_size_logical, self.window_size, max_nodes, self.hidden_dim)
        
        if self.temporal_setup == 'cnn':
            x = x.permute(0, 2, 3, 1) 
            x_temporal_in = x.reshape(batch_size_logical * max_nodes, self.hidden_dim, self.window_size)
            temporal_out = self.temporal_net(x_temporal_in)
        else:
            x = x.permute(0, 2, 1, 3) 
            x_atom_time = x.reshape(batch_size_logical * max_nodes, self.window_size, self.hidden_dim)
            temporal_out = self.temporal_net(x_atom_time)
        
        # Ripristino dimensioni batch
        temporal_out = temporal_out.view(batch_size_logical, max_nodes, self.hidden_dim)
        
        # Maschera
        mask_reshaped = mask.view(batch_size_logical, self.window_size, max_nodes)
        mask_final = mask_reshaped[:, 0, :] 
        
        # Dropout e Pooling
        temporal_out = self.final_dropout(temporal_out)
        graph_embedding = self.pooling(temporal_out, mask=mask_final)
        
        # Projector
        out = self.projector(graph_embedding)
        
        # UPDATED: Return Logits if classifier active, else Normalized Embeddings
        if self.num_classes is not None:
            return self.classifier_head(out) # Return Logits (unnormalized)
        else:
            return F.normalize(out, p=2, dim=1) # Return Embeddings (normalized)
