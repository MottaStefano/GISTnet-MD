import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupContrastiveLoss(nn.Module):
    """
    Contrastive Loss flessibile con tre modalità operative:
    
    1. 'group_aware' (Default):
       - Positivi: Stessa classe MA gruppo diverso (es. repliche diverse).
       - Negativi: Classe diversa (opzionalmente ignora nomi condivisi).
       - Obiettivo: Imparare invarianze fisiche globali, ignorando correlazioni temporali locali.
       
    2. 'base':
       - Positivi: Stessa classe (inclusi frame della stessa replica).
       - Negativi: Classe diversa.
       - Obiettivo: Standard Contrastive Learning. I frame vicini si attraggono.
       
    3. 'hard_negative':
       - Positivi: Come 'base'.
       - Negativi: Mining dei "Hard Negatives". Per ogni ancora, considera solo il negativo 
         più vicino (più difficile) invece di fare la media su tutti i negativi.
       - Obiettivo: Forzare il modello a risolvere i casi limite.
    """
    def __init__(self, margin=1.0, map_negative_name_to_exclude=False, mode='group_aware'):
        super().__init__()
        self.margin = margin
        self.map_negative_name_to_exclude = map_negative_name_to_exclude
        self.mode = mode
        
        valid_modes = ['group_aware', 'base', 'hard_negative']
        if self.mode not in valid_modes:
            raise ValueError(f"Mode {mode} non valido. Scegliere tra: {valid_modes}")

    def forward(self, embeddings, labels, groups, shared_names=None):
        """
        Args:
            embeddings: (Batch, Dim)
            labels: (Batch,) Class labels
            groups: (Batch,) Group IDs (trajectory/replica IDs)
            shared_names: (Batch,) Optional shared name IDs
        """
        # Calcola matrice delle distanze Euclidee a coppie
        dists = torch.cdist(embeddings, embeddings, p=2)
        
        labels = labels.view(-1, 1)
        groups = groups.view(-1, 1)
        batch_size = embeddings.size(0)

        # --- MASCHERE BASE ---
        mask_same_class = torch.eq(labels, labels.T)
        mask_diff_class = ~mask_same_class
        
        # Maschera Identità (esclude la diagonale, cioè se stesso)
        mask_identity = torch.eye(batch_size, device=embeddings.device).bool()

        # --- DEFINIZIONE POSITIVI ---
        if self.mode == 'group_aware':
            # Originale: Positivi solo se gruppi diversi (es. rep_1 vs rep_2)
            mask_diff_group = ~torch.eq(groups, groups.T)
            mask_valid_pos = mask_same_class & mask_diff_group
        else:
            # Base/HardNeg: Positivi anche nello stesso gruppo (es. rep_1 frame 10 vs rep_1 frame 20)
            # Escludiamo solo l'identità (se stesso)
            mask_valid_pos = mask_same_class & (~mask_identity)
        
        # --- DEFINIZIONE NEGATIVI ---
        if self.mode == 'group_aware' and self.map_negative_name_to_exclude and shared_names is not None:
            # Logica speciale per escludere negativi con nomi condivisi (solo in group_aware solitamente)
            shared_names = shared_names.view(-1, 1)
            mask_same_shared_name = torch.eq(shared_names, shared_names.T)
            mask_valid_neg = mask_diff_class & (~mask_same_shared_name)
        else:
            # Base/HardNeg: Tutti i campioni di classe diversa sono negativi validi
            mask_valid_neg = mask_diff_class

        # --- CALCOLO LOSS POSITIVA ---
        # Somma delle distanze quadrate dei positivi / numero di positivi
        num_pos = mask_valid_pos.sum() + 1e-8
        pos_loss = (dists.pow(2) * mask_valid_pos.float()).sum() / num_pos
        
        # --- CALCOLO LOSS NEGATIVA ---
        if self.mode == 'hard_negative':
            # Hard Negative Mining: Per ogni ancora (riga), trova il negativo con distanza minima
            
            # 1. Maschera i non-negativi settandoli a infinito (così min() li ignora)
            neg_dists_masked = dists.clone()
            # Dove NON è un negativo valido, metti infinito
            neg_dists_masked[~mask_valid_neg] = float('inf')
            
            # 2. Trova il negativo più vicino (hardest) per ogni riga
            hardest_neg_dists, _ = neg_dists_masked.min(dim=1)
            
            # 3. Filtra infiniti (casi in cui una riga non ha negativi validi nel batch, raro ma possibile)
            valid_anchors_mask = hardest_neg_dists != float('inf')
            
            if valid_anchors_mask.sum() > 0:
                # Calcola hinge loss solo sui hardest negatives
                # Loss = ReLU(margin - min_dist)^2
                hard_neg_loss = F.relu(self.margin - hardest_neg_dists[valid_anchors_mask]).pow(2).mean()
            else:
                hard_neg_loss = torch.tensor(0.0, device=embeddings.device)
                
            neg_loss = hard_neg_loss

        else:
            # Group Aware o Base (Soft Margin su tutti i negativi)
            neg_dist_term = self.margin - dists
            # Considera solo le coppie che violano il margine
            violating_pairs = F.relu(neg_dist_term).pow(2) * mask_valid_neg.float()
            
            num_neg = mask_valid_neg.sum() + 1e-8
            neg_loss = violating_pairs.sum() / num_neg
        
        return pos_loss + neg_loss
