import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import os
import sys
import copy
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

from .utils import setup_logger, seed_everything, AverageMeter
from .dataset import MDFlexibleWindowDataset, collate_windows
from .architecture import HybridStSchnet

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

class LinearProbeClassifier(nn.Module):
    def __init__(self, backbone, num_classes, embedding_dim):
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.head = nn.Sequential(
            nn.BatchNorm1d(embedding_dim, affine=True),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, batched_data):
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(batched_data)
        logits = self.head(features)
        return logits

class LinearTrainer:
    def __init__(self, model, optimizer, criterion, device, logger, scheduler=None, scaler=None, args=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.scheduler = scheduler
        self.scaler = scaler or GradScaler()
        self.args = args
        self.best_acc = 0.0
        self.patience_counter = 0

        self.history = {'train_loss': [], 'val_acc': []}

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        loss_meter = AverageMeter('Loss', ':.4f')
        acc_meter = AverageMeter('Acc', ':.2f')

        pbar = tqdm(train_loader, desc=f"Ep {epoch} [Linear]", leave=False)
        # Ignoro i paths '_' qui
        for batched_data, labels, _, _, _ in pbar:
            batched_data = batched_data.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            with autocast(device_type=self.device.type):
                logits = self.model(batched_data)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item() * 100

            loss_meter.update(loss.item(), labels.size(0))
            acc_meter.update(acc, labels.size(0))
            pbar.set_postfix(loss=loss_meter.avg, acc=acc_meter.avg)

        return loss_meter.avg

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for batched_data, labels, _, _, _ in loader:
                batched_data = batched_data.to(self.device)
                labels = labels.to(self.device)
                with autocast(device_type=self.device.type):
                    logits = self.model(batched_data)

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        acc = 100. * correct / total if total > 0 else 0.0
        return acc, y_true, y_pred

    def fit(self, train_loader, val_loader, test_loader=None):
        best_model_path = os.path.join(self.args.out_dir, "best_linear_model.pt")

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)

            if self.scheduler:
                self.scheduler.step()

            val_acc, y_true, y_pred = self.evaluate(val_loader)
            self.history['val_acc'].append(val_acc)

            saved_str = ""
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), best_model_path)
                saved_str = " [Saved Best]"
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.logger.info(f"Ep {epoch} Linear: Loss {train_loss:.4f} | Val Acc: {val_acc:.2f}%{saved_str}")

            if not getattr(self.args, 'use_scheduler', False) and self.patience_counter >= getattr(self.args, 'patience', 10):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Final Report
        if len(y_true) > 0:
            self.logger.info("\n" + classification_report(y_true, y_pred, digits=3))

        if test_loader:
            if os.path.exists(best_model_path):
                self.model.load_state_dict(torch.load(best_model_path))

            test_acc, _, _ = self.evaluate(test_loader)
            self.logger.info(f"[Linear Probe] Final Test Acc (Hold-out): {test_acc:.2f}%")

        return self.history

def resolve_shuffling_params(args, current_split_role):
    if args.shuffling == 'none':
        return False, False

    target_set = args.shuffling_set
    should_shuffle = False

    if target_set == 'all':
        should_shuffle = True
    elif target_set == 'training' and current_split_role == 'train':
        should_shuffle = True
    elif target_set == 'validation' and current_split_role == 'val':
        should_shuffle = True

    if not should_shuffle:
        return False, False

    g_s = (args.shuffling == 'global')
    w_s = (args.shuffling == 'windows')
    return g_s, w_s

def run_linear_probe(args, external_logger=None, shared_cache=None):
    if external_logger:
        logger = external_logger
        logger.info("--- Starting Linear Probing Phase (Internal Call) ---")
    else:
        logger = setup_logger(args.out_dir)
        logger.info("--- Starting Linear Probing Phase (Standalone) ---")

    seed = getattr(args, 'seed', 42)
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not hasattr(args, 'shuffling'): args.shuffling = 'none'
    if not hasattr(args, 'shuffling_set'): args.shuffling_set = 'all'

    safe_batch_size = getattr(args, 'micro_batch_size', args.batch_size)
    if hasattr(args, 'micro_batch_size') and args.micro_batch_size < args.batch_size:
        safe_batch_size = args.micro_batch_size
    preload_ram = getattr(args, 'preload_ram', False)
    no_loro = getattr(args, 'noLORO', False)
    val_groups = getattr(args, 'val_groups', [])

    test_loader = None
    if no_loro:
        if len(val_groups) > 0:
            logger.info(f"[Linear Probe] HYBRID MODE: Hold-out Test on Groups {val_groups}.")

            g_s_train, w_s_train = resolve_shuffling_params(args, 'train')
            pool_ds = MDFlexibleWindowDataset(args.data_class_dirs, split='train', val_groups=val_groups,
                                              window_size=args.window, window_offset=args.window_offset,
                                              stride=args.stride, skip=args.skip, preload_ram=preload_ram,
                                              global_shuffle=g_s_train, window_shuffle=w_s_train,
                                              shared_cache=shared_cache, logger=logger, ignore_validation_logic=False)

            g_s_val, w_s_val = resolve_shuffling_params(args, 'val')
            test_ds = MDFlexibleWindowDataset(args.data_class_dirs, split='val', val_groups=val_groups,
                                              window_size=args.window, window_offset=args.window_offset,
                                              stride=args.stride, skip=args.skip, preload_ram=preload_ram,
                                              global_shuffle=g_s_val, window_shuffle=w_s_val,
                                              shared_cache=shared_cache, logger=logger, ignore_validation_logic=False)

            train_size = int(0.8 * len(pool_ds))
            val_size = len(pool_ds) - train_size
            generator = torch.Generator().manual_seed(seed)
            train_ds, val_ds = random_split(pool_ds, [train_size, val_size], generator=generator)
            workers = 0 if preload_ram else 4
            test_loader = DataLoader(test_ds, batch_size=safe_batch_size, shuffle=False, collate_fn=collate_windows, num_workers=workers)
        else:
            logger.info(f"[Linear Probe] noLORO active: Using Random Split 80/20 (Seed {seed})")
            g_s, w_s = resolve_shuffling_params(args, 'train')
            full_ds = MDFlexibleWindowDataset(args.data_class_dirs, split='train', val_groups=[],
                                              window_size=args.window, window_offset=args.window_offset,
                                              stride=args.stride, skip=args.skip, preload_ram=preload_ram,
                                              global_shuffle=g_s, window_shuffle=w_s,
                                              shared_cache=shared_cache, logger=logger, ignore_validation_logic=True)
            train_size = int(0.8 * len(full_ds))
            val_size = len(full_ds) - train_size
            generator = torch.Generator().manual_seed(seed)
            train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=generator)
    else:
        g_s_train, w_s_train = resolve_shuffling_params(args, 'train')
        train_ds = MDFlexibleWindowDataset(args.data_class_dirs, split='train', val_groups=args.val_groups,
                                           window_size=args.window, window_offset=args.window_offset,
                                           stride=args.stride, skip=args.skip, preload_ram=preload_ram,
                                           global_shuffle=g_s_train, window_shuffle=w_s_train,
                                           shared_cache=shared_cache, logger=logger)

        g_s_val, w_s_val = resolve_shuffling_params(args, 'val')
        val_ds = MDFlexibleWindowDataset(args.data_class_dirs, split='val', val_groups=args.val_groups,
                                         window_size=args.window, window_offset=args.window_offset,
                                         stride=args.stride, skip=args.skip, preload_ram=preload_ram,
                                         global_shuffle=g_s_val, window_shuffle=w_s_val,
                                         shared_cache=shared_cache, logger=logger)

    num_classes = len(args.data_class_dirs)
    sampler = None
    if getattr(args, 'balance_classes', False):
        if isinstance(train_ds, Subset):
            full_weights = train_ds.dataset.get_weights()
            weights = full_weights[train_ds.indices]
        else:
            weights = train_ds.get_weights()
        if len(weights) > 0:
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    workers = 0 if preload_ram else 4
    train_loader = DataLoader(train_ds, batch_size=safe_batch_size, shuffle=(sampler is None),
                              sampler=sampler, collate_fn=collate_windows, num_workers=workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=safe_batch_size, shuffle=False,
                            collate_fn=collate_windows, num_workers=workers)

    # --- MODEL SETUP ---
    backbone = HybridStSchnet(
        hidden_dim=args.hidden_dim, embedding_dim=args.embedding_dim,
        window_size=args.window, n_layers=args.n_layers,
        cutoff=args.cutoff, dihedral_dim=args.dihedral_dim,
        use_checkpointing=args.use_checkpointing, pooling_type=args.pooling_type,
        dropout=args.dropout, attn_temperature=args.attn_temperature,
        temporal_setup=getattr(args, 'temporal_setup', 'cnn'),
        pooling_activation=getattr(args, 'pooling_activation', 'softmax')
    )

    try:
        state_dict = torch.load(args.pretrained_path, map_location='cpu', weights_only=True)
        backbone.load_state_dict(state_dict, strict=True)
    except Exception as e:
        logger.error(f"[Linear Probe] Errore caricamento pesi: {e}")
        return None

    model = LinearProbeClassifier(backbone, num_classes, args.embedding_dim).to(device)
    optimizer = optim.Adam(model.head.parameters(), lr=args.lr)

    scheduler = None
    if getattr(args, 'use_scheduler', False):
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=getattr(args, 'min_lr', 1e-6))

    criterion = nn.CrossEntropyLoss()

    trainer = LinearTrainer(model, optimizer, criterion, device, logger, scheduler=scheduler, args=args)
    linear_history = trainer.fit(train_loader, val_loader, test_loader)

    return linear_history
