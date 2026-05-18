import torch
import os
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from .utils import AverageMeter
from .dataset import collate_windows

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

class HybridTrainer:
    def __init__(self, model, optimizer, criterion, device, logger,
                 scheduler=None, scaler=None, args=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.scheduler = scheduler
        self.scaler = scaler or GradScaler()
        self.args = args

        self.best_val_metric = -1.0
        self.best_loss_at_best_metric = float('inf')
        self.patience_counter = 0

        # Tracking storico per il plot di convergenza
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        loss_meter = AverageMeter('Loss', ':.4f')

        accumulated_batches = []
        accumulated_labels = []
        accumulated_groups = []
        accumulated_shared_names = []

        pbar = tqdm(train_loader, desc=f"Ep {epoch} Training")

        # Aggiunto `paths` (underscore per ignorarlo qui)
        for i, (batched_data, labels, groups, shared_names, _) in enumerate(pbar):
            accumulated_batches.append(batched_data)
            accumulated_labels.append(labels)
            accumulated_groups.append(groups)
            accumulated_shared_names.append(shared_names)

            current_accumulated_size = sum(b.num_graphs for b in accumulated_batches) // self.args.window

            if current_accumulated_size >= self.args.batch_size:
                self.optimizer.zero_grad()

                cached_outputs = []
                with torch.no_grad():
                    for mb in accumulated_batches:
                        mb = mb.to(self.device)
                        with autocast(device_type=self.device.type):
                            out = self.model(mb)
                        cached_outputs.append(out)

                full_outputs = torch.cat(cached_outputs, dim=0)
                full_outputs.requires_grad_(True)

                full_labels = torch.cat(accumulated_labels).to(self.device)
                full_groups = torch.cat(accumulated_groups).to(self.device)
                full_shared = torch.cat(accumulated_shared_names).to(self.device) if accumulated_shared_names[0] is not None else None

                with autocast(device_type=self.device.type):
                    if self.args.contrastive == 'none':
                        loss = self.criterion(full_outputs, full_labels)
                    else:
                        loss = self.criterion(full_outputs, full_labels, full_groups, full_shared)

                self.scaler.scale(loss).backward()
                loss_meter.update(loss.item())

                upstream_grads = full_outputs.grad
                start_idx = 0
                for j, mb in enumerate(accumulated_batches):
                    mb = mb.to(self.device)
                    mb_size = cached_outputs[j].shape[0]
                    end_idx = start_idx + mb_size
                    mb_grads = upstream_grads[start_idx:end_idx]

                    with autocast(device_type=self.device.type):
                        mb_out = self.model(mb)
                    mb_out.backward(mb_grads)
                    start_idx = end_idx

                self.scaler.step(self.optimizer)
                self.scaler.update()

                accumulated_batches = []
                accumulated_labels = []
                accumulated_groups = []
                accumulated_shared_names = []

                pbar.set_postfix(loss=loss_meter.avg)

        return loss_meter.avg

    def evaluate(self, loader, train_ds_for_knn=None):
        self.model.eval()
        all_outputs = []
        all_labels = []
        all_groups = []

        loss_meter = AverageMeter('ValLoss', ':.4f')

        with torch.no_grad():
            # Aggiunto `paths` (underscore per ignorarlo qui)
            for batched_data, labels, groups, shared_names, _ in loader:
                batched_data = batched_data.to(self.device)
                labels = labels.to(self.device)
                groups = groups.to(self.device)

                with autocast(device_type=self.device.type):
                    out = self.model(batched_data)

                    if self.args.contrastive == 'none':
                        val_loss = self.criterion(out, labels)
                    else:
                        temp_mode = self.criterion.mode
                        if self.criterion.mode == 'group_aware':
                            self.criterion.mode = 'base'
                        val_loss = self.criterion(out, labels, groups, None)
                        self.criterion.mode = temp_mode

                loss_meter.update(val_loss.item(), labels.size(0))
                all_outputs.append(out)
                all_labels.append(labels)
                all_groups.append(groups)

        if not all_outputs: return {'loss': 0.0, 'acc': 0.0}

        full_out = torch.cat(all_outputs, dim=0)
        full_lbls = torch.cat(all_labels, dim=0)
        out_np = full_out.float().cpu().numpy()
        lbls_np = full_lbls.cpu().numpy()

        metrics = {'loss': loss_meter.avg}

        if self.args.contrastive == 'none':
            preds = np.argmax(out_np, axis=1)
            acc = accuracy_score(lbls_np, preds) * 100
            metrics['acc'] = acc
        else:
            metrics['acc'] = 0.0
            if train_ds_for_knn is not None:
                tr_embs, tr_lbls = [], []
                from torch.utils.data import DataLoader
                temp_loader = DataLoader(train_ds_for_knn, batch_size=64, collate_fn=collate_windows, shuffle=True)
                count = 0
                self.model.eval()
                with torch.no_grad():
                    # Aggiunto `_`
                    for batched_data, labels, _, _, _ in temp_loader:
                        batched_data = batched_data.to(self.device)
                        with autocast(device_type=self.device.type):
                            embs = self.model(batched_data)
                        tr_embs.append(embs.float().cpu().numpy())
                        tr_lbls.append(labels.numpy())
                        count += len(labels)
                        if count >= 3000: break

                if len(tr_embs) > 0:
                    tr_embs = np.concatenate(tr_embs)
                    tr_lbls = np.concatenate(tr_lbls)
                    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
                    knn.fit(tr_embs, tr_lbls)
                    pred_knn = knn.predict(out_np)
                    metrics['acc'] = accuracy_score(lbls_np, pred_knn) * 100

        return metrics

    def fit(self, train_loader, val_loader, test_loader=None, train_ds=None):
        warmup_epochs = int(self.args.epochs * 0.1) if self.args.use_scheduler else 0
        if warmup_epochs < 1: warmup_epochs = 1

        scheduler_warmup = None
        scheduler_cosine = None
        if self.args.use_scheduler:
            decay_epochs = self.args.epochs - warmup_epochs
            scheduler_warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
            scheduler_cosine = CosineAnnealingLR(self.optimizer, T_max=decay_epochs, eta_min=self.args.scheduler_min_lr)

        best_model_path = os.path.join(self.args.out_dir, "best_model.pt")

        for epoch in range(1, self.args.epochs + 1):
            avg_train_loss = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(avg_train_loss)

            if self.args.use_scheduler:
                if epoch <= warmup_epochs: scheduler_warmup.step()
                else: scheduler_cosine.step()

            should_eval = (not self.args.use_scheduler) or (epoch in [1, 3, 5, 10]) or (epoch % 10 == 0) or (epoch == self.args.epochs)

            metric_str = "Skipped"
            if should_eval:
                val_metrics = self.evaluate(val_loader, train_ds_for_knn=train_ds if self.args.contrastive != 'none' else None)

                metric_key = 'acc'
                current_metric = val_metrics[metric_key]
                avg_val_loss = val_metrics['loss']

                # Registra nello storico
                self.history['val_loss'].append(avg_val_loss)
                self.history['val_acc'].append(current_metric)

                metric_str = f"Acc: {current_metric:.2f}% | Val Loss: {avg_val_loss:.4f}"

                is_best = False
                if current_metric > self.best_val_metric:
                    is_best = True
                elif abs(current_metric - self.best_val_metric) < 1e-6:
                    if avg_val_loss < self.best_loss_at_best_metric:
                        is_best = True

                if is_best:
                    self.best_val_metric = current_metric
                    self.best_loss_at_best_metric = avg_val_loss
                    torch.save(self.model.state_dict(), best_model_path)
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if not self.args.use_scheduler and self.patience_counter >= self.args.patience:
                    self.logger.info(f"Early Stopping. Best Metric: {self.best_val_metric:.2f}%")
                    # Riempi la history mancante per via dell'early stop per allineare gli array
                    break
            else:
                # Se saltiamo l'eval, riempiamo con NaN o l'ultimo valore valido
                last_vl = self.history['val_loss'][-1] if len(self.history['val_loss']) > 0 else np.nan
                last_va = self.history['val_acc'][-1] if len(self.history['val_acc']) > 0 else np.nan
                self.history['val_loss'].append(last_vl)
                self.history['val_acc'].append(last_va)

            self.logger.info(f"Ep {epoch}: Train Loss {avg_train_loss:.4f} | {metric_str}")

        self._final_report(best_model_path, train_loader, val_loader, test_loader, train_ds)
        return self.history

    def _final_report(self, best_model_path, train_loader, val_loader, test_loader, train_ds):
        self.logger.info("\n" + "="*50)
        self.logger.info("FINAL EVALUATION ON BEST MODEL")
        self.logger.info("="*50)

        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            self.logger.info(f"Loaded best weights from {best_model_path}")

        val_metrics = self.evaluate(val_loader, train_ds_for_knn=train_ds)
        self.logger.info(f"-> VAL Set Results: Accuracy: {val_metrics['acc']:.2f}% | Loss: {val_metrics['loss']:.4f}")

        train_metrics = self.evaluate(train_loader, train_ds_for_knn=train_ds)
        self.logger.info(f"-> TRAIN Set Results: Accuracy: {train_metrics['acc']:.2f}% | Loss: {train_metrics['loss']:.4f}")

        if test_loader:
            test_metrics = self.evaluate(test_loader, train_ds_for_knn=train_ds)
            self.logger.info(f"-> TEST Set Results: Accuracy: {test_metrics['acc']:.2f}% | Loss: {test_metrics['loss']:.4f}")

        self.logger.info(f"Overfitting Gap (Train - Val): {train_metrics['acc'] - val_metrics['acc']:.2f}%")
        self.logger.info("="*50 + "\n")
