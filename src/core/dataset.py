import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
import os
import re
import logging
import random
import hashlib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class MDFlexibleWindowDataset(Dataset):
    def __init__(self, data_class_dirs, split='train',
                 val_groups=None,
                 window_size=10,
                 window_offset=None,
                 stride=1,
                 skip=0,

                 preload_ram=False,
                 global_shuffle=False,
                 window_shuffle=False,
                 shared_cache=None,
                 logger=None,
                 ignore_validation_logic=False):

        self.window_size = window_size
        self.window_offset = window_offset if window_offset is not None else window_size
        self.stride = stride
        self.skip = skip
        self.global_shuffle = global_shuffle
        self.window_shuffle = window_shuffle
        self.preload_ram = preload_ram
        self.ignore_validation_logic = ignore_validation_logic

        self.filename_re = re.compile(r"(.+)_frame_(\d+)\.pt")
        self.samples = []
        self.class_counts = {}
        self.logger = logger

        self.ram_cache = shared_cache if shared_cache is not None else {}


        if val_groups is None: val_groups = []
        val_groups_set = set(val_groups)

        if logger:
            shuffle_mode = "GLOBAL" if self.global_shuffle else "Sequential"
            if self.window_shuffle: shuffle_mode += "+WIN_SHUFFLE"
            cache_info = f"Shared({len(self.ram_cache)})" if shared_cache else "New"
            split_info = "FULL_LOAD" if self.ignore_validation_logic else split.upper()
            logger.info(f"--- DS Init {split_info}: Off={self.window_offset}, Sub={self.stride}, Mode={shuffle_mode}, RAM={self.preload_ram} [{cache_info}] ---")

        # 1. Discovery
        class_to_groups = {}
        shared_name_to_id = {}
        next_shared_id = 0


        if isinstance(data_class_dirs, str): data_class_dirs = [data_class_dirs]

        for class_label, root_dir in enumerate(data_class_dirs):
            if class_label not in class_to_groups: class_to_groups[class_label] = set()
            if not os.path.isdir(root_dir): continue

            for current_root, dirs, files in os.walk(root_dir):
                pt_files = [f for f in files if f.endswith('.pt')]
                if pt_files:
                    abs_path = os.path.abspath(current_root)
                    class_to_groups[class_label].add(abs_path)

                    leaf_name = os.path.basename(abs_path)
                    if leaf_name not in shared_name_to_id:
                        shared_name_to_id[leaf_name] = next_shared_id
                        next_shared_id += 1





        # 3. Role Assignment
        path_is_val = {}
        path_to_global_group_id = {}
        path_to_shared_name_id = {}
        global_group_counter = 0

        for class_label in sorted(class_to_groups.keys()):
            sorted_groups = sorted(list(class_to_groups[class_label]))
            for local_idx, group_path in enumerate(sorted_groups):
                user_id = local_idx + 1
                is_val = (user_id in val_groups_set)
                path_is_val[group_path] = is_val
                path_to_global_group_id[group_path] = global_group_counter
                leaf_name = os.path.basename(group_path)
                path_to_shared_name_id[group_path] = shared_name_to_id[leaf_name]
                global_group_counter += 1

        sequences_buffer = {}
        all_unique_paths = set()

        for class_label, root_dir in enumerate(data_class_dirs):
            if class_label not in self.class_counts: self.class_counts[class_label] = 0
            if not os.path.isdir(root_dir): continue

            for current_root, dirs, files in os.walk(root_dir):
                abs_path = os.path.abspath(current_root)
                if abs_path not in path_is_val: continue

                if not self.ignore_validation_logic:
                    is_val_group = path_is_val[abs_path]
                    if split == 'train' and is_val_group: continue
                    if split == 'val' and not is_val_group: continue

                global_grp_id = path_to_global_group_id[abs_path]
                shared_nm_id = path_to_shared_name_id[abs_path]

                pt_files = [f for f in files if f.endswith(".pt")]
                for fname in pt_files:
                    match = self.filename_re.search(fname)
                    if not match: continue
                    prefix_name = match.group(1)
                    frame_id = int(match.group(2))

                    if frame_id < self.skip: continue

                    full_path = os.path.join(current_root, fname)
                    seq_key = (class_label, global_grp_id, shared_nm_id, prefix_name)
                    if seq_key not in sequences_buffer: sequences_buffer[seq_key] = []
                    sequences_buffer[seq_key].append((frame_id, full_path))

                    if self.preload_ram: all_unique_paths.add(full_path)

        # 4. Window Creation
        for (cls, grp, sh_id, prefix), frames_list in sequences_buffer.items():
            if self.global_shuffle:
                seed_val = int(hashlib.md5(prefix.encode('utf-8')).hexdigest(), 16) % (2**32)
                rng = random.Random(seed_val)
                rng.shuffle(frames_list)
            else:
                frames_list.sort(key=lambda x: x[0])

            if self.stride > 1:
                frames_list = frames_list[::self.stride]

            paths = [x[1] for x in frames_list]
            num_frames = len(paths)

            for i in range(0, num_frames - self.window_size + 1, self.window_offset):
                window_paths = paths[i : i + self.window_size]
                self.samples.append((window_paths, cls, grp, sh_id))
                self.class_counts[cls] += 1

        # 5. RAM Preload
        if self.preload_ram and len(all_unique_paths) > 0:
            paths_to_load = [p for p in all_unique_paths if p not in self.ram_cache]
            if len(paths_to_load) > 0:
                self._preload_data(paths_to_load)
            elif logger:
                logger.info("RAM Cache hit.")

        if logger:
            logger.info(f"  [{'FULL' if self.ignore_validation_logic else split.upper()}] Windows: {len(self.samples)} | Files Required: {len(all_unique_paths)}")

    def _preload_data(self, paths_list):
        if self.logger: self.logger.info(f"Preloading {len(paths_list)} files...")
        def load_single(p):
            return p, torch.load(p, weights_only=False)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(tqdm(executor.map(load_single, paths_list), total=len(paths_list), desc="RAM Load"))
        self.ram_cache.update(dict(results))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, label, group_id, shared_name_id = self.samples[idx]

        if self.window_shuffle:
            paths = list(paths)
            seed_key = os.path.basename(paths[0])
            seed_val = int(hashlib.md5(seed_key.encode('utf-8')).hexdigest(), 16) % (2**32)
            rng = random.Random(seed_val)
            rng.shuffle(paths)



        graph_list = []
        for p in paths:
            if self.preload_ram:
                data = self.ram_cache[p].clone()
            else:
                data = torch.load(p, weights_only=False)

            if hasattr(data, 'edge_index') and data.edge_index is not None:
                data.edge_index = data.edge_index.to(torch.int32)

            graph_list.append(data)

        # Modifica: Ritorno anche la lista dei path per gli embeddings
        return graph_list, label, group_id, shared_name_id, paths

    def get_weights(self):
        weights = []
        class_weights = {}
        for cls, count in self.class_counts.items():
            class_weights[cls] = 1.0 / count if count > 0 else 0.0
        for _, label, _, _ in self.samples:
            weights.append(class_weights[label])
        return torch.tensor(weights, dtype=torch.float)

def collate_windows(batch):
    # Modifica: accumulo i path nel batch
    labels, groups, shared_names, flat_graphs, batch_paths = [], [], [], [], []
    for graph_list, label, group_id, shared_name_id, paths in batch:
        labels.append(label)
        groups.append(group_id)
        shared_names.append(shared_name_id)
        flat_graphs.extend(graph_list)
        batch_paths.append(paths)
    batched_data = Batch.from_data_list(flat_graphs)
    return batched_data, torch.tensor(labels), torch.tensor(groups), torch.tensor(shared_names), batch_paths
