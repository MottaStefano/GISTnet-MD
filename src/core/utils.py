import argparse
import ast
import torch
import logging
import os
import sys
import random
import numpy as np

def setup_logger(out_dir):
    """
    Configura il logger per scrivere su file e console.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
            
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(out_dir, "train.log")), 
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

def seed_everything(seed=42):
    """
    Imposta il seed per la riproducibilità totale.
    Gestisce random, numpy, torch, cuda e cudnn.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Determinismo CUDA (può rallentare leggermente, ma garantisce riproducibilità)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_mask_string(mask_str):
    if not mask_str: return []
    indices = set()
    parts = mask_str.split(',')
    for part in parts:
        part = part.strip()
        if not part: continue
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                indices.update(range(start, end + 1))
            except ValueError:
                continue
        else:
            try:
                indices.add(int(part))
            except ValueError:
                continue
    return sorted(list(indices))

class MaskAndClampTransform:
    def __init__(self, mask_ids):
        self.mask_ids = torch.tensor(mask_ids, dtype=torch.long)

    def __call__(self, data, ref_dist_matrix):
        if hasattr(data, 'x'):
            data.x[self.mask_ids] = 0
        if hasattr(data, 'x_dihe'):
            data.x_dihe[self.mask_ids] = 0

        row, col = data.edge_index
        node_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        node_mask[self.mask_ids] = True
        edge_mask = node_mask[row] | node_mask[col]
        
        if edge_mask.any():
            masked_indices = torch.where(edge_mask)[0]
            m_rows = row[masked_indices]
            m_cols = col[masked_indices]
            ref_values = ref_dist_matrix[m_rows, m_cols]
            
            valid_mask = ~torch.isnan(ref_values)
            if valid_mask.any():
                final_indices = masked_indices[valid_mask]
                final_values = ref_values[valid_mask]
                if data.edge_attr.dim() > 1:
                    final_values = final_values.unsqueeze(-1)
                
                data.edge_attr = data.edge_attr.clone()
                data.edge_attr[final_indices] = final_values.to(data.edge_attr.dtype)
        return data

class AverageMeter:
    """
    Utility class per calcolare e memorizzare la media e il valore corrente di una metrica.
    Molto utile per pulire i loop di training.
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# =============================================================================
# CONFIGURATION PARSING (TXT/INI)
# =============================================================================

def parse_with_config(parser: argparse.ArgumentParser):
    """
    Parses arguments blending CLI overrides with an optional text config file.
    Expected format: key=value (lines starting with # are comments).
    """
    # 1. Parse initial known args to check if --config was passed
    args, remaining = parser.parse_known_args()
    
    config_dict = {}
    if hasattr(args, 'config') and args.config:
        config_path = args.config
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    
                    # Strip inline comments
                    if ' #' in line:
                        line = line.split(' #')[0].strip()
                        
                    if '=' in line:
                        key, val_str = line.split('=', 1)
                        key = key.strip()
                        val_str = val_str.strip()
                        
                        # Handle spaces -> transform into lists of strings
                        if ' ' in val_str:
                            config_dict[key] = val_str.split()
                        else:
                            # Try to cast single values
                            try:
                                val = ast.literal_eval(val_str)
                                config_dict[key] = val
                            except (ValueError, SyntaxError):
                                # Fallback to true/false boolean or string
                                if val_str.lower() == 'true': config_dict[key] = True
                                elif val_str.lower() == 'false': config_dict[key] = False
                                else: config_dict[key] = val_str
        else:
            print(f"Warning: Configuration file {config_path} not found.")

    # 2. Inject config dictionary as parser defaults
    if config_dict:
        # Resolve argparse destinations because keys might have dashes (e.g. data-class-dirs)
        resolved_defaults = {}
        for action in parser._actions:
            if action.dest in config_dict:
                resolved_defaults[action.dest] = config_dict[action.dest]
            # Handle dashed CLI arguments matched to underscored keys
            elif action.dest.replace('_', '-') in config_dict:
                resolved_defaults[action.dest] = config_dict[action.dest.replace('_', '-')]
                
        parser.set_defaults(**resolved_defaults)

    # 3. Final parse: CLI arguments will naturally override these injected defaults
    return parser.parse_args()

def save_and_reload_config(args: argparse.Namespace, out_path: str, parser: argparse.ArgumentParser = None):
    """
    Dumps the fully resolved argparse namespace into a key=value text file.
    If a parser is provided, it groups the arguments logically and adds explanations as comments,
    excluding internal debug/help flags.
    Returns the namespace for continuity.
    """
    args_dict = vars(args)
    exclude_keys = {'debug_help', 'advanced_help', 'help'}
    
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    
    with open(out_path, 'w') as f:
        f.write("# Auto-generated Configuration File\n")
        f.write("# CLI parameters and defaults have been merged.\n\n")
        
        if parser is None:
            # Fallback for when parser is not provided
            for key, value in sorted(args_dict.items()):
                if value is None or key in exclude_keys:
                    continue # Skip none values and excluded keys
                
                if isinstance(value, list):
                    val_str = " ".join([str(v) for v in value])
                    f.write(f"{key}={val_str}\n")
                elif isinstance(value, bool):
                    f.write(f"{key}={str(value).lower()}\n")
                else:
                    f.write(f"{key}={value}\n")
        else:
            # Enhanced generation grouped by argparse groups
            for group in parser._action_groups:
                # Filter actions that shouldn't be saved
                actions = [a for a in group._group_actions if a.dest not in exclude_keys and a.dest != 'help']
                if not actions:
                    continue
                    
                title = group.title.upper() if group.title else "OPTIONS"
                f.write(f"\n# {'=' * 50}\n")
                f.write(f"# {title}\n")
                if group.description:
                    f.write(f"# {group.description}\n")
                f.write(f"# {'=' * 50}\n")
                
                for action in actions:
                    val = args_dict.get(action.dest)
                    if val is None:
                        val_str = "None"
                    elif isinstance(val, list):
                        val_str = " ".join([str(v) for v in val])
                    elif isinstance(val, bool):
                        val_str = str(val).lower()
                    else:
                        val_str = str(val)
                        
                    # Add trailing comment if help text exists
                    if action.help and action.help != argparse.SUPPRESS:
                        # Clean up help string if it contains linebreaks (which is rare but possible)
                        clean_help = action.help.replace('\n', ' ')
                        f.write(f"{action.dest}={val_str} # {clean_help}\n")
                    else:
                        f.write(f"{action.dest}={val_str}\n")
                        
    return args
