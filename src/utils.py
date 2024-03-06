import re
import os
import json
import argparse 
import random
import numpy as np

from pathlib import Path

import torch
from torch.nn.functional import tanh

import code 
import inspect

NFT_Projects = ['Axies Infinity', 'Bored Ape Yacht Club', 'Crypto Kitties', 'Fat Ape Club', 'Heterosis', 'Roaring Leader', 'StepN']
nft_project_names = [''.join(Project_Name.split()).lower() for Project_Name in NFT_Projects]

Baseline_Methods = ['Random']
# Baseline_Methods = ['Random', 'Favorite', 'main']
Breeding_Types = ['Homogeneous', 'ChildProject', 'Heterogeneous']

def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class NamespaceEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, argparse.Namespace):
      return obj.__dict__
    else:
      return super().default(obj)

def dumpj(dictionary, filepath):
    with open(filepath, "w") as f:
        # json.dump(dictionary, f, indent=4)
        obj = json.dumps(dictionary, indent=4, cls=NamespaceEncoder)
        obj = re.sub(r'("|\d+),\s+', r'\1, ', obj)
        obj = re.sub(r'\[\n\s*("|\d+)', r'[\1', obj)
        obj = re.sub(r'("|\d+)\n\s*\]', r'\1]', obj)
        f.write(obj)

def loadj(filepath):
    with open(filepath) as f:
        return json.load(f)

def check():
    current_frame = inspect.currentframe()
    caller_frame = current_frame.f_back
    caller_locals = caller_frame.f_locals
    caller_globals = caller_frame.f_globals
    for key in caller_globals:
        if key not in globals():
            globals()[key] = caller_globals[key]
    code.interact(local=dict(globals(), **caller_locals))

def ask_proceed(objectname='file'):
    ans = input(f'{objectname} exists, proceed?').lower()
    if ans in ['', 'yes', 'y']:
        return True
    else:
        return False

def padd_list(original_list):
    max_length = max(len(sublist) for sublist in original_list)
    padded_list = [sublist + [0] * (max_length - len(sublist)) for sublist in original_list]
    return padded_list

def writef(text, path):
    with open(path, 'w') as f:
        f.write(text)

def mkdirpath(dirpath):
    path_dir = Path(dirpath)
    path_dir.mkdir(parents=True, exist_ok=True)
    return path_dir


def set_plotter():
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as font_manager

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 50
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams['xtick.labelsize'] = 40
    plt.rcParams['ytick.labelsize'] = 40
    
    thecolors = ['#FFD92F', '#2CA02C', '#FF7F0E', '#1F77B4', '#D62728']
    markers = ['X', '^', 'o', 'P', 'D']

    return thecolors, markers


def inclusive_range(end, step):
    return range(step, end+step, step)

def batch_indexes(total, batch_size):
    return (range(i, min(i + batch_size, total)) for i in range(0, total, batch_size))