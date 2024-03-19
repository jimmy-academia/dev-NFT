import torch
from pathlib import Path

from .central_plotter import make_legend, bar_plot
from utils import *

def plot_ablation():
    
    out_sub_dir = 'ablation/' 
    (output_dir/out_sub_dir).mkdir(parents=True, exist_ok=True)

    color_pallete = thecolors[:-4:-1]
    for nft_project_name in nft_project_names[1:]:
        for _breeding in Breeding_Types:
            filename = f'{nft_project_name}_{_breeding}.jpg'
            filepath = output_dir/out_sub_dir/filename
            if check_file_exists(filepath, f'ablation plot'):
                continue
            values = []
            for aid in range(3):
                filepth = Path(f'ckpt/ablation/{nft_project_name}_{_breeding}_ablation{aid}.pth')
                if filepth.exists():
                    values.append(torch.load(filepth)['seller_revenue'].item())

            # set plot height
            y_axis_lim = max(values) * 1.1
            infos = {
                'ylabel': 'Revenue',
                'y_axis_lim': y_axis_lim,
                'colors': color_pallete
            }
            bar_plot(values, infos, filepath)

    filepath = output_dir/out_sub_dir/'legend.jpg'
    if check_file_exists(filepath, 'ablation legends'):
        return
    make_legend(['BANTER', 'BANTER (no init)', 'BANTER (only init)'], filepath, 'bar', color_pallete)

