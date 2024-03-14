import torch
from pathlib import Path

from .central_plotter import make_legend, tripple_bar_plot
from utils import *




def plot_total_revenue():
    out_sub_dir = 'total_revenue/' 
    (output_dir/out_sub_dir).mkdir(parents=True, exist_ok=True)

    for nft_project_name in nft_project_names:
        filename = f'{nft_project_name}.jpg'
        filepath = output_dir/out_sub_dir/filename
        if check_file_exists(filepath, 'revenue plot'):
            continue

        project_revenues = []
        for _method in Baseline_Methods:
            revenues = []
            for _breeding in Breeding_Types:
                filepth = Path(f'ckpt/main_exp/{nft_project_name}_{_method}_{_breeding}.pth')
                if filepth.exists():
                    revenues.append(torch.load(filepth)['seller_revenue'].item())
                else:
                    revenues.append(0)
            project_revenues.append(revenues)

        # set plot height
        y_axis_lim = max([max(revenues) for revenues in project_revenues])
        y_axis_lim = y_axis_lim + 0.1 * y_axis_lim

        infos = {
            'ylabel': 'Revenue',
            'y_axis_lim': y_axis_lim
        }
        tripple_bar_plot(project_revenues, infos, filepath)

    filepath = output_dir/out_sub_dir/'legend.jpg'
    if check_file_exists(filepath, 'revenue legends'):
        return
    make_legend(Baseline_Methods + Breeding_Types, filepath, 'tripple')

def quick_check():
    ## quick check
    for nft_project_name in nft_project_names:
        for _breeding in Breeding_Types:
            revenues = []
            method_list = []
            for _method in Baseline_Methods:
                filepath = f'ckpt/main_exp/{nft_project_name}_{_method}_{_breeding}.pth'
                try:
                    revenues.append(torch.load(filepath)['seller_revenue'])
                    method_list.append(_method)
                except:
                    pass

            if len(revenues) > 0:
                print(f'[{nft_project_name} {_breeding}]')
                print(method_list)
                print(f'rev: {revenues}')

