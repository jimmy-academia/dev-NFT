import torch

from .central_plotter import bar_plot, make_legend
from utils import *

def plot_total_revenue():
    out_sub_dir = 'total_revenue/' 
    for nft_project_name in nft_project_names:
        project_revenues = []
        for _breeding in Breeding_Types:
            revenues = []
            for _method in Baseline_Methods:
                filepath = f'ckpt/main_exp/{nft_project_name}_{_method}_{_breeding}.pth'
                revenues.append(torch.load(filepath)['seller_revenue'])
        project_revenues.append(revenues)

        # set plot height
        y_axis_lim = max([max(revenues) for revenues in project_revenues])
        y_axis_lim = y_axis_lim + 0.1 * y_axis_lim
        for _breeding, revenues in zip(Breeding_Types, project_revenues):
            infos = {
                'ylabel': 'Revenue',
                'y_axis_lim': y_axis_lim
            }
            filename = f'{nft_project_name}_{_breeding}.jpg'
            bar_plot(revenues, infos, out_sub_dir, filename)
    make_legend(Baseline_Methods, out_sub_dir)

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

