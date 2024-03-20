import torch
from pathlib import Path

from .central_plotter import line_plot, make_legend
from utils import *

def plot_sensitivity():
    out_sub_dir = 'sensitivity/' 
    (output_dir/out_sub_dir).mkdir(parents=True, exist_ok=True)

    N_M_infos = loadj('ckpt/N_M_infos.json')

    for nft_project_name in nft_project_names[-2:-1]:
        for _breeding in Breeding_Types[2:3]:
            for tag in ['num', 'run', 'bud']:
                project_values = []

                N = N_M_infos[nft_project_name]['N']
                X = list(range(10, 110, 10)) if tag == 'bud' else [N//10 * scale for scale in range(1, 10)] + [N]
                filename = f'{tag}_{nft_project_name}_{_breeding}.jpg'
                filepath = output_dir/out_sub_dir/filename
                if check_file_exists(filepath, f'{tag} sensitivity plot'):
                    continue

                s = 1
                if tag == 'num':
                    s = torch.load(f'ckpt/main_exp/{nft_project_name}_BANTER_{_breeding}.pth')['seller_revenue'].item() / torch.load(f'ckpt/sensitivity/{nft_project_name}_BANTER_{_breeding}_num10.pth')['seller_revenue'].item()
                

                for _method in Baseline_Methods:
                    values = []
                    for scale in range(1, 11):
                        _tag = 'bud' if tag == 'bud' else 'num'
                        filepth = Path(f'ckpt/sensitivity/{nft_project_name}_{_method}_{_breeding}_{_tag}{scale}.pth')
                        if tag in ['num', 'bud']:
                            values.append(torch.load(filepth)['seller_revenue'].item()*s)
                        else:
                            values.append(torch.load(filepth)['runtime'])
                    project_values.append(values)

                ylabel = 'Runtime (s)' if tag == 'run' else 'Revenue'
                xlabel = 'Buyer Budget (%)' if tag == 'bud' else 'Number of Buyers'
                infos = {
                    'ylabel': ylabel,
                    'xlabel': xlabel,
                    'colors': thecolors,
                    'markers': themarkers,
                }
                line_plot(X, project_values, infos, filepath)

            filepath = output_dir/out_sub_dir/'legend.jpg'
            if check_file_exists(filepath, 'sensitivity legends'):
                return
            make_legend(Baseline_Methods, filepath, 'line', thecolors, markers=themarkers)

