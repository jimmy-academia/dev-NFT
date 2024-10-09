import torch
from pathlib import Path
from utils import *

def plot_main_exp():
    out_sub_dir = 'main_exp/' 
    (output_dir/out_sub_dir).mkdir(parents=True, exist_ok=True)
    for tag, ylabel in zip(['revenue', 'buyer_utility', 'runtime'], ['Revenue', 'Avg. Utility', 'Runtime (s)']):
        for nft_project_name in nft_project_names:
            filename = f'{tag}_{nft_project_name}.jpg'
            filepath = output_dir/out_sub_dir/filename
            if check_file_exists(filepath, f'main_exp {tag} plot'):
                continue

            if Path(f'ckpt/temp/{tag}_temp.pth').exists():
                project_values, infos = torch.load(f'ckpt/temp/{tag}_temp.pth')
            else:

                project_values = []
                # xticks = None
                # xticks = ['Hetero-\ngeneous', 'Homo-\ngeneous', 'Child\nProject', 'No\nBreeding']
                xticks = ['Heter', 'Homo', 'Child', 'None']

                for _breeding in Breeding_Types:
                    breed_vals = []
                    for _method in Baseline_Methods:
                        filepth = Path(f'ckpt/main_exp/{nft_project_name}_{_method}_{_breeding}.pth')
                        if filepth.exists():
                            if tag == 'revenue':
                                breed_vals.append(torch.load(filepth)['seller_revenue'].item())
                            elif tag == 'buyer_utility':
                                if _method == 'BANTER':
                                    breed_vals.append(torch.load(filepth)['buyer_utilities'][:, :3].sum(1).mean().item())  ## average buyer utility
                                else:
                                    breed_vals.append(torch.load(filepth)['buyer_utilities'][:, :2].sum(1).mean().item())  ## no breeding recommendation
                            elif tag == 'runtime':
                                breed_vals.append(torch.load(filepth)['runtime'])
                                # xticks = ['Hetero-\ngeneous', 'Homo-\ngeneous', 'Child\nProject', 'No\nBreeding']
                        else:
                            breed_vals.append(0)
                    project_values.append(breed_vals)

                # set plot height
                y_axis_lim = max([max(breed_vals) for breed_vals in project_values])
                # _increase =  if tag == 'runtime' else 0.1
                y_axis_lim = y_axis_lim + 0.1 * y_axis_lim

                y_axis_min = min([min(breed_vals) for breed_vals in project_values]) if tag == 'runtime' else 0
                y_axis_min = y_axis_min - 0.1 * y_axis_min

                infos = {
                    'log': tag == 'runtime',
                    'ylabel': ylabel,
                    'y_axis_lim': y_axis_lim,
                    'y_axis_min': y_axis_min,
                    'colors': thecolors,
                    'xticks': xticks,
                }
            # torch.save([project_values, infos], f'{tag}_temp.pth')
            rainbow_bar_plot(project_values, infos, filepath)

    filepath = output_dir/out_sub_dir/'legend.jpg'
    if check_file_exists(filepath, 'main_exp legends'):
        return
    make_legend(Baseline_Methods, filepath, 'bar', thecolors)

