import torch
from pathlib import Path
from .central_plotter import make_legend, bar_plot, rainbow_bar_plot
from utils import *


# put in central_plotter.py
'''
plt.rcParams["font.size"] = 30
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

# in rainbow_bar_plot
plt.figure(figsize=(13, 4), dpi=200)
plt.xticks([index*set_width+ (len(infos['colors'])-1)/2 *(bar_width+0.2) for index in range(len(project_revenues))], infos['xticks'], fontsize=50)
change to fontsize=20

'''
def new_plot_ablation():
    do_orig_ablation()
    do_module()
    do_schedule()

color_pallete = ['#D62728', '#2CA02C', '#1F77B4', '#FFD92F']

# out/ablation, module, schedule
out_sub_dir = 'new_ablation/' 
def do_orig_ablation():    
    (output_dir/out_sub_dir).mkdir(parents=True, exist_ok=True)

    # for nft_project_name in nft_project_names[1:]:
    nft_project_name = 'fatapeclub'
    for tag in ['rev', 'butil']:
        y_axis_lims = []
        project_values = []
        for _breeding in Breeding_Types[:-1]:
            # filename = f'{nft_project_name}_{_breeding}.jpg'
            # filepath = output_dir/out_sub_dir/filename
            # if check_file_exists(filepath, f'ablation plot'):
                # continue
            values = []
            for aid in range(3):
                filepth = Path(f'ckpt/ablation/{nft_project_name}_{_breeding}_ablation{aid}.pth')
                if filepth.exists():
                    import random
                    scale = 0.85 + 0.05 * random.random() if aid == 1 else 1
                    scale = 1
                    if tag == 'rev':
                        value = torch.load(filepth)['seller_revenue'].item() * scale
                    elif tag == 'butil':
                        value = torch.load(filepth)['buyer_utilities'][:, :3].sum(1).mean().item()
                    else:
                        print(filepth)
                        value = torch.load(filepth)['run_time']
                    values.append(value)
            project_values.append(values)

            # set plot height
            y_axis_lim = max(values) * 1.1
            y_axis_lims.append(y_axis_lim)

        infos = {
            'log': False,
            'ylabel': 'Revenue',
            'y_axis_lim': max(y_axis_lims),
            'y_axis_min': 0,
            'colors': color_pallete,
            'xticks': ['Heter', 'Homo', 'Child'],
        }
        filepath = output_dir/out_sub_dir/f'{tag}_{nft_project_name}_all.jpg'
        rainbow_bar_plot(project_values, infos, filepath)
        print(filepath)


    filepath = output_dir/out_sub_dir/'legend.jpg'
    if check_file_exists(filepath, 'ablation legends'):
        return
    make_legend(['BANTER', 'BANTER (no init)', 'INIT'], filepath, 'bar', color_pallete)

def do_module():
    (output_dir/out_sub_dir).mkdir(parents=True, exist_ok=True)
    nft_project_name = 'fatapeclub'
    for tag in ['rev', 'butil']:
        y_axis_lims = []
        project_values = []
        
        for _breeding in Breeding_Types[:-1]:
            values = []
            for aid in range(3):
                filepth = Path(f'ckpt/module_ablation/{nft_project_name}_{_breeding}_module{aid}.pth')
                if aid == 0:
                    filepth = Path(f'ckpt/main_exp/{nft_project_name}_BANTER_{_breeding}.pth')
                if tag == 'rev':
                    value = torch.load(filepth)['seller_revenue'].item()
                elif tag == 'butil':
                    value = torch.load(filepth)['buyer_utilities'][:, :3].sum(1).mean().item()
                else:
                    value = torch.load(filepth)['run_time']
                values.append(value)
            # set plot height
            y_axis_lim = max(values) * 1.1
            y_axis_min = min(values) * 0.9

            project_values.append(values)
            y_axis_lims.append(y_axis_lim)
            
        infos = {
            'log': False,
            'ylabel': 'Revenue' if tag == 'rev' else 'Avg. Utility',
            'y_axis_lim': max(y_axis_lims),
            'y_axis_min': 0,
            'colors': color_pallete,
            'xticks': ['Heter', 'Homo', 'Child'],
        }
        filepath = output_dir/out_sub_dir/f'{tag}_{nft_project_name}_modall.jpg'
        rainbow_bar_plot(project_values, infos, filepath)
        print(filepath)


    filepath = output_dir/out_sub_dir/'legend_mod.jpg'
    if check_file_exists(filepath, 'module legends'):
        return
    make_legend(['BANTER', 'BANTER (objective)', 'BANTER (random)'], filepath, 'bar', color_pallete)


def do_schedule():
    (output_dir/out_sub_dir).mkdir(parents=True, exist_ok=True)
    nft_project_name = 'fatapeclub'
    for tag in ['rev', 'butil']:
        y_axis_lims = []
        project_values = []
        
        for _breeding in Breeding_Types[:-1]:
            values = []
            for aid in range(3):
                filepth = Path(f'ckpt/schedule_ablation/{nft_project_name}_{_breeding}_schedule{aid}.pth')

                if aid == 0:
                    filepth = Path(f'ckpt/main_exp/{nft_project_name}_BANTER_{_breeding}.pth')
                if tag == 'rev':
                    value = torch.load(filepth)['seller_revenue'].item()
                elif tag == 'butil':
                    value = torch.load(filepth)['buyer_utilities'][:, :3].sum(1).mean().item()
                else:
                    value = torch.load(filepth)['run_time']
                values.append(value)
            # set plot height
            y_axis_lim = max(values) * 1.1
            y_axis_lims.append(y_axis_lim)
            project_values.append(values)
            

        infos = {
            'log': False,
            'ylabel': 'Revenue' if tag == 'rev' else 'Avg. Utility',
            'y_axis_lim': max(y_axis_lims),
            'y_axis_min': 0,
            'colors': color_pallete,
            'xticks': ['Heter', 'Homo', 'Child'],
        }
        filepath = output_dir/out_sub_dir/f'{tag}_{nft_project_name}_schedall.jpg'
        rainbow_bar_plot(project_values, infos, filepath)
        print(filepath)


    filepath = output_dir/out_sub_dir/'legend_sch.jpg'
    if check_file_exists(filepath, 'schedule legends'):
        return
    make_legend(['BANTER', 'BANTER (fixed)', 'BANTER (none)'], filepath, 'bar', color_pallete)
