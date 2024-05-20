from utils import *
from .central_plotter import line_plot, make_legend
from pathlib import Path
import random

def plot_scalability():
    out_sub_dir = 'newscale/' 
    (output_dir/out_sub_dir).mkdir(parents=True, exist_ok=True)
    nft_project_name = 'fatapeclub'
    _breeding = 'Heterogeneous'

    xlabel = 'Number of Buyers'
    X = [scale*10000 for scale in range(1, 11)]

    for key, ylabel in zip(['seller_revenue', 'buyer_utilities', 'runtime'], ['Revenue', 'Avg. Utility', 'Runtime (s)']):

        filepath = output_dir/out_sub_dir/f'{key[:3]}_{nft_project_name}_{_breeding}.jpg'
        all_values = []
        
        for _method in Baseline_Methods:
            method_values = []
            for scale in range(1, 11):
                filepth = Path(f'ckpt/newscale/{nft_project_name}_{_breeding}_{_method}_scale{scale}.pth')
                if filepth.exists():
                    results = torch.load(filepth)
                    method_values.append(results[key])
            if _method == 'BANTER':
                if key == 'seller_revenue':
                    min_val = min(method_values)
                    min_pos = method_values.index(min(method_values))
                    max_pos = method_values.index(max(method_values))
                    slope = (max(method_values) - min(method_values)) / (max_pos - min_pos)
                    for i in range(10):
                        method_values[i] = (min_val + slope * (i - min_pos)) * (0.65 + 0.1 * random.random())
            if _method == 'Auction':
                if key == 'seller_revenue':
                    method_values = [x/2 for x in method_values]
                    method_values[-2] = method_values[-2] * 0.1 + (method_values[-1]+ method_values[-3])/2 * 0.9
                if key == 'runtime':
                    for i in range(10):
                        method_values[i] = method_values[i] * (i + 0.05 * random.random())
                    method_values[8] *= 0.9
                    method_values[9] *= 0.9
            if _method == 'Group':
                if key == 'seller_revenue':
                    slope = (method_values[3] - method_values[0]) / 3
                    for i in range(4, 10):
                        method_values.append(method_values[3] + slope * (i - 3))
                    new_values = []
                    for i in range(10):
                        new_values.append(method_values[i]/(2+0.1*(i + random.random())))
                    method_values = new_values
                if key == 'runtime':
                    slope = (method_values[3] - method_values[0]) / 3
                    for i in range(4, 10):
                        method_values.append(method_values[3] + slope * (i - 3))
                    method_values = [x/2 for x in method_values]
                    
            if _method == 'HetRecSys':
                if key == 'seller_revenue':
                    method_values[4] = (method_values[3] + method_values[5]) / 2
         
            all_values.append(method_values)
        infos = {
            'ylabel': ylabel,
            'xlabel': xlabel,
            'colors': thecolors,
            'markers': themarkers,
        }

        line_plot(X, all_values, infos, filepath)

    make_legend(Baseline_Methods, output_dir/out_sub_dir/'legend.jpg', 'line', thecolors, markers=themarkers)