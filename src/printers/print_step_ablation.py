

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 30
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

import sys 
sys.path.append('.')
from printers.central_plotter import make_legend



a = [3090.61572265625, 10560.900390625, 15984.0224609375, 16414.244140625, 16955.89453125, 17461.103515625, 18496.29296875, 18541.6484375, 18540.060546875]
b = [1310.11865234375, 3341.5537109375, 4815.095703125, 6716.3623046875, 9263.5439453125, 12655.548828125, 14778.748046875, 17294.212890625, 17330.17578125]
c = [3090.61572265625]*9

X = list(range(1,18,2))
use_colors = ['#D62728', '#2CA02C', '#1F77B4']
use_markers = ['*', '^', 'P']
infos = {
    'ylabel': 'Revenue',
    'xlabel': 'iteration step',
    'colors': use_colors,
    'markers': use_markers,
    'xticks': X,
}

def line_plot(X, project_values, infos, filepath):
    figsize = infos['figsize'] if 'figsize' in infos else (13, 4)
    plt.figure(figsize=figsize, dpi=200)
    plt.ylabel(infos['ylabel'], fontsize=40, fontweight='bold', y=0.3)
    plt.xlabel(infos['xlabel'], fontsize=40, fontweight='bold')
    plt.tick_params(axis='y', labelsize=20)
    for values, color, marker in zip(project_values, infos['colors'], infos['markers']):
        plt.plot(X[:len(values)], values, color=color, marker=marker, markersize=18, linewidth=3.5)
    if 'legends' in infos:
        plt.legend(infos['legends'], loc='upper left', fontsize=30, markerscale=1.8)
    if 'no_xtic' in infos and infos['no_xtic']:
        plt.xticks([])
    if 'xticks' in infos:
        plt.xticks(infos['xticks'], fontsize=20)

    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


line_plot(X, [a,b,c], infos, 'out/step/ablation_step.jpg')


d = [3090.61572265625, 10560.900390625, 15984.0224609375, 16414.244140625, 16955.89453125, 17461.103515625, 17496.29296875, 17541.6484375, 17540.060546875]
e = [3090.61572265625, 6642.15869140625, 11337.4375, 12051.76953125, 13675.615234375, 14268.234375, 15950.6904296875, 16144.837890625, 16129.8046875]
f = [2918.892578125, 6313.533203125, 10855.30859375, 11902.2236328125, 12072.8232421875, 12354.380859375, 13994.8837890625, 14892.6953125, 14893.2099609375]


line_plot(X, [d,e,f], infos, 'out/step/schedule_step.jpg')

make_legend(['BANTER', 'BANTER (no init)', 'INIT'], 'out/step/ablation_legend.jpg', 'line', use_colors, patterns=None, markers=use_markers)
make_legend(['BANTER', 'BANTER (fixed)', 'BANTER (none)'], 'out/step/schedule_legend.jpg', 'line', use_colors, patterns=None, markers=use_markers)


print('=== all done ===')

