from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
    
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 50
plt.rcParams["font.weight"] = "bold"
plt.rcParams['xtick.labelsize'] = 40
plt.rcParams['ytick.labelsize'] = 40

thecolors = ['#FFD92F', '#2CA02C', '#FF7F0E', '#1F77B4', '#008080', '#ADD8E6', '#D62728']
markers = ['X', '^', 'o', 'P', 'D']
patterns = ['x', 'o', '']


def bar_plot(revenues, infos, filepath):
    plt.rcParams['ytick.labelsize'] = 40
    plt.figure(figsize=(8, 6), dpi=200)
    plt.ylabel(infos['ylabel'])
    plt.ylim(0, infos['y_axis_lim'])
    for x, revenue, color in zip(range(len(revenues)), revenues, thecolors):
        plt.bar(x, revenue, color=color)
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight') # bbox_inches='tight'??
    plt.close()

def tripple_bar_plot(project_revenues, infos, filepath):
    plt.rcParams['ytick.labelsize'] = 40
    plt.figure(figsize=(8, 6), dpi=200)
    plt.ylabel(infos['ylabel'])
    plt.ylim(0, infos['y_axis_lim'])
    
    bar_width = 0.25

    indexes = range(len(project_revenues))
    # from utils import check
    # check()
    for index, rev_tripple, color in zip(indexes, project_revenues, thecolors):
        for k, (rev, pat) in enumerate(zip(rev_tripple, patterns)):
            plt.bar(index+k*bar_width, rev, bar_width, color=color, hatch=pat)
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight') # bbox_inches='tight'??
    plt.close()


def make_legend(legends, filepath, tag, markers=None):

    # tag = 'bar', 'tripple', 'line'
    fig, ax = plt.subplots()
    if tag == 'bar':
        [ax.bar(0, 0, color=thecolors[i], label=legends[i]) for i in range(len(legends))]
    elif tag == 'line':
        [ax.plot(0, 0, color=thecolors[i], label=legends[i], marker=markers[i], markersize=30, linewidth=12)  for i in range(len(legends))]
    elif tag == 'tripple':
        [ax.bar(0,0, color=thecolors[i], label=legends[i]) for i in range(len(legends) - 3)]
        [ax.bar(0,0, color='white', edgecolor='black', hatch=patterns[i], label=legends[i-3]) for i in range(3)]
    else:
        raise Exception(f'{tag} not found')

    # Extract the handles and labels from the plot
    handles, labels = ax.get_legend_handles_labels()

    plt.close(fig)
    legend_fig_width = len(legends) * 1.5  # 1.5 inches per entry, adjust as needed
    fig_legend = plt.figure(figsize=(legend_fig_width, 1), dpi=300)  # High DPI for quality
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis('off')
    ax_legend.legend(handles, labels, loc='center', ncol=len(legends), frameon=False, fontsize=50)
    fig_legend.savefig(filepath, bbox_inches='tight')
    plt.close(fig_legend)
