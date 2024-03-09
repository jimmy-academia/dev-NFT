from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
    
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 50
plt.rcParams["font.weight"] = "bold"
plt.rcParams['xtick.labelsize'] = 40
plt.rcParams['ytick.labelsize'] = 40

thecolors = ['#FFD92F', '#2CA02C', '#FF7F0E', '#1F77B4', '#D62728']
markers = ['X', '^', 'o', 'P', 'D']

output_dir = Path('out')
output_dir.mkdir(parents=True, exist_ok=True)


def bar_plot(revenues, infos, out_sub_dir, filename):
    (output_dir/out_sub_dir).mkdir(parents=True, exist_ok=True)
    plt.rcParams['ytick.labelsize'] = 40
    plt.figure(figsize=(8, 6), dpi=200)
    plt.ylabel(infos['ylabel'])
    plt.ylim(0, infos['y_axis_lim'])
    for x, revenue, color in zip(range(len(revenues)), revenues, thecolors):
        plt.bar(x, revenue, color=color)
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(output_dir/out_sub_dir/filename, bbox_inches='tight') # bbox_inches='tight'??
    plt.close()

    
def make_legend(legends, out_sub_dir, bar=True, markers=None):
    fig, ax = plt.subplots()
    for i in range(len(legends)):
        if bar:
            ax.bar(0, 0, color=colors[i], label=legends[i])
        else:
            ax.plot(0, 0, color=colors[i], label=legends[i], marker=markers[i], markersize=30, linewidth=12)

    # Extract the handles and labels from the plot
    handles, labels = ax.get_legend_handles_labels()

    plt.close(fig)
    legend_fig_width = len(legends) * 1.5  # 1.5 inches per entry, adjust as needed
    fig_legend = plt.figure(figsize=(legend_fig_width, 1), dpi=300)  # High DPI for quality
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis('off')
    ax_legend.legend(handles, labels, loc='center', ncol=len(legends), frameon=False, fontsize=50)
    fig_legend.savefig(output_dir/out_sub_dir/'legend.jpg', bbox_inches='tight')
    plt.close(fig_legend)
