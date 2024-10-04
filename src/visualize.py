'''
This is the main visualization module of the dev-NFT project.
'''

__author__ = "Jimmy Yeh"

import argparse
from printers.stats import print_nft_data_stats
from printers.main_exp_barplots import plot_main_exp
from printers.sensitivity_plots import plot_sensitivity
from printers.new_ablation_barplots import new_plot_ablation
from printers.prunning_plots import plot_prunning
from printers.plot_scale import plot_scalability

def main():
    """
    Run all visualizations
    """
    parser = argparse.ArgumentParser(description='Visualize NFT data')
    parser.add_argument('-c', choices=['stats', 'main', 'sensitivity', 'ablation', 'prunning', 'scale', 'all'], default='scale')
    args = parser.parse_args()
    if args.c == 'stats':
        print_nft_data_stats()
    elif args.c == 'main':
        plot_main_exp()
    elif args.c == 'sensitivity':
        plot_sensitivity()
    elif args.c == 'ablation':
        new_plot_ablation()
    elif args.c == 'prunning':
        plot_prunning()
    elif args.c == 'scale':
        plot_scalability()
    elif args.c == 'all':
        print_nft_data_stats()
        plot_main_exp()
        plot_sensitivity()
        new_plot_ablation()
        plot_prunning()

if __name__ == "__main__":
    main()