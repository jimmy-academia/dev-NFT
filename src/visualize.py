'''
This is the main visualization module of the dev-NFT project.
'''

__author__ = "Jimmy Yeh"

import argparse
from printers.stats import print_nft_data_stats
from printers.main_exp_barplots import plot_main_exp
from printers.ablation_barplots import plot_ablation

def main():
    """
    Run all visualizations
    """
    parser = argparse.ArgumentParser(description='Visualize NFT data')
    parser.add_argument('-c', choices=['stats', 'main', 'ablation', 'all'], default='ablation')
    args = parser.parse_args()
    if args.c == 'stats':
        print_nft_data_stats()
    elif args.c == 'main':
        plot_main_exp()
    elif args.c == 'ablation':
        plot_ablation()
    elif args.c == 'all':
        print_nft_data_stats()
        plot_main_exp()
        plot_ablation()

if __name__ == "__main__":
    main()