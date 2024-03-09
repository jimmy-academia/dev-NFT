'''
This is the main visualization module of the dev-NFT project.
'''

__author__ = "Jimmy Yeh"

import argparse
from printers import *

def main():
    """
    Run all visualizations
    """
    parser = argparse.ArgumentParser(description='Visualize NFT data')
    parser.add_argument('-c', choices=['stats', 'rev', 'all'], default='all')
    args = parser.parse_args()
    if args.c == 'stats':
        print_nft_data_stats()
    elif args.c == 'rev':
        plot_total_revenue()
    elif args.c == 'all':
        print_nft_data_stats()
        plot_total_revenue()

if __name__ == "__main__":
    main()