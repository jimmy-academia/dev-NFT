'''
This is the main module of the dev-NFT project.
'''

__author__ = "Jimmy Yeh"

import argparse
from data_preprocessing import prepare_nft_data
from experiments import *

def main():
    """
    Run all (unfinished) experiments
    """
    exp_list = [run_experiments, run_sensitivity_tests, run_runtime_tests, adjust_pruning_tests]
    choices = ['main', 'sensitivity', 'runtime', 'ablation', 'prunning']
    parser = argparse.ArgumentParser(description='Visualize NFT data')
    parser.add_argument('-c', choices=choices+['all'], default='main')
    args = parser.parse_args()
    
    prepare_nft_data() # prepares nft data into files 
    
    if args.c == 'all':
        run_experiments() 
        run_sensitivity_tests()
        run_runtime_tests()    
        run_ablation_tests()
        adjust_pruning_tests()
    else:
        exp_list[choices.index(args.c)]()
    # do_case_study()

if __name__ == "__main__":
    main()