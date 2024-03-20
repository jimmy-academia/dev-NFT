'''
This is the main module of the dev-NFT project.
'''

__author__ = "Jimmy Yeh"

from data_preprocessing import prepare_nft_data
from experiments import *

def main():
    """
    Run all (unfinished) experiments
    """
    prepare_nft_data() # prepares nft data into files 
    
    run_experiments() 
    run_sensitivity_tests()
    run_runtime_tests()    
    run_ablation_tests()
    adjust_pruning_tests()
    # do_case_study()

if __name__ == "__main__":
    main()