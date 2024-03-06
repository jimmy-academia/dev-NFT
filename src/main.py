'''
This is the main module of the dev-NFT project.
'''

__author__ = "Jimmy Yeh"

from data_preprocessing import prepare_nft_data
from experiments import *

def main():
    """
    Run all unfinished experiments
    """
    print(
        '''Todos:
        1. data processing
        2. experiment + method
        3. visualization
        ''' 
    )
    prepare_nft_data() # prepares nft data into files 

    # do different experiments
    run_experiments() # run all experiments for each baseline methods
    # ablation_tests()
    # runtime_tests()
    # case_study()
    # large_scale_dupliate()

if __name__ == "__main__":
    main()