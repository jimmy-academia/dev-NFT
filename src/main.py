'''
This is the main module of the dev-NFT project.
'''

__author__ = "Jimmy Yeh"

from data_preprocessing import prepare_nft_data


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
    
    
    # for project in 

if __name__ == "__main__":
    main()