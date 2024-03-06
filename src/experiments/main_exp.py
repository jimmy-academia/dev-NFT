from solver import get_solver
from utils import *

'''
Main experiment for 5+1 methods x 3 breeding cases

Todo:
define arguments
initiate solver with argument
run solver => store into file
collect results from file and visualize => save into figures/tables...
'''

def run_experiments():
    args = default_args
    args.setN = None
    args.setM = None

    for nft_project_name in nft_project_names:
        args.nft_project_name = nft_project_name
        for _method in Baseline_Methods:
            for _breeding in Breeding_Types:
                args.breeding_type = _breeding
                Solver = get_solver(args, _method)
                Solver.solve() # recommend pricing and NFT purchase for each user

                Solver.evalate() # evaluate buyer utility, seller revenue
                print('save files')
