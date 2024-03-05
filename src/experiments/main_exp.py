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

for _method in Baseline_Methods:
    for _breeding in Breeding_Types:
        Solver = get_solver(_method, _breeding)
        Solver.solve() # recommend pricing and NFT purchase for each user

        Solver.evalate() # evaluate buyer utility, seller revenue
        print('save files')
