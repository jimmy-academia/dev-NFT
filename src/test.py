from solver import get_solver
from utils import *
import time


args = default_args()
args.setN = None
args.setM = None

args.nft_project_name = nft_project_names[-1]
args.breeding_type = Breeding_Types[1]
Solver = get_solver(args, Baseline_Methods[0])

Solver.solve() 
Solver.evaluate() 
