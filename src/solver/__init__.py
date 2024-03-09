from .heuristics import RandomSolver, PopularSolver
from .optimization import GreedySolver, AuctionSolver
from .market import BANTERSolver

def get_solver(args, _method):
    if _method == 'Random':
        return RandomSolver(args)
    if _method == 'Popular':
        return PopularSolver(args)
    if _method == 'Greedy':
        return GreedySolver(args)
    if _method == 'Auction':
        return AuctionSolver(args)
    if _method == 'BANTER':
        return BANTERSolver(args)