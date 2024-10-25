from .heuristics import RandomSolver, PopularSolver
from .optimization import GreedySolver, AuctionSolver
from .group import GroupSolver
from .hetrecsys import HetRecSysSolver
from .market import BANTERSolver

from .ncf import NCFSolver
from .lightgcn import LightGCNSolver
from .reciprocal import ReciprocalSolver

def get_solver(args, _method):
    if _method == 'Random':
        return RandomSolver(args)
    if _method == 'Popular':
        return PopularSolver(args)
    if _method == 'Greedy':
        return GreedySolver(args)
    if _method == 'Auction':
        return AuctionSolver(args)
    if _method == 'Group':
        return GroupSolver(args)
    if _method == 'HetRecSys':
        return HetRecSysSolver(args)

    if _method == 'LightGCN':
        return LightGCNSolver(args)
    if _method == 'Reciprocal':
        return ReciprocalSolver(args)
    if _method == 'NCF':
        return NCFSolver(args)

    if _method == 'BANTER':
        return BANTERSolver(args)
