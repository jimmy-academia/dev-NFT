import torch
from tqdm import tqdm

from .base import BaseSolver
from utils import *

class BANTERSolver(BaseSolver):
    def __init__(self, args):
        super().__init__(args)

    def solve(self):
        '''
        proposed BANTER method for NFT and pricing recommendation to obtain Market Equilibrium
        '''
        # self.pricing, self.holdings.
        ## pricing initialization
        self.pricing = torch.rand(self.nftP.M, device=self.args.device) * self.buyer_budgets.mean()
        for __ in range(16):
            spending = self.Uij/self.pricing 
            spending = spending/spending.sum(1).unsqueeze(1)
            self.pricing = (spending * self.buyer_budgets.unsqueeze(1)).sum(0) / self.nft_counts
                
        ## demand-based optimization
        eps = 1000
        pbar = tqdm(range(128), ncols=88, desc='BANTER Solver!')
        for __ in pbar:
            demand = self.solve_user_demand()
            demand = demand.sum(0)
            excess = demand - self.nft_counts
            # old_self.pricing = self.pricing.clone()
            self.pricing *= ( 1 +  eps * excess/(excess.abs().sum()))
            self.pricing = torch.where(self.pricing < 1e-10, 1e-10, self.pricing) 
            pbar.set_postfix(excess=float(excess.sum()))
        
        self.holdings = self.solve_user_demand()

