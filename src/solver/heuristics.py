import torch
from .base import BaseSolver

from utils import *

class RandomSolver(BaseSolver):
    def __init__(self, args):
        super().__init__(args)
        print('inialized random solver')
    
    def solve(self):
        # random recommendation of NFT to buyers
        self.holdings = torch.zeros(self.nftP.N, self.nftP.M).to(self.args.device)

        k = 10 if self.nftP.N * 10 > self.nftP.M else self.nftP.M//self.nftP.N +1 

        random_assignments = torch.stack([torch.randperm(self.nftP.M)[:k] for _ in range(self.nftP.N)])
        self.holdings[torch.arange(self.nftP.N)[:, None], random_assignments] = 1

        all_values = torch.arange(self.nftP.M)
        flat_assignments = random_assignments.flatten()
        zero_tensor = torch.zeros(self.nftP.M, dtype=torch.bool)
        zero_tensor.index_fill_(0, flat_assignments, True)
        unassigned = all_values[~zero_tensor]

        unassigned_indices = torch.randint(high=self.nftP.N, size=(len(unassigned),))
        self.holdings[unassigned_indices, unassigned] = 1

        budget_per_item = self.buyer_budgets / self.holdings.sum(1)
        buyer_spendings = self.holdings * budget_per_item.unsqueeze(1)
        self.pricing = buyer_spendings.sum(0)/self.nft_counts
        # determine holding
        self.holdings = buyer_spendings/self.pricing * (1-1e-4)
        # self.holdings = torch.where(self.holdings == 0, 0, self.holdings-1e-4)

        # assert all(self.holdings.sum(0) <= self.nft_counts) ## item constraint
        # assert all((self.holdings*self.pricing).sum(1) <= self.buyer_budgets) ## budget constraint

class FavoriteSolver(BaseSolver):
    def __init__(self, args):
        super().__init__(args)
    
    def solve(self):
        print('todo')