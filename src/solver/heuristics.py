import torch
from .base import BaseSolver

class RandomSolver(BaseSolver):
    def __init__(self, args):
        super().__init__(args)
    
    def solve(self):
        # random recommendation of NFT to buyers
        self.holdings = torch.zeros(self.nftP.N, 100)

        k = 10 if self.nftP.N * 10 > self.nftP.M else self.nftP.M//self.nftP.N +1 

        random_assignments = torch.stack([torch.randperm(self.nftP.M) for _ in range(self.nftP.N)])[:, :k]
        self.holdings[torch.arange(self.nftP.N)[:, None], random_assignments] = 1

        unassigned = [eta_j for eta_j in torch.arange(self.nftP.M) if eta_j not in random_assignments.unique()]
        unassigned_indices = torch.randint(high=self.nftP.N, size=(len(unassigned),))
        self.holdings[unassigned_indices, unassigned] = 1
        
        # determine pricing, with self.holdings as binary matrix
        num_items = self.holdings.sum(dim=1)
        allocated_budgets = self.nftP.user_budgets / num_items
        item_budgets = self.holdings * allocated_budgets.unsqueeze(1)
        self.pricing = item_budgets.sum(dim=0)/self.nftP.item_count

        # determine holding
        self.holdings = item_budgets/self.pricing
        
class FavoriteSolver(BaseSolver):
    def __init__(self, args):
        super().__init__(args)
    
    def solve(self):
        print('todo')