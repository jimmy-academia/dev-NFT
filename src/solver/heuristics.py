import torch
from .base import BaseSolver
from utils import *

class HeuristicsSolver(BaseSolver):
    def __init__(self, args):
        super().__init__(args)
        # self.k = 128
        self.k = 2

    def initial_assignment(self):
        raise NotImplementedError
    
    def solve(self):
        # random or popular recommendation of NFT to buyers
        _assignments = self.initial_assignment()

        # self.holdings = torch.zeros(self.nftP.N, self.nftP.M)
        # self.holdings[torch.arange(self.nftP.N)[:, None], _assignments] = 1
        
        batch_size = 1000
        budget_per_item = self.buyer_budgets.cpu() / self.k

        self.pricing = torch.zeros(self.nftP.M)

        for batch_users in make_batch_indexes(self.nftP.N, batch_size):
            batch_size = len(batch_users)
            holdings = torch.zeros(batch_size, self.nftP.M)
            holdings[torch.arange(batch_size)[:, None], _assignments[batch_users]] = 1
            buyer_spendings = holdings * budget_per_item[batch_users].unsqueeze(1)
            self.pricing += buyer_spendings.sum(0)/self.nft_counts.cpu()

        self.pricing.clamp_(0.1)
        self.pricing = self.pricing.to(self.args.device)

class RandomSolver(HeuristicsSolver):
    def __init__(self, args):
        super().__init__(args)
    def initial_assignment(self):
        random_assignments = torch.stack([torch.randperm(self.nftP.M)[:2] for _ in range(self.nftP.N)]).to(self.args.device)
        return random_assignments

class PopularSolver(HeuristicsSolver):
    def __init__(self, args):
        super().__init__(args)

    def initial_assignment(self):
        favorite_assignments = (self.Uij * self.Vj).topk(100)[1][:, -self.k:]  #shape N, k
        return favorite_assignments
