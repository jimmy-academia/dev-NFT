import torch
from utils import *
from .project import NFTProject

class BaseSolver:
    def __init__(self, args):
        self.args = args
        nft_project_data = loadj(f'../NFT_data/clean/{args.nft_project_name}.json')
        self.nftP = NFTProject(nft_project_data, args.setN, args.setM)
        self.prepare_tensors()
        self.breeding_type = args.breeding_type
        self.set_utilities_values()

    def tensorize(self, label_vec):
        # variables
        num_selections_list = [len(options) for (_, options) in self.nftP.trait_dict.items()]
        max_selections = max(num_selections_list)
        num_attributes = len(num_selections_list)

        # tensorize
        label_vec_tensor = torch.LongTensor(label_vec).to(self.args.device)
        label_vec_tensor = label_vec_tensor.unsqueeze(2) if len(label_vec_tensor.shape) == 2 else label_vec_tensor
        binary = torch.zeros(label_vec_tensor.shape[0], num_attributes, max_selections).to(self.args.device)
        binary.scatter_(2, label_vec_tensor, 1)
        return binary.view(label_vec_tensor.shape[0], -1)

    def prepare_tensors(self):
        self.nft_attributes = self.tensorize(self.nftP.item_attributes)
        self.buyer_preferences = self.tensorize(self.nftP.user_preferences)
        self.buyer_preferences = torch.softmax(self.buyer_preferences, dim=1)
        buyer_budgets = torch.Tensor(self.nftP.user_budgets).to(self.args.device)
        # scale to  [10, 100]
        buyer_budgets.clamp_(min=0)  # Ensure that the minimum value is 0
        buyer_budgets.sub_(buyer_budgets.min()).div_(buyer_budgets.max() - buyer_budgets.min()).mul_(90).add_(10)
        self.buyer_budgets = buyer_budgets
        self.nft_counts = torch.LongTensor(self.nftP.item_counts).to(self.args.device)

    def set_utilities_values(self):
        self.child_population_factor = 1

        def calculate_raw_value(item_vec_list, trait_counts, args, M):
            raw_values = []
            for item_vec in item_vec_list:
                value = 0
                for attr_id, count in zip(item_vec, trait_counts):
                    value += torch.log(torch.Tensor([M / max(1, count[attr_id])]))
                raw_values.append(value)
            return torch.cat(raw_values).to(args.device)

        raw_values = calculate_raw_value(self.nftP.item_attributes, self.nftP.trait_counts, self.args, self.nftP.M)
        self.alpha = sum(self.nftP.user_budgets) / sum(raw_values)
        self.Vj = raw_values * self.alpha

    def solve(self):
        '''
        yields:
        self.pricing
        self.holdings (recommendation of purchase amount to each buyer)
        '''
        raise NotImplementedError

    def evaluate(self):
        print('evaluating....')
        epsilon = 1e-6
        self.holdings.clamp_(min=0)
        self.pricing.clamp_(min=0)
        self.holdings *= torch.clamp(self.nft_counts / (self.holdings.sum(0)+epsilon), max=1) ## fit item constraints
        self.holdings *= torch.clamp(self.buyer_budgets / ((self.holdings * self.pricing).sum(1)+epsilon), max=1).view(-1, 1) ## fit budget constraints
        # batch process buyers for buyer utility
        self.utilities = []
        for batch_users in batch_indexes(self.nftP.N, 100):
            self.utilities.append(self.calculate_buyer_utilities(
                batch_users,
                self.holdings[batch_users],
                self.buyer_budgets[batch_users],
                self.pricing,   
            ))
        self.utilities = torch.cat(self.utilities, dim=0)

        # seller revenue
        self.revenue = (self.pricing * self.holdings).sum(0)

    def calculate_buyer_utilities(self, user_index, holdings, budgets, pricing):
        '''
        U^i = U^i_{Item} + U^i_{Collection} + U^i{Breeding}
        U^i_{Item} = sum_j V_j * holdings_j
        U^i_{Collection} = sum_r  a^i_r log (sum_j multistep(x^i[j], Q[j]) t_jr)
        U^i_{Breeding} = sum_k topk expectation value * multistep(x^i[p], Q[p]) * multistep(x^i[q], Q[q])
        R = budgets - sum_j price_j * holdings_j
        '''
        U_item = (holdings * (self.Vj)).sum(1)
        # sub_batch operation
        chunk_size = 32 
        subtotals = []
        for sub_batch in batch_indexes(len(holdings), chunk_size):
            subtotals.append((holdings[sub_batch].unsqueeze(2) * self.nft_attributes).sum(1))
        subtotals = torch.cat(subtotals, dim=0) + 1
        U_coll = (torch.log(subtotals) * self.buyer_preferences[user_index]).sum(1)

        R = budgets - (holdings * pricing).sum(1)

        U_breeding = self.breeding_utility(holdings, user_index)
        return U_item + U_coll + U_breeding + R
    
    def breeding_utility(self, holdings, user_index):
        U_breeding = 0
        if self.breeding_type == 'none':
            return U_breeding
        
        # calculate probability * expectation up to topk
        parents = self.ranked_parent_nfts[user_index]
        parent_nft_probs = [torch.gather(holdings, 1, parents[..., p]) for p in range(parents.shape[-1])]
        probability = torch.prod(torch.stack(parent_nft_probs), dim=0)
        expectation = self.ranked_paraent_expectations[user_index]

        cum_prob = torch.cumsum(probability, dim=1)
        selection_mask = torch.where(cum_prob > self.breeding_topk, probability, torch.zeros_like(probability))

        if self.breeding_type == 'homogeneous':
            # calculate frequencies based on selection_mask 
            parent_attr_freq = torch.stack([self.nft_attributes[parents[..., p]]*selection_mask for p in range(parents.shape[-1])]).sum(0)
            # adjust expectation
            child_population_factor = (torch.stack([self.nft_attributes[parents[..., p]] for p in range(parents.shape[-1])]) * parent_attr_freq).sum(0)
            expectation = expectation / (1+ child_population_factor/10)
            print('check!!')

        U_breeding = (selection_mask * expectation).sum(1)

        return U_breeding
                


