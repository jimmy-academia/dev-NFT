import torch
from utils import *
from .project import NFTProject

class BaseSolver:
    def __init__(self, args):
        self.args = args
        nft_project_data = loadj(f'../NFT_data/clean/{args.nft_project_name}.json')
        self.nftP = NFTProject(nft_project_data, args.setN, args.setM)
        self.tensorize()        
        self.breeding_type = args.breeding_type
        self.set_utilities_values()
        

    def tensorize(self):
        num_selections_list = [len(options) for (_, options) in self.nftP.trait_dict.items()]
        flat_one_hot = []
        for item_vector in self.nftP.item_vec_list:
            item_one_hot = [torch.zeros(num_selections).to(self.args.device) for num_selections in num_selections_list]
            for value, one_hot in zip(item_vector, item_one_hot):
                one_hot[value] = 1
            flat_one_hot.append(torch.cat(item_one_hot))
        self.nft_attributes = torch.stack(flat_one_hot)

        flattened_preferences = []
        for user_preferences in self.nftP.user_preferences:
            flattened_preferences.append(torch.Tensor([item for sublist in user_preferences for item in sublist])) 
        self.user_preferences = torch.stack(flattened_preferences).to(self.args.device) + 1

    def set_utilities_values(self):
        self.child_population_factor = 1

        def calculate_raw_value(item_vec_list, trait_counts, args):
            raw_values = []
            for item_vec in item_vec_list:
                value = 0
                for attr_id, count in zip(item_vec, trait_counts):
                    value += torch.log(torch.Tensor([args.M / max(1, count[attr_id])]))
                raw_values.append(value)
            return torch.cat(raw_values).to(args.device)

        def calculate_alpha(raw_values, user_budgets, N):
            total_raw_value = total_budget = 0
            for i in range(N):
                total_raw_value += sum([raw_values[aid] for aid in self.nftP.data['buyer_assets_ids'][i % N] if aid < len(raw_values)])
                total_budget += user_budgets[i]
            return total_budget / total_raw_value

        raw_values = calculate_raw_value(self.nftP.item_vec_list, self.nftP.trait_counts, self.args)
        self.alpha = calculate_alpha(raw_values, self.nftP.user_budgets, self.nftP.N)
        self.Vj = raw_values * self.alpha


    def solve(self):
        '''
        yields:
        self.pricing
        self.holdings (recommendation of purchase amount to each buyer)
        '''
        raise NotImplementedError

    def evaluate(self):
        # assert budget constraints, fix pricing and clip holdings if exceed
        self.holdings = torch.where(self.holdings * self.pricings > self.nftP.user_budgets, self.nftP.user_budgets/self.pricings, self.holdings)
        
        # batch process buyers for buyer utility
        self.utilities = []
        for batch_users in batch_indexes(self.nftP.N, 1000):
            self.utilities.append(self.calculate_buyer_utilities(
                batch_users,
                self.holdings[batch_users],
                self.nftP.user_budgets[batch_users],
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
            subtotals.append((holdings[sub_batch].unsqueeze(2) * self.Att).sum(1))
        subtotals = torch.cat(subtotals, dim=0) + 1
        U_coll = (torch.log(subtotals) * self.Air[user_index]).sum(1)

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
                


