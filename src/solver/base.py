import torch
from utils import *

class NFTProject:
    def __init__(self, nft_project_data, setN, setM):
        self.N = setN if setN is not None else len(nft_project_data['buyer_budgets'])
        self.M = setM if setM is not None else len(nft_project_data['asset_traits'])
        self.trait_dict = nft_project_data['trait_system']
        self.item_trait_vectorize(nft_project_data['asset_traits'])
        self.compute_trait_stats() 
        self.item_count = nft_project_data['item_counts']
        self.get_user_preferences(nft_project_data['buyer_assets_ids'], nft_project_data['asset_traits'])
        self.get_user_budgets(nft_project_data['buyer_budgets'])

    def item_trait_vectorize(self, asset_traits):
        self.item_vec_list = []
        for item in asset_traits[:self.M]:
            item_vec = []
            for (trait, options), choice in zip(self.trait_dict.items(), item):
                try:
                    item_vec.append(options.index(choice))
                except:
                    check()
            self.item_vec_list.append(item_vec)

    def compute_trait_stats(self):
        self.trait_counts = []
        for trait, options in self.trait_dict.items():
            self.trait_counts.append([1 for __ in options])
        for item_vec in self.item_vec_list:
            for t, choice in enumerate(item_vec):
                self.trait_counts[t][choice] += 1

    def get_user_preferences(self, buyer_assets_ids, asset_traits):
        self.user_preferences = []
        for __ in range(self.N):
            user_pref = []
            for trait, options in self.trait_dict.items():
                user_pref.append([0 for option in options])
            self.user_preferences.append(user_pref)
        for i in range(self.N):
            attr_list = asset_traits[buyer_assets_ids[i]]
            for attr in attr_list:
                for t, (trait, options) in enumerate(self.trait_dict.items()):
                    self.user_preferences[i][t][options.index(attr[t])] +=1

    def get_user_budgets(self, buyer_budgets):
        print('do normalization??!!!')
        self.user_budgets = buyer_budgets


class BaseSolver:
    def __init__(self, args):
        nft_project_data = loadj(f'../NFT_data/clean/{args.nft_project_name}.json')
        self.nftP = NFTProject(nft_project_data, args.setN, args.setM)
        self.breeding_type = args.breeding_type
        self.set_utilities_values()

    def set_utilities_values(self):
        self.child_population_factor = 1

        print('=====todo======')
        nsel_list = [len(options) for (__, options) in self.nftP.trait_dict.items()]
        flatonehot = []
        for item_vec in self.nftP.item_vec_list:
            itemonehot = []
            for x, nsel in zip(item_vec, nsel_list):
                onehot = torch.zeros(nsel)
                onehot[x] = 1
                itemonehot.append(onehot)
            flatonehot.append(torch.cat(itemonehot))
        self.Att = torch.stack(flatonehot).to(self.args.device)

        flatpreferences = []
        for user_pref in self.nftP.user_preferences:
            flatpreferences.append(torch.Tensor([it for sub in user_pref for it in sub])) 
        self.Air = torch.stack(flatpreferences).to(self.args.device) + 1

        raw_value = []
        for item_vec in self.nftP.item_vec_list:
            u = 0
            for attr_id, count in zip(item_vec, self.nftP.trait_counts):
                u += torch.log(torch.Tensor([self.args.M / max(1, count[attr_id])]))
            raw_value.append(u)
        raw_value = torch.cat(raw_value).to(self.args.device)
        
        origN = self.args.N // self.args.duplicate
        total_raw_value = total_budget = 0
        for i in range(self.args.N):
            total_raw_value += sum([raw_value[aid] for aid in self.nftP.data['buyer_assets_ids'][i % origN] if aid < len(self.nftP.item_vec_list)])
            total_budget += self.nftP.user_budgets[i]

        self.alpha = total_budget/total_raw_value 
        self.Vj = raw_value * self.alpha


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
        probability = torch.take(holdings, self.ranked_parent_nfts[user_index])
        # modify expectation value by parent population factor
        # accumulate the parent population factor for next round
        cumulative_probability = 0

        for prob, exp in zip(probability, expectation):
            proportion = min(1, (self.breeding_topk - cumulative_probability) / prob)
            U_breeding += prob * exp * proportion
            cumulative_probability += prob
            if cumulative_probability >= self.breeding_topk:
                break

        if self.breeding_type == 'homogeneous':
            parent_population_factor = self.nftP.item_count[self.ranked_parent_nfts[user_index]]
            expectation *= parent_population_factor
            self.ranked_paraent_expectations[user_index] = expectation
        else:
            expectation = self.ranked_paraent_expectations[user_index]

        return U_breeding
                
    def breeding_utility(self, user_index):
        cumulative_probability = 0
        selection_probabilities = []
        selection_expectations = []
        selected_parent_sets = []

        for parent_set in self.get_ranked_parent_sets(user_index):
            if cumulative_probability >= self.breeding_topk:
                break
            
            probability, expectation = self.calculate_parent_set_utility(parent_set)
            cumulative_probability += probability

            selection_probabilities.append(probability)
            selection_expectations.append(expectation)
            selected_parent_sets.append(parent_set)

        if self.breeding_type == 'homogeneous':
            attribute_frequencies = self.calculate_dynamic_frequencies(selected_parent_sets)
            adjusted_expectations = self.adjust_expectations(selection_expectations, attribute_frequencies)
            utility = self.calculate_final_utility(selection_probabilities, adjusted_expectations)
        else:
            utility = sum(prob * exp for prob, exp in zip(selection_probabilities, selection_expectations))
        
        return utility

def calculate_parent_set_utility(self, parent_set):
    # Calculate and return the probability and expectation for a given parent set
    # This is a placeholder; you'll need to implement based on your model's logic
    return probability, expectation

def calculate_dynamic_frequencies(self, selected_parent_sets):
    # Dynamically calculate the frequencies of attributes across selected parent sets
    # This function should return a structure (like a dictionary) mapping attributes to their frequencies
    frequencies = {}
    # Implement frequency calculation
    return frequencies

def adjust_expectations(self, expectations, frequencies):
    # Adjust the expectations based on attribute frequencies
    adjusted = []
    for expectation in expectations:
        # Modify expectation based on frequencies
        adjusted.append(modified_expectation)
    return adjusted

def calculate_final_utility(self, probabilities, expectations):
    # Calculate the final utility by applying adjusted expectations to the probabilities
    utility = sum(prob * exp for prob, exp in zip(probabilities, expectations))
    return utility

