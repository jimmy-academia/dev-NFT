import random
from utils import *

class NFTProject:
    def __init__(self, nft_project_data, setN, setM, nft_project_name):
        self.N = setN 
        self.M = setM 
        self.nft_project_name = nft_project_name
        self.trait_dict = nft_project_data['trait_system']
        self.item_attributes, self.user_preferences, self.user_budgets, self.item_counts = self.numericalize(nft_project_data)
        # self.trait_counts = self.compute_trait_stats() 

    def numericalize(self, nft_project_data):
        asset_traits, buyer_assets_ids, buyer_budgets, item_counts = nft_project_data['asset_traits'], nft_project_data['buyer_assets_ids'], nft_project_data['buyer_budgets'], nft_project_data['item_counts']

        max_aid_len = max([len(x) for x in buyer_assets_ids])
        min_aid_len = min_purchase[nft_project_names.index(self.nft_project_name)] # filter buyers with min purchases
        user_preferences = []
        buyer_num = self.N if self.N is not None else len(buyer_assets_ids)
        aid_set = set()
        bid_list = []
        for i in range(buyer_num):
            if len(buyer_assets_ids[i]) <= min_aid_len:
                continue
            aid_set |= set(buyer_assets_ids[i])
            user_prefs = self.trait2label_vec([asset_traits[aid] for aid in buyer_assets_ids[i]])
            # pad until max length, then change shape
            user_prefs = user_prefs + [user_prefs[-1]] * (max_aid_len - len(user_prefs))
            user_prefs = [list(x) for x in zip(*user_prefs)]
            user_preferences.append(user_prefs)
            bid_list.append(i)

        self.N = len(bid_list)
        self.M = len(aid_set) #filter items purchased by buyers
        return self.trait2label_vec([asset_traits[i] for i in aid_set]), user_preferences, [buyer_budgets[bi] for bi in bid_list], [item_counts[i]+1 for i in aid_set]

    def trait2label_vec(self, asset_traits):
        item_vec_list = []
        for item in asset_traits:
            item_vec = []
            for (trait, options), choice in zip(self.trait_dict.items(), item):
                choice = 'none' if choice == 'None' else choice
                try:
                    item_vec.append(options.index(choice))
                except:
                    item_vec.append(random.randint(0, len(options)-1))
            item_vec_list.append(item_vec)
        return item_vec_list

    # def compute_trait_stats(self):
    #     trait_counts = []
    #     for trait, options in self.trait_dict.items():
    #         trait_counts.append([1 for __ in options])
    #     for item_vec in self.item_attributes:
    #         for t, choice in enumerate(item_vec):
    #             trait_counts[t][choice] += 1
    #     return trait_counts

    