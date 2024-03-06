import random
from utils import *

class NFTProject:
    def __init__(self, nft_project_data, setN, setM):
        self.N = setN if setN is not None else len(nft_project_data['buyer_budgets'])
        self.M = setM if setM is not None else len(nft_project_data['asset_traits'])
        self.trait_dict = nft_project_data['trait_system']
        self.item_vec_list = self.item_trait_vectorize(nft_project_data['asset_traits'])
        self.trait_counts = self.compute_trait_stats() 
        self.item_count = nft_project_data['item_counts']
        self.user_preferences = self.get_user_preferences(nft_project_data['buyer_assets_ids'], nft_project_data['asset_traits'])
        self.user_budgets = self.get_user_budgets(nft_project_data['buyer_budgets'])

    def item_trait_vectorize(self, asset_traits):
        item_vec_list = []
        for item in asset_traits[:self.M]:
            item_vec = []
            for (trait, options), choice in zip(self.trait_dict.items(), item):
                choice = 'none' if choice == 'None' else choice
                try:
                    item_vec.append(options.index(choice))
                except:
                    item_vec.append(random.randint(0, len(options)-1))
            item_vec_list.append(item_vec)
        return item_vec_list

    def compute_trait_stats(self):
        trait_counts = []
        for trait, options in self.trait_dict.items():
            trait_counts.append([1 for __ in options])
        for item_vec in self.item_vec_list:
            for t, choice in enumerate(item_vec):
                trait_counts[t][choice] += 1
        return trait_counts

    def get_user_preferences(self, buyer_assets_ids, asset_traits):
        user_preferences = []
        for __ in range(self.N):
            user_pref = []
            for trait, options in self.trait_dict.items():
                user_pref.append([0 for option in options])
            user_preferences.append(user_pref)
        for i in range(self.N):
            attr_list = [asset_traits[aid] for aid in buyer_assets_ids[i]]
            for attr in attr_list:
                for t, (trait, options) in enumerate(self.trait_dict.items()):
                    try:
                        user_preferences[i][t][options.index(attr[t])] +=1
                    except:
                        user_preferences[i][t][random.randint(0, len(options)-1)] +=1
        return user_preferences

    def get_user_budgets(self, buyer_budgets):
        print('do budget normalization??!!!')
        return buyer_budgets