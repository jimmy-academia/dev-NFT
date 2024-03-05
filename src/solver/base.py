from utils import *


class NFTProject:
    def __init__(self, nft_project_data, setN, setM):
        self.N = setN if setN is not None else len(nft_project_data['buyer_budgets'])
        self.M = setM is setM is not None else len(nft_project_data['asset_traits'])
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
                item_vec.append(options.index(choice))
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
        nft_project_data = loadj(f'../NFT_data/clean/{nft_project_name}')
        self.nftP = NFTProject(nft_project_data, args.setN, args.setM)

    def solve(self):
        '''
        yields:
        self.pricing
        self.holdings (recommendation of purchase amount to each buyer)
        '''
        raise NotImplementedError

    def evaluate(self):
        # evaluate buyer utility, seller revenue
        
        # batch process buyer utility

