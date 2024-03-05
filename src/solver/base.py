from utils import *


class NFTProject:
    def __init__(self, nft_project_data, setN, setM):
        self.setN = setN
        self.setM = setM
        self.trait_dict = nft_project_data['trait_system']
        self.item_trait_vectorize(nft_project_data['asset_traits'])
        self.compute_trait_stats() 
        self.item_count = nft_project_data['item_counts']
        self.get_user_preferences(nft_project_data['buyer_assets_ids'], nft_project_data['asset_traits'])
        self.get_user_budgets(nft_project_data['buyer_budgets'])

    def item_trait_vectorize(self, asset_traits):
        self.item_vec_list = []
        for item in asset_traits[:self.setM]:
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
        N = self.setN if self.setN is not None else len(buyer_assets_ids)
        for __ in range(N):
            user_pref = []
            for trait, options in self.trait_dict.items():
                user_pref.append([0 for option in options])
            self.user_preferences.append(user_pref)
        for i in range(N):
            attr_list = asset_traits[buyer_assets_ids[i]]
            for attr in attr_list:
                for t, (trait, options) in enumerate(self.trait_dict.items()):
                    self.user_preferences[i][t][options.index(attr[t])] +=1

    def get_user_budgets(self, buyer_budgets):
        print('do normalization!!!')
        pass
        self.user_budgets = 111
        # x = self.data['buyer_budgets']
        # return [(b - min(x)) /(max(x) - min(x))*90 + 10 for b in x]


class BaseSolver:
    def __init__(self, args):
        nft_project_data = loadj(f'../NFT_data/clean/{nft_project_name}')
        self.nftP = NFTProject(nft_project_data, args.setN, args.setM)

    def solve(self):
        raise NotImplementedError

    def 