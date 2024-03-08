import torch
import math
from tqdm import tqdm
from itertools import product

from utils import *
from .project import NFTProject

class BaseSolver:
    def __init__(self, args):
        self.args = args
        self.breeding_type = args.breeding_type

        # cache system: same setN, setM, nft_project_name yeilds same results.
        cache_dir = args.ckpt_dir / f'cache_{args.nft_project_name}_N_{args.setN}_M_{args.setM}'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_nft_project_file = cache_dir / f'project.pth'
        if cache_nft_project_file.exists():
            self.nftP = torch.load(cache_nft_project_file)
        else:
            nft_project_data = loadj(f'../NFT_data/clean/{args.nft_project_name}.json')
            self.nftP = NFTProject(nft_project_data, args.setN, args.setM, args.nft_project_name)
            torch.save(self.nftP, cache_nft_project_file)
        
        self.child_population_factor = 1
        self.num_traits = len(self.nftP.trait_dict)
        self.num_selections_list = [len(options) for (_, options) in self.nftP.trait_dict.items()]
        self.max_selections = max(self.num_selections_list)
        
        self.prepare_tensors()
        self.alpha = None
        self.Vj = self.calc_objective_valuations(self.nft_attributes)
        self.Uij = torch.matmul(self.buyer_preferences, self.nft_attributes.T.float())
        if self.breeding_type == 'Heterogeneous':
            cache_heter_labels_path = cache_dir / f'heter_files_{args.num_trait_div}_{args.num_attr_class}_{self.nftP.N}_{self.nftP.M}.pth'
            if cache_heter_labels_path.exists():
                self.nft_trait_divisions, self.nft_attribute_classes, self.buyer_types = torch_cleanload(cache_heter_labels_path, self.args.device)
            else:
                self.nft_trait_divisions =  torch.randint(args.num_trait_div, (self.nftP.M,)).to(self.args.device)
                self.nft_attribute_classes =  torch.randint(args.num_attr_class, (self.nftP.M,)).to(self.args.device)
                self.buyer_types = torch.randint(2, (self.nftP.N,)).to(self.args.device)
                torch_cleansave((self.nft_trait_divisions, self.nft_attribute_classes, self.buyer_types), cache_heter_labels_path)

        cache_parents_path = cache_dir / f'parents_{args.breeding_type}_{args.num_child_sample}_{args.mutation_rate}.pth'
        if cache_parents_path.exists():
            self.ranked_parent_nfts, self.ranked_parent_expectations = torch_cleanload(cache_parents_path, self.args.device)
        else:
            self.ranked_parent_nfts, self.ranked_parent_expectations = self.prepare_parent_nfts()
            torch_cleansave((self.ranked_parent_nfts, self.ranked_parent_expectations), cache_parents_path)

    def tensorize(self, label_vec, yield_mask=False):
        # tensorize
        label_vec_tensor = torch.LongTensor(label_vec).to(self.args.device)
        label_vec_tensor = label_vec_tensor.unsqueeze(2) if len(label_vec_tensor.shape) == 2 else label_vec_tensor
        binary = torch.zeros(label_vec_tensor.shape[0], self.num_traits, self.max_selections).to(self.args.device)
        binary.scatter_(2, label_vec_tensor, 1)
        binary = binary.view(label_vec_tensor.shape[0], -1)

        if not yield_mask:
            return binary

        mask = torch.zeros(label_vec_tensor.shape[0], self.num_traits, self.max_selections).to(self.args.device).bool()
        index_tensor = torch.arange(mask.size(2)).unsqueeze(0).unsqueeze(0)
        num_selections_tensor = torch.tensor(self.num_selections_list)
        num_selections_expanded = num_selections_tensor.unsqueeze(-1).expand(-1, mask.size(2))
        condition = index_tensor >= num_selections_expanded
        mask[condition.expand_as(mask)] = 1
        mask = mask.view(label_vec_tensor.shape[0], -1)

        return binary, mask

    def prepare_tensors(self):
        self.nft_counts = torch.LongTensor(self.nftP.item_counts).to(self.args.device)
        self.nft_attributes = self.tensorize(self.nftP.item_attributes).long()
        self.nft_trait_counts = (self.nft_attributes * self.nft_counts.unsqueeze(1)).sum(0)

        buyer_preferences, buyer_preferences_mask = self.tensorize(self.nftP.user_preferences, True)
        buyer_preferences = buyer_preferences.masked_fill(buyer_preferences_mask, float('-inf'))
        self.buyer_preferences = torch.softmax(buyer_preferences, dim=1)
        self.preference_mask = ~buyer_preferences_mask[0]
        self.nft_trait_counts = torch.where(self.preference_mask * (self.nft_trait_counts==0), 1, self.nft_trait_counts)
        assert (self.preference_mask * (self.nft_trait_counts==0)).sum() == 0

        buyer_budgets = torch.Tensor(self.nftP.user_budgets).to(self.args.device)
        # scale to  [10, 100]
        buyer_budgets.clamp_(min=0)  # Ensure that the minimum value is 0
        buyer_budgets.sub_(buyer_budgets.min()).div_(buyer_budgets.max() - buyer_budgets.min()).mul_(90).add_(10)
        self.buyer_budgets = buyer_budgets

    def calc_objective_valuations(self, nft_attributes):
        attr_rarity_prod = nft_attributes * self.nft_trait_counts
        attr_rarity_prod = attr_rarity_prod[attr_rarity_prod!= 0].view(-1, self.num_traits)
        objective_values = torch.log(sum(self.nft_counts)/attr_rarity_prod).sum(1)
        if self.alpha is None:
            self.alpha = sum(self.buyer_budgets) / sum(objective_values)
        return objective_values * self.alpha

    def gen_rand_nft(self, batch_shape):
        indices = torch.stack([torch.randint(0, _seg, (math.prod(batch_shape),)) + i*self.max_selections for i, _seg in enumerate(self.num_selections_list)]).T.to(self.args.device)
        nft_attributes = torch.zeros(math.prod(batch_shape), self.num_traits *self.max_selections).long().to(self.args.device)
        nft_attributes.scatter_(1, indices, 1)
        nft_attributes = nft_attributes.view(*batch_shape, -1)
        return nft_attributes

    def batch_pairing(self, batch_candidates):
        """
        pairing for Homogeneous and ChildProject
        """
        idx = torch.combinations(torch.arange(batch_candidates.size(1), device=batch_candidates.device), 2)
        idx = idx.unsqueeze(0).repeat(batch_candidates.size(0), 1, 1)
        # Gather combinations from each vector in the batch_candidates
        combos = torch.gather(batch_candidates.unsqueeze(1).repeat(1, idx.size(1), 1), 2, idx)
        return combos

    def batch_assembling(self, trait_divisions, batch_candidates):
        '''
        assembling for Heterogeneous
        '''
        combos = []
        for candidates in tqdm(batch_candidates, ncols=88, desc='assembling', leave=False):
            labels_vector = trait_divisions[candidates]
            unique_labels = labels_vector.unique(sorted=True)
            indices_list = [(labels_vector == label).nonzero(as_tuple=True)[0] for label in unique_labels]
            rank_list = [torch.arange(len(indices)) for indices in indices_list]
            parent_sets = torch.cartesian_prod(*indices_list)
            combined_ranks = torch.cartesian_prod(*rank_list).sum(-1)
            parent_sets = parent_sets[combined_ranks.argsort()]
            combos.append(parent_sets)
        min_len = min(len(combo) for combo in combos)        
        combos = torch.stack([combo[:min_len] for combo in combos])
        return combos

    def prepare_parent_nfts(self):
        print('store and reuse!!!')
        '''
        estimate expectation value for each parent pair
        sort parent nfts by expectation value
        yields self.ranked_parent_nfts, self.ranked_paraent_expectations
        '''
        if self.breeding_type == 'None':
            return
        
        if self.breeding_type == 'Heterogeneous':
            niche_buyer_ids, eclectic_buyer_ids = [torch.where(self.buyer_types==i)[0] for i in range(2)]
            parent_nft_candidates = (self.Uij * self.Vj).topk(self.args.cand_lim)[1]
            parent_nft_sets = self.batch_assembling(self.nft_trait_divisions, parent_nft_candidates)
            # niche
            ## count number of same attribute class
            niche_sets = parent_nft_sets[niche_buyer_ids]
            labeled_sets = self.nft_attribute_classes[niche_sets]
            majority_label = labeled_sets.mode(dim=-1)[0].unsqueeze(-1).expand_as(labeled_sets)
            same_class_count = (labeled_sets == majority_label).sum(-1)
            del majority_label

            # eclectic
            eclectic_sets = parent_nft_sets[eclectic_buyer_ids]
            labeled_sets = self.nft_attribute_classes[eclectic_sets]
            hash_map = torch.cartesian_prod(*[torch.arange(self.args.num_attr_class)]*self.args.num_trait_div).to(self.args.device)
            hash_map = hash_map.sort()[0].unique(dim=0)
            num_unique = torch.LongTensor([len(hash.unique()) for hash in hash_map]).to(self.args.device)
            
            query = labeled_sets.view(-1, 3).sort(-1)[0]
            matching_indices = torch.full((query.size(0),), -1, dtype=torch.long, device=query.device)  # Fill with -1 to indicate no match by default
            for i, hash in enumerate(hash_map):
                matches = (query == hash).all(dim=1)
                matching_indices[matches] = i
            div_class_count = num_unique[matching_indices].view(labeled_sets.shape[:2])

            del hash_map, query, matching_indices, num_unique
            ## interleave same_class_count and div_class_count to get the final expectation together
            _class_count = torch.zeros(parent_nft_sets.shape[:2], dtype=torch.long, device=self.args.device)
            _class_count[niche_buyer_ids] = same_class_count
            _class_count[eclectic_buyer_ids] = div_class_count
            breeding_expectation_values = (self.Uij * self.Vj).unsqueeze(1).expand(-1, parent_nft_sets.size(1), -1).gather(2, parent_nft_sets).sum(-1)
            breeding_expectation_values *= _class_count
            del same_class_count, div_class_count, _class_count
            torch.cuda.empty_cache()  # Release CUDA memory
        else:
            parent_nft_candidate = (self.Uij * self.Vj).topk(self.args.cand_lim)[1]
            chunk_size = 32
            parent_nft_sets = []
            breeding_expectation_values = []
            for batch_buyer_idx in batch_indexes(parent_nft_candidate.size(0), chunk_size):
                batch_parent_nft_candidate = parent_nft_candidate[batch_buyer_idx]
                batch_parent_nft_sets = self.batch_pairing(batch_parent_nft_candidate)
                batch_expectation_values = torch.zeros(batch_parent_nft_sets.size()[:2]).to(self.args.device)
                parent_nft_attributes = self.nft_attributes[batch_parent_nft_sets]

                for __ in tqdm(range(self.args.num_child_sample), ncols=88, desc='sampling child NFT', leave=False):
                    _shape = (*parent_nft_attributes.shape[:2], self.num_traits)
                    if self.breeding_type == 'Homogeneous':
                        trait_inherit_mask = torch.randint(0, 2, _shape).to(self.args.device)
                        inheritance_mask = trait_inherit_mask.repeat_interleave(self.max_selections, dim=2)
                        child_attribute = inheritance_mask * parent_nft_attributes[:, :, 0, :] + \
                            (1 - inheritance_mask) * parent_nft_attributes[:, :, 1, :]
                    else:
                        r = self.args.mutation_rate
                        trait_inherit_mask = torch.multinomial(torch.Tensor([(1-r)/2, (1-r)/2, r]), math.prod(_shape), replacement=True)
                        trait_inherit_mask = trait_inherit_mask.view(_shape).to(self.args.device)
                        inheritance_mask = trait_inherit_mask.repeat_interleave(self.max_selections, dim=2)
                        child_attribute = torch.where(inheritance_mask==0, 1, 0) * parent_nft_attributes[:, :, 0, :] + \
                            torch.where(inheritance_mask==1, 1, 0) * parent_nft_attributes[:, :, 1, :] + \
                            torch.where(inheritance_mask==2, 1, 0) * self.gen_rand_nft(_shape[:-1])
                        # assert all((child_attribute*self.preference_mask).view(-1, 288).sum(-1) == 6)
                    Uj = (self.buyer_preferences[batch_buyer_idx].unsqueeze(1)  * child_attribute).sum(-1)
                    Vj = self.calc_objective_valuations(child_attribute.view(-1, child_attribute.size(-1))).view(Uj.shape)
                    batch_expectation_values += (Vj * Uj).squeeze(-1)
                batch_expectation_values /= self.args.num_child_sample

                parent_nft_sets.append(batch_parent_nft_sets)
                breeding_expectation_values.append(batch_expectation_values)

            # sort parent_nft_sets and parent_nft_expectations by parent_nft_expectations
            parent_nft_sets = torch.cat(parent_nft_sets)
            breeding_expectation_values = torch.cat(breeding_expectation_values)
        
        sorted_indices = breeding_expectation_values.argsort(descending=True)
        ranked_parent_nfts = torch.gather(parent_nft_sets, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, parent_nft_sets.size(-1)))
        ranked_parent_expectations = torch.gather(breeding_expectation_values, 1, sorted_indices)

        return ranked_parent_nfts, ranked_parent_expectations

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
        self.buyer_utilities = []
        for batch_users in batch_indexes(self.nftP.N, 100):
            self.buyer_utilities.append(self.calculate_buyer_utilities(
                batch_users,
                self.holdings[batch_users],
                self.buyer_budgets[batch_users],
                self.pricing,   
            ))
        self.buyer_utilities = torch.cat(self.buyer_utilities, dim=0)

        # seller revenue
        self.seller_revenue = (self.pricing * self.holdings).sum(0)

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
        selection_mask = torch.where(cum_prob > self.args.breeding_topk, probability, torch.zeros_like(probability))

        if self.breeding_type == 'homogeneous':
            # calculate frequencies based on selection_mask 
            parent_attr_freq = torch.stack([self.nft_attributes[parents[..., p]]*selection_mask for p in range(parents.shape[-1])]).sum(0)
            # adjust expectation
            child_population_factor = (torch.stack([self.nft_attributes[parents[..., p]] for p in range(parents.shape[-1])]) * parent_attr_freq).sum(0)
            expectation = expectation / (1+ child_population_factor/10)
            print('todo examine!!')

        U_breeding = (selection_mask * expectation).sum(1)
        return U_breeding
                


