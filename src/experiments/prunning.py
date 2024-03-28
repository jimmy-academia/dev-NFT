import shutil
import time
from solver import BANTERSolver
from utils import *

from collections import defaultdict

def adjust_pruning_tests():
    args = default_args()
    args.setN = None
    args.setM = None
    args.checkpoint_dir = args.ckpt_dir / 'prunning'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    args.num_trait_div = 2
    args.num_attr_class = 2

    print('>>> (ablation.py) Ablation tests \n')
    msg = f'''
         {nft_project_names}
        x {Breeding_Types[:2]} x BANTER
        x 10 scale for prunning
        '''
    print(msg)
    for nft_project_name in nft_project_names:
        args.nft_project_name = nft_project_name
        for _breeding in Breeding_Types[:-1]:
            args.breeding_type = _breeding

            compact_results = defaultdict(list)
            new_result_path = args.checkpoint_dir / f'{nft_project_name}_{_breeding}.pth'
            if new_result_path.exists():
                print(f'{new_result_path} exists, experiment is completed.')
                continue

            cand_lim_list = [4,5,6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50] 
            #, 60, 70, 80, 90, 100
            for cand_lim in cand_lim_list:
                
                print(f'running [{nft_project_name}, {_breeding}, prunning{cand_lim}] experiment...')

                args.cand_lim = cand_lim
                Solver = BANTERSolver(args)
                start = time.time()
                Solver.solve()    
                runtime = time.time() - start
                Solver.evaluate()

                compact_results['revenue'].append(Solver.seller_revenue.item())
                compact_results['utility'].append(Solver.buyer_utilities[:, :3].sum(1).mean().item())
                compact_results['item_utility'].append(Solver.buyer_utilities[:, 0].mean().item())
                compact_results['collect_utility'].append(Solver.buyer_utilities[:, 1].mean().item())
                compact_results['breeding_utility'].append(Solver.buyer_utilities[:, 2].mean().item())
                compact_results['budget_utility'].append(Solver.buyer_utilities[:, 3].mean().item())
                compact_results['runtime'].append(runtime)
                compact_results['clenth'].append(len(Solver.ranked_parent_expectations[0]))

            torch.save(compact_results, new_result_path)
            print('____________________________________________experiment done.')

