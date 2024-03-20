import shutil
import time
from solver import BANTERSolver
from utils import *

def adjust_pruning_tests():
    args = default_args()
    args.setN = None
    args.setM = None
    args.checkpoint_dir = args.ckpt_dir / 'prunning'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print('>>> (ablation.py) Ablation tests \n')
    msg = f'''
         {nft_project_names}
        x {Breeding_Types[:2]} x BANTER
        x 10 scale for prunning
        '''
    print(msg)
    for nft_project_name in nft_project_names:
        args.nft_project_name = nft_project_name
        for _breeding in Breeding_Types[:2]:
            args.breeding_type = _breeding

            cand_lim_list = [1,2,3,4,5,10, 15, 20, 25, 30, 40, 50]
            for cand_lim in cand_lim_list:

                result_file = args.checkpoint_dir / f'{nft_project_name}_{_breeding}_prunning{cand_lim}.pth'
                if result_file.exists():
                    print(f'{result_file} exists, experiment is completed.')
                    continue
                
                print(f'running [{nft_project_name}, {_breeding}, prunning{cand_lim}] experiment...')

                args.cand_lim = cand_lim
                Solver = BANTERSolver(args)
                start = time.time()
                Solver.solve()    
                runtime = time.time() - start
                Solver.evaluate()
                Result = {
                    'runtime': runtime,
                    'seller_revenue': Solver.seller_revenue,
                    'buyer_utilities': Solver.buyer_utilities, 
                    'pricing': Solver.pricing, 
                    'holdings': Solver.holdings, 
                    'buyer_budgets': Solver.buyer_budgets,
                    'nft_counts': Solver.nft_counts,
                    }
                torch.save(deep_to_cpu(Result), result_file)
                print('____________________________________________experiment done.')

