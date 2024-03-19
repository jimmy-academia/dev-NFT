import shutil

from solver import BANTERSolver
from utils import *

def run_ablation_tests():
    args = default_args()
    args.setN = None
    args.setM = None
    args.checkpoint_dir = args.ckpt_dir / 'ablation'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print()
    print('>>> (ablation.py) Ablation tests')
    msg = f'''
         {nft_project_names[1:]}
        x {Breeding_Types} x BANTER
        x 3 ablation (0: original, 1:no init, 2:only init)
        '''
    print(msg)
    for nft_project_name in nft_project_names[1:]:
        args.nft_project_name = nft_project_name
        for _breeding in Breeding_Types:
            args.breeding_type = _breeding
            for ablation_id in range(3):
                result_file = args.checkpoint_dir / f'{nft_project_name}_{_breeding}_ablation{ablation_id}.pth'
                if result_file.exists():
                    print(f'{result_file} exists, experiment is completed.')
                    continue
                if ablation_id == 0:
                    # original version, just copy results
                    shutil.copy(args.ckpt_dir/'main_exp'/f'{nft_project_name}_BANTER_{_breeding}.pth', result_file)
                    continue

                print(f'running [{nft_project_name}, {_breeding}, ablation{ablation_id}] experiment...')

                args.ablation_id = ablation_id
                Solver = BANTERSolver(args)
                Solver.solve()    
                Solver.evaluate()
                Result = {
                        'seller_revenue': Solver.seller_revenue,
                        'buyer_utilities': Solver.buyer_utilities, 
                        'pricing': Solver.pricing, 
                        'holdings': Solver.holdings, 
                        'buyer_budgets': Solver.buyer_budgets,
                        'nft_counts': Solver.nft_counts,
                    }
                torch.save(deep_to_cpu(Result), result_file)
                print(f'[{nft_project_name}, {_breeding}, ablation{ablation_id}] experiment done.')

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
            for scale in range(1, 11):
                result_file = args.checkpoint_dir / f'{nft_project_name}_{_breeding}_prunning{scale}.pth'
                if result_file.exists():
                    print(f'{result_file} exists, experiment is completed.')
                    continue
                if scale == 10:
                    # original version, just copy results
                    shutil.copy(args.ckpt_dir/'main_exp'/f'{nft_project_name}_BANTER_{  _breeding}.pth', result_file)
                    continue

                print(f'running [{nft_project_name}, {_breeding}, prunning{scale}] experiment...')

                args.cand_lim = args.cand_lim//10 * scale
                Solver = BANTERSolver(args)
                Solver.solve()    
                Solver.evaluate()
                Result = {
                        'seller_revenue': Solver.seller_revenue,
                        'buyer_utilities': Solver.buyer_utilities, 
                        'pricing': Solver.pricing, 
                        'holdings': Solver.holdings, 
                        'buyer_budgets': Solver.buyer_budgets,
                        'nft_counts': Solver.nft_counts,
                    }
                torch.save(deep_to_cpu(Result), result_file)
                print(f'[{nft_project_name}, {_breeding}, prunning{scale}] experiment done.')

