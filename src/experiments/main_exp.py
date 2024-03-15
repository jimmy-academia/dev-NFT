from solver import get_solver
from utils import *

'''
Main experiment for seller revenue: 5+1 methods x 3 breeding cases
'''

def run_revenue_experiments():
    args = default_args()
    args.setN = None
    args.setM = None
    args.checkpoint_dir = args.ckpt_dir / 'main_exp'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print('>>> (main_exp.py) Main experiments for seller revenue: \n')
    msg = f'''
         {nft_project_names[:4]} ... 
         ... {nft_project_names[4:]} 
        x {Baseline_Methods} 
        x {Breeding_Types}'''
    print(msg)
    for nft_project_name in nft_project_names:
        args.nft_project_name = nft_project_name
        for _breeding in Breeding_Types:
            for _method in Baseline_Methods:
                result_file = args.checkpoint_dir / f'{nft_project_name}_{_method}_{_breeding}.pth'
                if result_file.exists():
                    print(f'{result_file} exists, experiment is completed.')
                else:
                    print(f'running [{nft_project_name}, {_method}, {_breeding}] experiment...')
                    args.breeding_type = _breeding
                    Solver = get_solver(args, _method)
                    Solver.solve() # recommend pricing and NFT purchase for each user
                    Solver.evaluate() # evaluate buyer utility, seller revenue
                    Result = {
                        'seller_revenue': Solver.seller_revenue,
                        'buyer_utilities': Solver.buyer_utilities, 
                        'pricing': Solver.pricing, 
                        'holdings': Solver.holdings, 
                        'buyer_budgets': Solver.buyer_budgets,
                        'nft_counts': Solver.nft_counts,
                    }
                    torch.save(deep_to_cpu(Result), result_file)
                    print(f'[{nft_project_name}, {_method}, {_breeding}] experiment done.')

                    
def run_buyer_utility():
    args = default_args()
    args.setN = None
    args.setM = None
    args.checkpoint_dir = args.ckpt_dir / 'buyer_util'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print('>>> (main_exp.py) Main experiments for buyer utilities: \n')

    msg = f'''
         {nft_project_names[:4]} ... 
         ... {nft_project_names[4:]} 
        x {Baseline_Methods} 
        x {Breeding_Types}'''
    print(msg)

    for nft_project_name in nft_project_names:
        args.nft_project_name = nft_project_name
    
        for _breeding in Breeding_Types:
            # get pricing from BANTER...

            for _method in Baseline_Methods:
                result_file = args.checkpoint_dir / f'{nft_project_name}_{_method}_{_breeding}.pth'
                if result_file.exists():
                    print(f'{result_file} exists, experiment is completed.')
                else:
                    print(f'running [{nft_project_name}, {_method}, {_breeding}] experiment...')
                