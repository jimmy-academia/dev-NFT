import shutil
from solver import get_solver
from utils import *
import time

def run_sensitivity_tests():

    N_M_infos = loadj('ckpt/N_M_infos.json')  ## prepare with python visualize -c stats

    args = default_args()
    # args.setN = None
    args.setM = None
    args.checkpoint_dir = args.ckpt_dir / 'sensitivity'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print('>>> (sensitivity.py) Sensitivity tests \n')
    msg = f'''
         {nft_project_names[-2:-1]}
        x {Breeding_Types[2:3]}
        x {Baseline_Methods}
        x num or budget
        x 10 scales
        '''
    print(msg)
    for nft_project_name in nft_project_names[-2:-1]:
        args.nft_project_name = nft_project_name
        for _breeding in Breeding_Types[2:3]:
            args.breeding_type = _breeding
            for _method in Baseline_Methods:
                for tag in ['num', 'bud']:
                    for scale in range(1, 11):
                        result_file = args.checkpoint_dir / f'{nft_project_name}_{_method}_{_breeding}_{tag}{scale}.pth'
                        if result_file.exists():
                            print(f'{result_file} exists, experiment is completed.')
                            continue
                        if scale == 10 and tag == 'bud':
                            # original version, just copy results
                            shutil.copy(args.ckpt_dir/'main_exp'/f'{nft_project_name}_{_method}_{_breeding}.pth', result_file)
                            continue

                        print(f'running [{nft_project_name}_{_method}_{_breeding}_{tag}{scale}] experiment...')

                        args.setN = None if tag == 'bud' else int(N_M_infos[nft_project_name]['N']/10 * scale)
                        Solver = get_solver(args, _method)
                        if tag == 'bud':
                            Solver.buyer_budgets *= (scale/10)
                        
                        start = time.time()
                        Solver.solve()  
                        runtime = time.time() - start
                        Solver.evaluate() # evaluate buyer utility, seller revenue
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
                        print('____________________________________ experiment done.')

def run_runtime_tests():
    pass