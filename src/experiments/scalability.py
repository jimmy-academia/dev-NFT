import time
from solver import BANTERSolver
from utils import *


def run_scalability_tests():
    N_M_infos = loadj('ckpt/N_M_infos.json') 
    args = default_args()
    args.setN = None
    # args.setM = None
    args.checkpoint_dir = args.ckpt_dir / 'duplicate'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for nft_project_name in nft_project_names[3:4]:
        args.nft_project_name = nft_project_name
        for scale in range(2, 11):
            for _breeding in Breeding_Types:
                args.breeding_type = _breeding
                args.setN = N_M_infos[nft_project_name]['N'] * scale
                args.setM = N_M_infos[nft_project_name]['M'] * scale

                print(f'running [{nft_project_name}_BANTER_{_breeding}_duplicate{scale}] experiment... N:{args.setN} M:{args.setM}')

                Solver = BANTERSolver(args)
                start = time.time()
                Solver.solve()  
                runtime = time.time() - start
                Solver.evaluate() # evaluate buyer utility, seller revenue

                Result = {
                    'runtime': runtime,
                    'seller_revenue': Solver.seller_revenue.item(),
                    'buyer_utilities': Solver.buyer_utilities[:, :3].sum(1).mean().item()
                    }
                
                result_file = args.checkpoint_dir / f'{nft_project_name}_{_breeding}_duplicate{scale}.pth'
                torch.save(Result, result_file)
                print('____________________________________________experiment done.')
