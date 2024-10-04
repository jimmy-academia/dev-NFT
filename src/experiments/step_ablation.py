import sys 
sys.path.append('.')
import shutil
import time
from solver import BANTERSolver
from utils import *

def run_ablation_tests():
    args = default_args()
    args.setN = None
    args.setM = None
    # args.checkpoint_dir = args.ckpt_dir / 'step_ablation'
    # args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print()
    print('>>> (ablation.py) Ablation tests')
    msg = f'''
         {nft_project_names[3:4]}
        x {Breeding_Types[0:1]} x BANTER
        x 3 ablation (0: original, 1:no init, 2:only init)
        '''
    print(msg)
    for nft_project_name in nft_project_names[3:4]:
        args.nft_project_name = nft_project_name
        for _breeding in Breeding_Types[0:1]:
            args.breeding_type = _breeding
            print(f'running [{nft_project_name}, {_breeding}] experiment...')
            for ablation_id in range(3):

                print(f'===== Ablation {ablation_id} =====')
                
                args.ablation_id = ablation_id
                Solver = BANTERSolver(args)
                Solver.solve() 

                revenues = []
                for price in Solver.pricing_list:
                    Solver.pricing = price
                    Solver.evaluate()
                    revenues.append(Solver.seller_revenue.cpu().item())
                
                print(revenues)

                # torch.save(deep_to_cpu(Result), result_file)
                print('______________________________ experiment done.')



def run_schedule_tests():
    args = default_args()
    args.setN = None
    args.setM = None
    args.checkpoint_dir = args.ckpt_dir / 'step_schedule_ablation'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print()
    print('>>> (ablation.py) Schedule tests')
    msg = f'''
         {nft_project_names[3:4]}
        x {Breeding_Types[0:1]} x BANTER
        x 3 schedule (0: original, 1:fixed, 2:none)
        '''
    print(msg)
    for nft_project_name in nft_project_names[3:4]:
        args.nft_project_name = nft_project_name
        for _breeding in Breeding_Types[0:1]:
            args.breeding_type = _breeding
            print(f'running [{nft_project_name}, {_breeding}] experiment...')

            for schedule_id in range(1,3):

                print(f'===== Schedule {schedule_id} =====')

                args.schedule_id = schedule_id
                Solver = BANTERSolver(args)
                Solver.solve()    
                revenues = []
                for price in Solver.pricing_list:
                    Solver.pricing = price
                    Solver.evaluate()
                    revenues.append(Solver.seller_revenue.cpu().item())
                
                print(revenues)
                print('______________________________ experiment done.')

if __name__ == '__main__':
    # run_ablation_tests()
    run_schedule_tests()