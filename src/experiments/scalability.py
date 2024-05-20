import time
from solver import get_solver
# from solver import BANTERSolver
from utils import *

def run_scalability_tests():
    args = default_args()
    args.setN = None
    args.setM = None
    args.checkpoint_dir = args.ckpt_dir / 'newscale'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.cand_lim = 4
    args.large = True
    args.nft_project_name = 'fatapeclub'

    for _breeding in ['Heterogeneous', 'ChildProject', 'Homogeneous']:
        for _method in New_Baseline_Methods:
            for scale in range(1, 11):
                try:
                    result_file = args.checkpoint_dir / f'{args.nft_project_name}_{_breeding}_{_method}_scale{scale}.pth'

                    if result_file.exists():
                        print(f'{result_file}_____________ is finished.')
                        continue
                    print(f'running [{args.nft_project_name}_{_breeding}_{_method}_scale{scale}] experiment...')
                    args.breeding_type = _breeding
                    Solver = get_solver(args, _method)

                    ## Scale
                    thenumber = scale*10000
                    M = 1000
                    etk = 2
                    Solver.nftP.N = thenumber
                    Solver.nftP.M = M
                    Solver.nft_counts = torch.ones(M).to(args.device)
                    _attr = torch.rand(M, etk).to(args.device)
                    Solver.nft_attributes = torch.where(_attr>0.5, torch.ones_like(_attr), torch.zeros_like(_attr)).long()
                    Solver.nft_trait_counts = (Solver.nft_attributes * Solver.nft_counts.unsqueeze(1)).sum(0)
                    Solver.buyer_preferences = Solver.buyer_preferences[:, :etk].repeat(thenumber// Solver.buyer_preferences.size(0)+1, 1) [:thenumber]
                    Solver.buyer_budgets = Solver.buyer_budgets.repeat(thenumber// Solver.buyer_budgets.size(0)+1) [:thenumber]
                    if _breeding != 'None':
                        Solver.ranked_parent_nfts =  Solver.ranked_parent_nfts[:, :etk, :].repeat(thenumber//Solver.ranked_parent_nfts.size(0) +1, 1, 1)[:thenumber]
                        Solver.ranked_parent_expectations = Solver.ranked_parent_expectations[:, :etk].repeat(thenumber//Solver.ranked_parent_expectations.size(0) +1, 1)[:thenumber]

                    Solver.Vj = Solver.Vj.repeat(thenumber// Solver.Vj.size(0)+1) [:M]
                    Solver.Uij = torch.rand(thenumber, M).to(args.device)/10

                    if _method == 'HetRecSys':
                        Solver.do_preparations()
                        # Solver.embed_dim = 2

                    start = time.time()
                    Solver.solve()  
                    runtime = time.time() - start
                    Solver.evaluate() # evaluate buyer utility, seller revenue

                    dd = 3 if _method == 'BANTER' else 2
                    Result = {
                        'runtime': runtime,
                        'seller_revenue': Solver.seller_revenue.item(),
                        'buyer_utilities': Solver.buyer_utilities[:, :dd].sum(1).mean().item()
                        }
                    
                    torch.save(Result, result_file)
                    print('____________________________________________experiment done.')
                except:
                    print(f'[{args.nft_project_name}_{_breeding}_{_method}_scale{scale}] experiment cannot run XXXXXXX')