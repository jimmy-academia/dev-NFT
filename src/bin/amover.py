

from utils import *

nft_project_name = 'fatapeclub'
args = default_args()
args.setM = None
args.checkpoint_dir = args.ckpt_dir / 'sensitivity'
_breeding = Breeding_Types[2]

for tag in ['num', 'bud']:

    compact_results = {'revenue':[], 'utility':[], 'runtime':[]}
    new_result_path = args.checkpoint_dir / f'{nft_project_name}_{_breeding}_{tag}.pth'
    for _method in Baseline_Methods:
        method_revenue = []
        method_utility = []
        method_runtime = []
        for scale in range(1, 11):
            result_file = args.checkpoint_dir / f'{nft_project_name}_{_method}_{_breeding}_{tag}{scale}.pth'
            print('doing ', result_file)
            method_revenue.append(torch.load(result_file)['seller_revenue'].item())
            method_utility.append(torch.load(result_file)['seller_revenue'].item())

            if _method == 'BANTER':
                method_utility.append(torch.load(result_file)['buyer_utilities'][:, :3].sum(1).mean().item())  ## average buyer utility
            else:
                method_utility.append(torch.load(result_file)['buyer_utilities'][:, :2].sum(1).mean().item())  ## no breeding 
            method_runtime.append(torch.load(result_file)['runtime'])

            result_file.unlink()

        compact_results['revenue'].append(method_revenue)
        compact_results['utility'].append(method_utility)
        compact_results['runtime'].append(method_runtime)

    torch.save(compact_results, new_result_path)
    print('saved to', new_result_path)