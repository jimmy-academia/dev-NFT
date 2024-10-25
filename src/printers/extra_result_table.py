import sys 
sys.path.append('.')
import torch
from pathlib import Path
from utils import *

def print_extra():
    print()
    out_sub_dir = 'main_exp/' 
    (output_dir/out_sub_dir).mkdir(parents=True, exist_ok=True)
    print(['revenue', 'buyer_utility', 'runtime'])
    for nft_project_name in nft_project_names:
        for _breeding in Breeding_Types:
            print()
            print(f'======== {_breeding} ========')
            for _method in Extra_Baseline_Methods+['BANTER']:
                project_values = []
                for tag, ylabel in zip(['revenue', 'buyer_utility', 'runtime'], ['Revenue', 'Avg. Utility', 'Runtime (s)']):
                    filepth = Path(f'ckpt/main_exp/{nft_project_name}_{_method}_{_breeding}.pth')
                    if filepth.exists():
                        if tag == 'revenue':
                            project_values.append(torch.load(filepth, weights_only=True)['seller_revenue'].item())
                        elif tag == 'buyer_utility':
                            if _method == 'BANTER':
                                project_values.append(torch.load(filepth, weights_only=True)['buyer_utilities'][:, :3].sum(1).mean().item())  ## average buyer utility
                            else:
                                project_values.append(torch.load(filepth, weights_only=True)['buyer_utilities'][:, :2].sum(1).mean().item())  ## no breeding recommendation
                        elif tag == 'runtime':
                            project_values.append(torch.load(filepth, weights_only=True)['runtime'])
                            # xticks = ['Hetero-\ngeneous', 'Homo-\ngeneous', 'Child\nProject', 'No\nBreeding']
                        else:
                            project_values.append(0)

                project_values = [round(num, 3) for num in project_values] 
                print(f'{_method}:', project_values)


if __name__ == '__main__':
    print_extra()
    print()