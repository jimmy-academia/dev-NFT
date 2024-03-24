from solver import BANTERSolver
from utils import *
from printers.central_plotter import line_plot

def do_case_study():
    # prepare comparison files
    args = default_args()
    args.setN = None
    args.setM = None
    args.checkpoint_dir = args.ckpt_dir / 'case'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print()
    print('>>> (case.py) Case study')
    args.nft_project_name = nft_project_name = 'fatapeclub'
    args.breeding_type = 'Heterogeneous'
    Solver = BANTERSolver(args)

    output_subdir = output_dir / 'case'
    output_subdir.mkdir(parents=True, exist_ok=True)

    # directly read files to compare pricing 
    homo_price = torch.load(f'ckpt/main_exp/{nft_project_name}_BANTER_Homogeneous.pth')['pricing']
    child_price = torch.load(f'ckpt/main_exp/{nft_project_name}_BANTER_ChildProject.pth')['pricing']

    numk = len(homo_price)//10 + 1
    sorted_indices = torch.argsort(Solver.Vj, descending=True).cpu()

    print('Homogeneous vs ChildProject')
    homo_line = []
    child_line = []
    X = []
    for x, batch_ids in enumerate(make_batch_indexes(sorted_indices, numk)):
        homo_line.append(torch.mean(homo_price[batch_ids]).item())
        child_line.append(torch.mean(child_price[batch_ids]).item())
        X.append(x)
    homo_line.reverse()
    child_line.reverse()
    infos = {
        'figsize': (10,6),
        'ylabel': 'Pricing',
        'xlabel': 'Rarity',
        'colors': thecolors[::-1],
        'markers': ['P', 'X'],
        'legends': ['Homogeneous', 'ChildProject'],
        'no_xtic': True
    }
    line_plot(X, [homo_line, child_line], infos, output_subdir/'pri_homo_v_child.jpg')

    # prepare more results to compare purchase recommendation (fix pricing)
    # Solver.solve(set_pricing=pricing)

    # Utilities_list = []
    # Holdings_list = []
    # for _breeding in Breeding_Types:
    #     Solver = BANTERSolver(args)
    #     Solver.pricing = pricing_list[2] # Childproject
    #     Solver.evaluate()
    #     Utilities_list.append(Solver.buyer_utilities)
    #     Holdings_list.append(Solver.holdings)

    # '''
    # find user id whose utility
    # 1.
    # 2.
    # '''

    # '''
    # find user id whose purchase recommendation
    # 1. 
    # 2. 
    # '''
