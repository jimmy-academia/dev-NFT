from solver import BANTERSolver
from utils import *
from printers.central_plotter import line_plot

def do_case_study():
    case_colors = ['#D62728', '#1770af']
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

    lineplot1 = output_subdir/'pri_homo_v_child.jpg'
    # directly read files to compare pricing 
    homo_price = torch.load(f'ckpt/main_exp/{nft_project_name}_BANTER_Homogeneous.pth')['pricing']
    child_price = torch.load(f'ckpt/main_exp/{nft_project_name}_BANTER_ChildProject.pth')['pricing']

    numk = len(homo_price)//10 + 1
    sorted_indices = torch.argsort(Solver.Vj, descending=True).cpu()

    if not lineplot1.exists():

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
            'colors': case_colors,
            'markers': ['P', 'X'],
            'legends': ['Homogeneous', 'ChildProject'],
            'no_xtic': True
        }
        line_plot(X, [homo_line, child_line], infos, lineplot1)

    # find instance example:
    diff = child_price[sorted_indices[:numk]] - homo_price[sorted_indices[:numk]]
    idx = diff.argmax()
    print(sorted_indices[idx], Solver.Vj[sorted_indices[idx]], homo_price[sorted_indices[idx]], child_price[sorted_indices[idx]])

    diff = child_price[sorted_indices[-numk:]] - homo_price[sorted_indices[-numk:]]
    idx = diff.argmin()
    print(sorted_indices[idx-numk], Solver.Vj[sorted_indices[idx-numk]], homo_price[sorted_indices[idx-numk]], child_price[sorted_indices[idx-numk]])


    cachefile = output_subdir/'heter.pth'
    lineplot2 = output_subdir/'pri_niche_v_eclectic.jpg'
    if not lineplot2.exists():
        print('Heterogeneous niche vs eclectic')
        ## percentage of selecting first attribute class selection vs boosting of first attribute class value

        niche_line = []
        eclectic_line = []
        X = [5*x for x in range(11)]
        multiplications = [1+x *0.01 for x in X]
        orig_vj = Solver.Vj[Solver.nft_attribute_classes == 0]

        topk = 10

        if not cachefile.exists():
            for multiple in multiplications:
                Solver.Vj[Solver.nft_attribute_classes == 0] = orig_vj * multiple
                Solver.buyer_types = torch.zeros_like(Solver.buyer_types)
                ranked_parent_nfts, ranked_parent_expectations = Solver.prepare_parent_nfts()

                nft_id_list = ranked_parent_nfts[:, :topk].flatten()
                nft_class_list = Solver.nft_attribute_classes[nft_id_list]
                niche_line.append((sum(nft_class_list == 0)/ len(nft_class_list)).item())

                Solver.buyer_types = torch.ones_like(Solver.buyer_types)
                ranked_parent_nfts, ranked_parent_expectations = Solver.prepare_parent_nfts()
                nft_id_list = ranked_parent_nfts[:, :topk].flatten()
                nft_class_list = Solver.nft_attribute_classes[nft_id_list]
                eclectic_line.append((sum(nft_class_list == 0)/ len(nft_class_list)).item())
            torch.save([niche_line, eclectic_line], cachefile)
        else:
            niche_line, eclectic_line = torch.load(cachefile)

        infos = {
            'figsize': (10,6),
            'ylabel': 'Ratio',
            'xlabel': 'Value boost (%)',
            'colors': case_colors,
            'markers': ['P', 'X'],
            'legends': ['all niche', 'all eclectic'],
        }
        line_plot(X, [niche_line, eclectic_line], infos, lineplot2)


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
