import random
from collections import defaultdict
from typing import Tuple, Dict

from utils import *

def prepare_nft_data():
    '''
    load and prepare nft data into [nft_project_data] and save into files
    '''
    data_dir = Path('../NFT_data')
    clean_dir, tiny_dir = map(lambda x: data_dir / x, ['clean', 'tiny'])
    clean_dir.mkdir(parents=True, exist_ok=True)
    tiny_dir.mkdir(parents=True, exist_ok=True)

    for Project_Name in NFT_Projects:
        project_name = ''.join(Project_Name.split()).lower()
        
        data_files = map(lambda x: Path(f'../NFT_data/{x}/{project_name}.json'), ['trades', 'NFT_attributes', 'trait_system'])

        project_file = clean_dir/f'{project_name}.json'
        tiny_project_file = tiny_dir/f'{project_name}.json'

        if not project_file.exists(): 
            nft_project_data = load_nft_project(project_name, clean_dir, data_files)
            dumpj(nft_project_data, project_file)
        if not tiny_project_file.exists(): 
            nft_project_data = load_nft_project(project_name, tiny_dir, data_files, subset=True)
            dumpj(nft_project_data, tiny_project_file)

def load_nft_project(project_name, tiny_dir, data_files, subset=False):

    '''
    return:
    nft_project_data: dict_keys(['trait_system', 'asset_traits', 'item_counts', 
            'buyer_budgets', 'buyer_assets', 'buyer_assets_ids'])
    
    inputs 
    - trade_info: dict, 'note', 'result'
    - NFT_info: list of dict
    - trait_system: dict
    '''
    trade_info, NFT_info, trait_system = map(loadj, data_files)
    NFT_info, trait_system = filter_nft_attributes(project_name, NFT_info, trait_system)
    nft_project_data = process_nft_trades(trade_info, NFT_info, trait_system)
    if subset:
        print('do subset things!')
    return nft_project_data

def filter_nft_attributes(project_name: str, NFT_info: list, trait_system: dict) -> tuple:
    """
    Processes NFT attributes and adjusts the trait system for missing data based on the project name.

    Parameters:
    - project_name (str): Name of the NFT project.
    - NFT_info (list of dict): List containing NFT instances, each listing the token_id and its attributes.
    - trait_system (dict): Trait system of the NFT project.

    Returns:
    - Tuple of updated NFT_info and trait_system.
    """
    if project_name in ['Axies', 'StepN']:
        trigger = 'None' if project_name == 'Axies' else 'none'
        NFT_info = [nft for nft in NFT_info if trigger not in nft['trait']]

    if project_name in ['BoredApeYachtClub', 'RoaringLeader', 'CryptoKitties']:
        for trait in trait_system.keys():
            trait_system[trait].append('none')
        if project_name in ['RoaringLeader', 'CryptoKitties']:
            lastT = -4 if project_name == 'RoaringLeader' else -16
            traits = list(trait_system.keys())[:lastT]
            trait_system = {key: trait_system[key] for key in traits}
            NFT_info = [{**nft, 'trait': nft['trait'][:lastT]} for nft in NFT_info]

    if project_name == 'StepN':
        NFT_info, trait_system = Augment_StepN(NFT_info, trait_system)

def Augment_StepN(asset_traits, trait_system):
    names = ['Efficiency', 'Comfort', 'Durability', 'Luck', 'Efficiency-lv1', 'Comfort-lv1', 'Durability-lv1', 'Luck-lv1', 'Efficiency-lv2', 'Comfort-lv2', 'Durability-lv2', 'Luck-lv2', 'Gem',]
    socket1 = [22036, 21577, 20264, 18207, 1546, 944, 828, 681, 603, 575, 352, 242, 220]
    socket2 = [23864, 22188, 21187, 19316, 396, 380, 356, 206, 179, 163, 144, 136, 103]
    trait_system['socket1'] = names
    trait_system['socket2'] = names
    M = len(asset_traits)
    attr1_list = random.choices(range(len(socket1)), weights=socket1, k=M)
    attr2_list = random.choices(range(len(socket2)), weights=socket2, k=M)
    new_asset_traits = []
    for asset, attr1, attr2 in zip(asset_traits, attr1_list, attr2_list):
        new_asset_traits.append(asset+[names[attr1], names[attr2]])
    
    return new_asset_traits, trait_system

def fetchinfo(transaction):
    return transaction['buyer_address'], transaction['price'], int(transaction['token_ids'][0])

def process_nft_trades(trade_info, NFT_info, trait_system):
    '''
    nft_project_data: dict_keys(['trait_system', 
    'asset_traits', 'item_counts', => size N
    'buyer_budgets', 'buyer_assets_ids', => size M])
    '''
    # Initialize dictionaries for buyers and assets
    buyer_info = defaultdict(lambda: {'budget': 0, 'asset_ids': []})
    asset_info = {'asset_traits':[], 'item_counts':[], 'atuples':[]}
    # asset_info = defaultdict(lambda: {'count': 0, 'id': []})
    new_asset_traits = []

    token_id2asset = {x['tokenId']:x for x in NFT_info}
    # Process each transaction
    for transaction in trade_info:
        buyer_add, price, token_id = fetchinfo(transaction)
        if token_id in token_id2asset:
            asset_trait = token_id2asset['token_id']['trait']
            atuple = tuple(asset_trait)
            
            if atuple not in asset_info['atuples']:
                aid = len(asset_info['atuples'])
                asset_info['atuples'].append(atuple)
                asset_info['asset_traits'].append(asset_trait)
                asset_info['item_counts'].append(0)
            else:
                aid = asset_info['atuples'].index(atuple)
                asset_info['item_counts'][aid] += 1
            
            buyer_info[buyer_add]['budget'] += price
            buyer_info[buyer_add]['asset_ids'].append(aid)


def consolidate(asset_traits, token_id_list, trade_info, params):
    buyer_add2bid = {}
    buyer_budgets = []
    buyer_assets = []
    buyer_assets_ids = []
    atuple2aid = {}
    item_counts = []
    new_asset_traits = []
    # organized_trade = []
    for transaction in tqdm(trade_info, ncols=88, desc='conso-iter-trade'):
        buyer_add, price, token_id = fetchinfo(transaction, params)
        if token_id in token_id_list:
            asset = asset_traits[token_id_list.index(token_id)]
            atuple = tuple(asset)
            if atuple not in atuple2aid:
                atuple2aid[atuple] = len(atuple2aid)
                new_asset_traits.append(asset)
                item_counts.append(1)
            else:
                item_counts[atuple2aid[atuple]] += 1
            aid = atuple2aid[atuple]
            if buyer_add in buyer_add2bid:
                bid = buyer_add2bid[buyer_add]
                buyer_budgets[bid] += price
                buyer_assets[bid].append(asset)
                buyer_assets_ids[bid].append(aid)
            else:
                buyer_add2bid[buyer_add] = bid = len(buyer_add2bid)
                buyer_budgets.append(price)
                buyer_assets.append([asset])
                buyer_assets_ids.append([aid])
            # organized_trade.append((bid, aid, price))
    return new_asset_traits, item_counts, buyer_budgets, buyer_assets, buyer_assets_ids

    
   

