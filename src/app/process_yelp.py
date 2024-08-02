import sys
sys.path.append('./')
from utils import dumpj, check
from pathlib import Path
import json
from collections import defaultdict
from tqdm import tqdm
import random

'''
format the yelp data into a dictionary with keys:
    - 'trait_system'
    - 'asset_traits'
    - 'item_counts'
    - 'buyer_budgets,
    'buyer_assets_ids': buyer_assets_ids,'   [28, 49, 42, ...]
    - 'buyer_assets_ids'   [[0, 2084], [1], [2, 1], [3], ...]


dumpj(yelp_nft_data, datadir/'yelp_nft_data.json')
 
'''
trait_system = {
    "State": ["AZ", "CA", "PA", "ID", "MN", "NY", "TX"],
    "Category": ["Restaurants", "Shopping", "Health", "Hotels", "Museums", "Gyms", "Grocery"],
    # "Category": ["Restaurants", "Shopping", "Health", "Automotive", "Bookstores", "Pet", "Hotels", "Museums", "Gyms", "Grocery"],
}

yelp_filename = 'yelp_academic_dataset_{}.json'
datadir = Path('../../../DATASET/yelp/')

asset_traits = []
item_counts = []
business_id = []

with open(datadir/yelp_filename.format('business')) as file:
    for line in tqdm(file, ncols=90, desc='Processing Business Data'):
        if random.random() > 0.25:
            continue
        bdict = json.loads(line)
        if bdict['review_count'] < 10 or bdict['categories'] is None:
            continue
        if bdict['state'] not in trait_system['State']:
            continue
        asset_cat = None
        for category in trait_system['Category']:
            if category in bdict['categories']:
                asset_cat = category
                break
        if asset_cat is None:
            continue
        asset_traits.append([bdict['state'], asset_cat])
        item_counts.append(bdict['review_count'])
        business_id.append(bdict['business_id'])

review_edge = defaultdict(set)
edge_count = 0

with open(datadir/yelp_filename.format('review')) as file:
    for line in tqdm(file, ncols=90, desc='Processing Review Data'):
        rdict = json.loads(line)
        if rdict['business_id'] not in business_id:
            continue
        if rdict['stars'] >= 3:
            review_edge[rdict['user_id']].add(business_id.index(rdict['business_id']))
            edge_count += 1

        if edge_count % 10000 == 0:
            print(len(review_edge))
        if edge_count > 12000:
            break

buyer_assets_ids = [list(v) for v in review_edge.values()]
min_len = min(len(v) for v in buyer_assets_ids)
max_len = max(len(v) for v in buyer_assets_ids)
buyer_budgets = [10 + (len(v) - min_len) * 90 / (max_len - min_len) for v in buyer_assets_ids]

yelp_nft_data = {
    'trait_system': trait_system,
    'asset_traits': asset_traits,
    'item_counts': item_counts,
    'buyer_budgets': buyer_budgets,
    'buyer_assets_ids': buyer_assets_ids,
}

print(len(asset_traits), len(buyer_budgets), len(buyer_assets_ids))
dumpj(yelp_nft_data, 'yelp.json')