import sys
sys.path.append('./')
from utils import dumpj, check
from pathlib import Path
import json
from collections import defaultdict

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
    "State": [],
    "Category": ["Restaurants", "Shopping", "Health", "Automotive", "Bookstores", "Pet", "Hotels", "Museums", "Gyms", "Grocery"],
}

yelp_filename = 'yelp_academic_dataset_{}.json'
datadir = Path('../../../DATASET/yelp/')

asset_traits = []
item_counts = []
business_id = []

with open(datadir/yelp_filename.format('business')) as file:
    for line in file:
        bdict = json.loads(line)
        if bdict['review_count'] < 10:
            continue
        if bdict['state'] not in trait_system['State']:
            trait_system['State'].append(bdict['state'])

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

buyer_assets_ids = defaultdict(set)

with open(datadir/yelp_filename.format('business')) as file:
    for line in file:
        rdict = json.loads(line)
        if rdict['business_id'] not in business_id:
            continue
        if rdict['stars'] >= 3:
            buyer_assets_ids[rdict['user_id']].add(business_id.index(rdict['business_id']))

buyer_assets_ids = [list(v) for v in buyer_assets_ids.values()]


yelp_nft_data = {
    'trait_system': trait_system,
    'asset_traits': asset_traits,
    'item_counts': item_counts,
    'buyer_budgets': buyer_budgets,
    'buyer_assets_ids': buyer_assets_ids,
}

dumpj(yelp_nft_data, datadir/'yelp_nft_data.json')