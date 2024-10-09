# redevelop NFT code
> God is with me, let's do this.


## Notes
(in `../extra_baselines`)
### reciprocal recommendation
> 
> https://github.com/CyberAgentAILab/FairReciprocalRecommendation
> 
- rye sync (remove from .bashrc afterwards?)
- https://github.com/CyberAgentAILab/FairReciprocalRecommendation/blob/main/notebooks/example.ipynb



## Logs 
[10/05/2024]
- start with multi-stakeholder .... give up
- next, try reciprocal ....


[10/04/2024]
- reopen this page to log experiment. 

- try reciprocal recommendation https://github.com/RUCAIBox/CRRS/tree/main `Revisiting Reciprocal Recommender Systems: Metrics, Formulation, and Method`

- has bug, switched to Recbole PJF
- try `python run_recbole_pjf.py -m LFRR`
  - pip install lightgbm
  - pip install xgboost
- work on dataset, try `python run_recbole_pjf.py -m LFRR -d jobrec`
  - `kaggle datasets download -d jsrshivam/job-recommendation-case-study` (should unzip in `dataset/jobrec`)
  - `python dataset/jobrec/prepare_jobrec.py`
  
- Final: use Fair Reciprocal Recommendation

- => TODO: 1) understand/formate rec-rl for baseline, 2) run fair reciprocal -> format

## Data:

> stored in NFT-game (now `dep`, need fix!!) \n
> include `trait_data` and `buyer_data`