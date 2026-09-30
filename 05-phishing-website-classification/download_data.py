import os,pandas as pd
from ucimlrepo import fetch_ucirepo
os.makedirs('data',exist_ok=True); ds=fetch_ucirepo(id=327); pd.concat([ds.data.features,ds.data.targets],axis=1).to_csv('data/dataset.csv',index=False); print('Dataset saved')
