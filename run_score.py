import pandas as pd
import numpy as np
import pickle

model_path = './model.pkl'
model_dict = pickle.load(open(model_path, "rb"))
model_list = model_dict['models']
ss = model_dict['normalization']
_dropvars = model_dict['dropvars']

X = pd.read_csv('./dataset.csv')
ID_column = X['ID']
X = X.drop(columns=_dropvars, errors='ignore')
X = pd.DataFrame(ss.transform(X[ss.feature_names_in_]),columns=ss.feature_names_in_)

Y_score = np.zeros(X.shape[0])
for model in model_list:
    Y_score += model.predict_proba(X)[:,1]/len(model_list)
    
df_score = pd.DataFrame(data={"Y_score":Y_score})
df_score['ID'] = ID_column

L = df_score.shape[0]
idx_array = np.linspace(0,L,101).astype(int)
idx_array = [L-1] + list(idx_array[[90,80,70,60,50,40,30,20,10,5,1,0]])
score_array = np.sort(df_score['Y_score'].to_numpy())[::-1]
score_array = score_array[idx_array]
score_array[0] = 0
idx_array = [100,90,80,70,60,50,40,30,20,10,5,1,0]

for idx, idx_next, score in zip(idx_array[:-1],idx_array[1:],score_array[:-1]):
    df_score.loc[df_score['Y_score']>=score,'RISK_GRP'] = ('%dth - %dth' %(idx_next+1,idx)) if idx_next else 'Top 1st percentile'
