import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
model=joblib.load('models/model.joblib')
ds=fetch_ucirepo(id=275); df=pd.concat([ds.data.features,ds.data.targets],axis=1)
df=df.drop(columns=[c for c in ['casual','registered','dteday'] if c in df.columns])
X=df.drop(columns='cnt'); y=df['cnt']; _,Xtest,_,ytest=train_test_split(X,y,test_size=.2,random_state=42)
p=model.predict(Xtest)
print(f'MAE={mean_absolute_error(ytest,p):.4f}')
print(f'RMSE={mean_squared_error(ytest,p)**0.5:.4f}')
print(f'R2={r2_score(ytest,p):.4f}')