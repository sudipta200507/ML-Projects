import os, joblib, pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
os.makedirs("models",exist_ok=True)
ds=fetch_ucirepo(id=222); X=ds.data.features.copy(); y=ds.data.targets.iloc[:,0].astype(str)
y=y.map({"yes":1,"no":0})
cat=X.select_dtypes(include="object").columns; num=X.select_dtypes(exclude="object").columns
prep=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),("num",StandardScaler(),num)])
pipe=Pipeline([("prep",prep),("model",LogisticRegression(max_iter=1000))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
pipe.fit(Xtr,ytr); p=pipe.predict(Xte); prob=pipe.predict_proba(Xte)[:,1]
print(classification_report(yte,p)); print("ROC-AUC:",roc_auc_score(yte,prob)); print("Confusion matrix:\n",confusion_matrix(yte,p))
joblib.dump(pipe,"models/model.joblib")
