import os,joblib,pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,roc_auc_score
os.makedirs("models",exist_ok=True)
ds=fetch_ucirepo(id=2); X=ds.data.features.copy(); y=ds.data.targets.iloc[:,0].astype(str).str.strip().str.replace(".","",regex=False)
y=(y.str.contains(">50K")).astype(int)
cat=X.select_dtypes(include="object").columns; num=X.select_dtypes(exclude="object").columns
prep=ColumnTransformer([("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]),cat),("num",SimpleImputer(strategy="median"),num)])
pipe=Pipeline([("prep",prep),("model",RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=42,n_jobs=-1))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42); pipe.fit(Xtr,ytr)
p=pipe.predict(Xte); print(classification_report(yte,p)); print("ROC-AUC:",roc_auc_score(yte,pipe.predict_proba(Xte)[:,1])); joblib.dump(pipe,"models/model.joblib")
