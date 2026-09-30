import os,joblib
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report,roc_auc_score
os.makedirs('models',exist_ok=True); ds=fetch_ucirepo(id=17); X=ds.data.features; y=ds.data.targets.iloc[:,0]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42); m=Pipeline([('scale',StandardScaler()),('model',SVC(kernel='rbf',probability=True,random_state=42))]); m.fit(Xtr,ytr); p=m.predict(Xte); print(classification_report(yte,p)); print('ROC-AUC:',roc_auc_score(yte,m.predict_proba(Xte)[:,1])); joblib.dump(m,'models/model.joblib')
