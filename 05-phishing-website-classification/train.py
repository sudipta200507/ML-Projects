import os,joblib
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
os.makedirs('models',exist_ok=True); ds=fetch_ucirepo(id=327); X=ds.data.features; y=ds.data.targets.iloc[:,0]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42); m=Pipeline([('scale',StandardScaler()),('model',KNeighborsClassifier(n_neighbors=5))]); m.fit(Xtr,ytr); p=m.predict(Xte); print(classification_report(yte,p)); print(confusion_matrix(yte,p)); joblib.dump(m,'models/model.joblib')
