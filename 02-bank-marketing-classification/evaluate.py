import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix
ds=fetch_ucirepo(id=222); X=ds.data.features; y=ds.data.targets.iloc[:,0].astype(str).map({'yes':1,'no':0})
_,Xtest,_,ytest=train_test_split(X,y,test_size=.2,stratify=y,random_state=42); model=joblib.load('models/model.joblib'); pred=model.predict(Xtest); prob=model.predict_proba(Xtest)[:,1]
print(classification_report(ytest,pred)); print('ROC-AUC:',roc_auc_score(ytest,prob)); print('Confusion matrix:\n',confusion_matrix(ytest,pred))