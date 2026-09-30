import joblib
from ucimlrepo import fetch_ucirepo
m=joblib.load("models/model.joblib"); ds=fetch_ucirepo(id=2); row=ds.data.features.iloc[[0]]
print("Predicted class:",int(m.predict(row)[0])); print("Probability >50K:",float(m.predict_proba(row)[0,1]))
