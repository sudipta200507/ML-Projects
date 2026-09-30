import joblib
from ucimlrepo import fetch_ucirepo
model=joblib.load("models/model.joblib"); ds=fetch_ucirepo(id=222)
row=ds.data.features.iloc[[0]]
print("Predicted subscription:", int(model.predict(row)[0]))
print("Probability:", float(model.predict_proba(row)[0,1]))
