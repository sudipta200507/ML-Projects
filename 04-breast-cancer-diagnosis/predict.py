import joblib
from ucimlrepo import fetch_ucirepo
m=joblib.load('models/model.joblib'); ds=fetch_ucirepo(id=17); row=ds.data.features.iloc[[0]]; print('Predicted label:',m.predict(row)[0]); print('Probabilities:',m.predict_proba(row)[0])
