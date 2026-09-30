import joblib
from ucimlrepo import fetch_ucirepo
import pandas as pd
model = joblib.load("models/model.joblib")
ds = fetch_ucirepo(id=275)
df = pd.concat([ds.data.features, ds.data.targets], axis=1)
row = df.drop(columns=[c for c in ["cnt","casual","registered","dteday"] if c in df]).iloc[[0]]
print("Predicted demand:", float(model.predict(row)[0]))
