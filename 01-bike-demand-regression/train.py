import os
import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("models", exist_ok=True)
ds = fetch_ucirepo(id=275)
df = pd.concat([ds.data.features, ds.data.targets], axis=1)
target = "cnt"
drop = [c for c in ["casual", "registered", "dteday"] if c in df.columns]
df = df.drop(columns=drop)
X = df.drop(columns=[target]); y = df[target]
num = X.columns.tolist()
pipe = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
print({"MAE":mean_absolute_error(y_test,pred),"RMSE":mean_squared_error(y_test,pred)**0.5,"R2":r2_score(y_test,pred)})
joblib.dump(pipe, "models/model.joblib")
