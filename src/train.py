import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
import json

df = pd.read_csv("data/housing.csv")

X = df[['rooms']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)
mse = mean_squared_error(y, pred)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("metrics.json", "w") as f:
    json.dump({"mse": mse}, f)

print("Training complete. MSE =", mse)