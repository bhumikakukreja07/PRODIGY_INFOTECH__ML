import pandas as pd
from sklearn.linear_model import LinearRegression

try:
    train = pd.read_csv("train.csv", usecols=["LotArea", "SalePrice", "FullBath", "BedroomAbvGr"])
except FileNotFoundError:
    print("File not found.")
    exit()

X = train[["LotArea", "FullBath", "BedroomAbvGr"]]
y = train["SalePrice"]

model = LinearRegression().fit(X, y)

test = pd.read_csv("test.csv", usecols=["LotArea", "FullBath", "BedroomAbvGr"])
test["SalePrice"] = model.predict(test)

print(test)
test.to_csv("test_result.csv", index=False)