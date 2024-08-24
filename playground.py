import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 2, 3, 4, 5])

model = LinearRegression()
model.fit(x, y)

print(model.coef_)
print(model.intercept_)