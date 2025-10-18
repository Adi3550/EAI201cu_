import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


data = {
    'House': [1, 2, 3, 4, 5],
    'Area (sqft)': [1200, 1400, 1600, 1700, 1850],
    'Rooms': [3, 4, 3, 5, 4],
    'Distance (km)': [5, 3, 8, 2, 4],
    'Age (years)': [10, 3, 20, 15, 7],
    'Price (₹ Lacs)': [120, 150, 130, 180, 170]
}
df = pd.DataFrame(data)
print("Dataset created!")
print(df)


plt.figure(figsize=(10, 6))
plt.scatter(df['Area (sqft)'], df['Price (₹ Lacs)'], color='blue')
plt.xlabel('Area (sqft)')
plt.ylabel('Price (₹ Lacs)')
plt.title('Area vs Price')
plt.show()


X = df[['Area (sqft)', 'Rooms', 'Distance (km)', 'Age (years)']]  
y = df['Price (₹ Lacs)']

print("Features shape:", X.shape)
print("Target shape:", y.shape)


model = LinearRegression()
model.fit(X, y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Model trained!")

predictions = model.predict(X)
print("Predicted Prices:", predictions.round(2))
print("Actual Prices:", y.values)

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y)), y, color='blue', label='Actual Price')
plt.scatter(range(len(predictions)), predictions, color='red', label='Predicted Price')
plt.xlabel('House Index')
plt.ylabel('Price (₹ Lacs)')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()

new_house = [[1500, 4, 6, 10]]  
predicted_price = model.predict(new_house)
print(f"Predicted price for new house: ₹ {predicted_price[0].round(2)} Lacs")
