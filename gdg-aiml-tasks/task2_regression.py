import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "Hours Studied": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    "Exam Score": [18, 32, 47, 63, 76, 84, 93, 98],
}
df = pd.DataFrame(data)

X = df[["Hours Studied"]]
y = df["Exam Score"]

model = LinearRegression()
model.fit(X, y)

hours_new = [[9.0]]
predicted_score = model.predict(hours_new)[0]

print(f"Slope (Coefficient): {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"\nPredicted Exam Score for 9 Hours of Study: {predicted_score:.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="blue", label="Actual Data Points")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.scatter(
    [9.0], [predicted_score], color="green", s=100, label="Prediction (9 hrs)"
)

plt.title("Study Hours vs Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.grid(True)
plt.show()