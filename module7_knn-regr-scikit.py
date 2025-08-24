import numpy as np
from sklearn.neighbors import KNeighborsRegressor

# - basic settings -
n_points = int(input("How many training points N? "))
if n_points <= 0:
    raise ValueError("N must be a positive integer.")

k_neighbors = int(input("Choose k (number of neighbors): "))
if k_neighbors <= 0:
    raise ValueError("k must be a positive integer.")

# - collect N (x, y) pairs into a preallocated NumPy array -
# column 0 -> x, column 1 -> y
pairs = np.empty((n_points, 2), dtype=float)
for i in range(n_points):
    xi = float(input(f"x[{i+1}]: "))
    yi = float(input(f"y[{i+1}]: "))
    pairs[i, 0] = xi
    pairs[i, 1] = yi

print("\nTraining data:")
for i, (xi, yi) in enumerate(pairs, start=1):
    print(f"{i}: ({xi}, {yi})")

# - query -
x_query = float(input("\nQuery X: "))

# variance of labels (population variance; use ddof=1 for sample variance)
y_vals = pairs[:, 1]
var_y = float(np.var(y_vals))



# - guard & prediction -
if k_neighbors > n_points:
    print(f"Error: k ({k_neighbors}) must be ≤ N ({n_points}).")
    print(f"Variance of labels (y): {var_y:.6f}")
else:
    X_train = pairs[:, 0].reshape(-1, 1)
    reg = KNeighborsRegressor(n_neighbors=k_neighbors)
    reg.fit(X_train, y_vals)

    y_pred = float(reg.predict(np.array([[x_query]]))[0])

    print(f"\nPredicted Y for X = {x_query}: {y_pred:.6f}")
    print(f"Variance of labels (y): {var_y:.6f}")
