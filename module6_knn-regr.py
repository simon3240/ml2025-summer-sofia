import numpy as np


N = int(input("Enter N (number of points): "))
if N <= 0:
    raise ValueError("N must be a positive integer.")

# Read k (positive integer)
k = int(input("Enter k: "))
if k <= 0:
    raise ValueError("k must be a positive integer.")

# Read N (x, y) points
data = []
for i in range(N):
    x = float(input(f"Enter x value for point {i+1}: "))
    y = float(input(f"Enter y value for point {i+1}: "))
    data.append([x, y])

print("\nData points entered:")
for point in data:
    print(point)




for i in range(len(data)):
    data[i][0]=abs(data[i][0]-x)
data=np.array(data)
# print(data)

data = data[data[:, 0].argsort()]
# print(data)

k_data=data[:k,:]
# print(k_data)

result=np.mean(k_data[:,1])

print(result)
