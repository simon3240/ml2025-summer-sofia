
N = int(input())
numbers = [int(input()) for _ in range(N)]
X = int(input())

try:
    print(numbers.index(X) + 1)  # 1-based index
except ValueError:
    print(-1)
