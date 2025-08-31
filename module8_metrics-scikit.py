import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix

def read_n(prompt: str) -> int:
    while True:
        try:
            n = int(input(prompt).strip())
            if n > 0:
                return n
            print("Enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def read_bin(prompt: str) -> int:
    while True:
        val = input(prompt).strip()
        if val in ("0", "1"):
            return int(val)
        print("Enter 0 or 1.")

def main():
    n = read_n("Number of points (N): ")

    # Preallocate: col 0 = true labels, col 1 = predictions
    data = np.empty((n, 2), dtype=int)

    for i in range(n):
        t = read_bin(f"True label {i+1}: ")
        p = read_bin(f"Pred label {i+1}: ")
        data[i] = [t, p]

    true = data[:, 0]
    pred = data[:, 1]

    prec = precision_score(true, pred, zero_division=0)
    rec  = recall_score(true, pred, zero_division=0)

    print(f"\nPrecision: {prec:.2f}")
    print(f"Recall:    {rec:.2f}")

    # Optional: confusion matrix
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0,1]).ravel()
    print(f"\nConfusion Matrix:\nTN={tn}, FP={fp}, FN={fn}, TP={tp}")

if __name__ == "__main__":
    main()
