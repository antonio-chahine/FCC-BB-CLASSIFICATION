import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from math import sqrt

'''
Need the same test set for both classifiers. 
'''

# ---------------------------------------------------------
# DeLong implementation (standalone, no installation needed)
# ---------------------------------------------------------
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m])
    ty = np.empty([k, n])

    for r in range(k):
        tx[r] = compute_midrank(positive_examples[r])
        ty[r] = compute_midrank(negative_examples[r])

    aucs = tx.mean(axis=1) / m - ty.mean(axis=1) / n

    v01 = (tx - tx.mean(axis=1, keepdims=True)) / m - (ty - ty.mean(axis=1, keepdims=True)) / n
    sx = np.cov(v01)

    return aucs, sx

def delong_roc_variance(y_true, y_scores):
    order = np.argsort(-y_scores)
    y_scores = y_scores[order]
    y_true = y_true[order]

    label_1_count = np.sum(y_true)
    predictions_sorted_transposed = np.vstack((y_scores,))

    aucs, delong_cov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return aucs[0], delong_cov[0,0]

def delong_test(y_true, scoresA, scoresB):
    aucA, varA = delong_roc_variance(y_true, scoresA)
    aucB, varB = delong_roc_variance(y_true, scoresB)

    diff = aucA - aucB
    var = varA + varB
    z = diff / sqrt(var)
    from math import erf
    p = 2 * (1 - 0.5 * (1 + erf(abs(z)/sqrt(2))))
    return diff, p

# ---------------------------------------------------------
# Load classifier predictions
# ---------------------------------------------------------
y_test_A, y_proba_A = pickle.load(open("results_classifierA.pkl", "rb"))
y_test_B, y_proba_B = pickle.load(open("results_classifierB.pkl", "rb"))

assert np.all(y_test_A == y_test_B), "Test sets do not match!"

diff, p = delong_test(y_test_A, y_proba_A, y_proba_B)

print("AUC(Classifier A):", roc_auc_score(y_test_A, y_proba_A))
print("AUC(Classifier B):", roc_auc_score(y_test_B, y_proba_B))
print("\nDifference A - B =", diff)
print("p-value =", p)

if p < 0.05:
    print("\n>>> The AUC difference IS statistically significant.")
else:
    print("\n>>> The AUC difference is NOT statistically significant.")
