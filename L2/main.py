import numpy as np
from scipy.stats import chi2, norm


def mardia_test(data):
    n, p = data.shape
    x_bar = np.mean(data, axis=0)
    centered_data = data - x_bar
    S = np.cov(data, rowvar=False)
    S_inv = np.linalg.inv(S)

    g = centered_data @ S_inv @ centered_data.T
    b1p = np.sum(g ** 3) / (n ** 2)

    A = n * b1p / 6
    df_skew = p * (p + 1) * (p + 2) / 6
    p_value_skew = chi2.sf(A, df_skew)

    mahalanobis_sq = np.diag(g)
    b2p = np.mean(mahalanobis_sq)

    numerator = b2p - p * (p + 2)
    denominator = np.sqrt(8 * p * (p + 2) / n)
    B = numerator / denominator
    p_value_kurt = 2 * norm.sf(np.abs(B))

    return {
        'skewness_b1p': b1p, 'p_value_skew': p_value_skew,
        'kurtosis_b2p': b2p, 'p_value_kurt': p_value_kurt
    }


def check_normality(results, alpha):
    is_skew_normal = results['p_value_skew'] > alpha
    print(
        f"За асиметрією (p={results['p_value_skew']:.4f}): розподіл {'можна вважати нормальним' if is_skew_normal else 'не є нормальним'}.")

    is_kurt_normal = results['p_value_kurt'] > alpha
    print(
        f"За ексцесом (p={results['p_value_kurt']:.4f}): розподіл {'можна вважати нормальним' if is_kurt_normal else 'не є нормальним'}.")

    if is_skew_normal and is_kurt_normal:
        print("Підсумок: дані є багатовимірно нормальними.")
    else:
        print("Підсумок: дані не є багатовимірно нормальними.")


# --- Дані для аналізу ---
np.random.seed(42)
data_normal = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.8], [0.8, 1]], size=150)
data_not_normal = np.random.exponential(size=(150, 2))

# --- Аналіз НЕнормальних даних ---
print("--- Аналіз для НЕнормальних даних ---")
results = mardia_test(data_not_normal)
print(f"Розрахована асиметрія (b1,p) = {results['skewness_b1p']:.4f}")
print(f"Розрахований ексцес (b2,p) = {results['kurtosis_b2p']:.4f}")

print(f"\nПеревірка при alpha = 0.005:")
check_normality(results, alpha=0.005)

print(f"\nПеревірка при alpha = 0.05:")
check_normality(results, alpha=0.05)