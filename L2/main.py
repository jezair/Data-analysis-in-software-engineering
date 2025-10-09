import numpy as np
from scipy.stats import chi2, norm


def mardia_test_manual(data):
    """
    Функція для обчислення статистики тесту Мардіа вручну за формулами.
    Приймає на вхід дані у вигляді numpy array.
    """

    n, p = data.shape

    # Віднімаємо вектор середніх від кожного спостереження
    x_bar = np.mean(data, axis=0)
    centered_data = data - x_bar

    S = np.cov(data, rowvar=False)
    S_inv = np.linalg.inv(S)

    # --- 1. Багатовимірна Асиметрія (b_1,p) ---
    # Формула: b_1,p = (1/n^2) * Σ_i Σ_j [ (x_i - x̄)' S⁻¹ (x_j - x̄) ]³

    # Розраховуємо g_ij = (x_i - x̄)' S⁻¹ (x_j - x̄) для всіх пар (i, j)
    # Це можна зробити ефективно за допомогою матричного множення
    g = centered_data @ S_inv @ centered_data.T

    # Підносимо кожен елемент до кубу і сумуємо
    b1p = np.sum(g ** 3) / (n ** 2)

    # Тестова статистика A для асиметрії
    # A = n * b_1,p / 6
    A = n * b1p / 6

    # Ступені свободи для розподілу хі-квадрат
    df_skew = p * (p + 1) * (p + 2) / 6

    # p-value - ймовірність отримати значення > A
    # Використовуємо Survival Function (1 - CDF) для розподілу chi2
    p_value_skew = chi2.sf(A, df_skew)

    # --- 2. Багатовимірний Ексцес (b_2,p) ---
    # Формула: b_2,p = (1/n) * Σ_i [ (x_i - x̄)' S⁻¹ (x_i - x̄) ]²

    # (x_i - x̄)' S⁻¹ (x_i - x̄) - це діагональні елементи матриці g,
    # тобто квадрат відстані Махаланобіса
    mahalanobis_sq = np.diag(g)

    # Розраховуємо ексцес
    b2p = np.sum(mahalanobis_sq ** 2) / n

    # Тестова статистика B для ексцесу
    # B = (b_2,p - p(p+2)) / sqrt(8p(p+2)/n)
    numerator = b2p - p * (p + 2)
    denominator = np.sqrt(8 * p * (p + 2) / n)
    B = numerator / denominator

    # p-value для нормального розподілу N(0, 1)
    # Оскільки відхилення може бути в обидва боки, це двосторонній тест.
    # Ми беремо ймовірність хвоста і множимо на 2.
    p_value_kurt = 2 * norm.sf(np.abs(B))

    return {
        'skewness': b1p, 'skew_statistic (A)': A, 'p_value_skew': p_value_skew,
        'kurtosis': b2p, 'kurt_statistic (B)': B, 'p_value_kurt': p_value_kurt
    }


# --- Створення даних для прикладу (ті ж самі, що й раніше) ---
np.random.seed(42)
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
data_normal = np.random.multivariate_normal(mean, cov, 100)
data_not_normal = np.random.exponential(size=(100, 2))
alpha = 0.005

# --- Застосування функції та вивід результатів ---

print("--- Результати ручних розрахунків для НОРМАЛЬНИХ даних ---")
manual_normal_results = mardia_test_manual(data_normal)
for key, value in manual_normal_results.items():
    print(f"{key}: {value:.6f}")

is_normal_skew = manual_normal_results['p_value_skew'] > alpha
is_normal_kurt = manual_normal_results['p_value_kurt'] > alpha
print(f"\nВідповідає нормальному розподілу (асиметрія)? -> {is_normal_skew}")
print(f"Відповідає нормальному розподілу (ексцес)? -> {is_normal_kurt}")
print("\n" + "=" * 50 + "\n")

print("--- Результати ручних розрахунків для НЕ НОРМАЛЬНИХ даних ---")
manual_not_normal_results = mardia_test_manual(data_not_normal)
for key, value in manual_not_normal_results.items():
    print(f"{key}: {value:.6f}")

is_not_normal_skew = manual_not_normal_results['p_value_skew'] > alpha
is_not_normal_kurt = manual_not_normal_results['p_value_kurt'] > alpha
print(f"\nВідповідає нормальному розподілу (асиметрія)? -> {is_not_normal_skew}")
print(f"Відповідає нормальному розподілу (ексцес)? -> {is_not_normal_kurt}")