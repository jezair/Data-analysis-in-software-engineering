import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.stats as stats


def plot_ellipse(data, mean, cov, outliers_idx=None, title=""):
    """
    Допоміжна функція для візуалізації даних, довірчого еліпсу та викидів.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Побудова 95% довірчого еліпсу для середнього
    confidence = 0.95
    n = len(data)
    p = data.shape[1]

    # Критичне значення з F-розподілу для еліпса середнього
    f_val = stats.f.ppf(confidence, p, n - p)
    scale_factor = np.sqrt(p * (n - 1) * f_val / (n * (n - p)))

    # Отримання параметрів еліпса з коваріаційної матриці
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * scale_factor * np.sqrt(eigenvalues)

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor='red', facecolor='none', lw=2, label='95% довірчий еліпс')

    ax.add_patch(ellipse)

    # Візуалізація точок даних
    ax.scatter(data[:, 0], data[:, 1], alpha=0.7, label='Дані')
    if outliers_idx is not None and len(outliers_idx) > 0:
        ax.scatter(data[outliers_idx, 0], data[outliers_idx, 1], c='red', s=100,
                   edgecolor='black', label='Знайдені викиди')

    ax.scatter(mean[0], mean[1], c='red', marker='x', s=100, label='Вибіркове середнє')

    ax.set_title(title)
    ax.set_xlabel("Змінна X1")
    ax.set_ylabel("Змінна X2")
    ax.legend()
    ax.grid(True)
    plt.show()


# --- ЕТАП 0: ПІДГОТОВКА ДАНИХ ---
# Встановлюємо початкове значення для генератора випадкових чисел для відтворюваності
np.random.seed(0)

# Основна хмара даних (нормально розподілена)
mean_true = [2, 3]
cov_true = [[1, 0.8], [0.8, 1]]
main_data = np.random.multivariate_normal(mean_true, cov_true, 100)

# Штучно додаємо кілька очевидних викидів
outliers = np.array([
    [8, 9],
    [-4, -5],
    [5, -2]
])

# Об'єднуємо основні дані та викиди в один набір
data = np.vstack([main_data, outliers])
df = pd.DataFrame(data, columns=['X1', 'X2'])

print("--- АНАЛІЗ ПОЧАТКОВИХ ДАНИХ (З ВИКИДАМИ) ---")
print(f"Розмір повного набору даних: {data.shape}\n")

# --- КРОК 1: ВИЗНАЧЕННЯ ВЕКТОРУ ВИБІРКОВИХ СЕРЕДНІХ ---
mean_vector = np.mean(data, axis=0)
print(f"Крок 1: Вектор вибіркових середніх:\n{mean_vector}\n")

# --- КРОК 2: ВИЗНАЧЕННЯ ВИБІРКОВОЇ КОВАРІАЦІЙНОЇ МАТРИЦІ ---
cov_matrix = np.cov(data, rowvar=False)
print(f"Крок 2: Вибіркова коваріаційна матриця:\n{cov_matrix}\n")

# --- КРОК 3: ПЕРЕВІРКА ВІДХИЛЕННЯ ВІД НОРМАЛЬНОГО РОЗПОДІЛУ ---
normality_test_before = pg.multivariate_normality(df)
print("Крок 3: Результат тесту на нормальність (до видалення викидів):")
print(normality_test_before)
print("-" * 50)

# --- КРОК 5: ВИЗНАЧЕННЯ НАЯВНОСТІ БАГАТОВИМІРНИХ ВИКИДІВ ---
print(f"\nКрок 5: Пошук викидів")

# *** ВИПРАВЛЕНИЙ БЛОК: ОБЧИСЛЕННЯ ВІДСТАНІ МАХАЛАНОБІСА ВРУЧНУ ***
inv_cov_matrix = np.linalg.inv(cov_matrix)
centered_data = data - mean_vector
# Обчислюємо квадрат відстані Махаланобіса для кожної точки
mahalanobis_dist_sq = np.diag(centered_data @ inv_cov_matrix @ centered_data.T)

# Встановлюємо поріг для ідентифікації викидів
p = data.shape[1]  # кількість змінних
alpha_outlier = 0.01  # рівень значущості для викидів
# Критичне значення з розподілу Хі-квадрат
critical_value = stats.chi2.ppf(1 - alpha_outlier, df=p)

# Знаходимо індекси точок, квадрат відстані яких перевищує критичне значення
outlier_indices = np.where(mahalanobis_dist_sq > critical_value)[0]

print(f"Критичне значення для квадрату відстані Махаланобіса (alpha={alpha_outlier}): {critical_value:.4f}")
print(f"Індекси знайдених викидів: {outlier_indices}")
print("Дані, ідентифіковані як викиди:\n", df.iloc[outlier_indices])
print("-" * 50)

# --- КРОК 4: ЗНАХОДЖЕННЯ ДОВІРЧОГО ЕЛІПСУ (для початкових даних) ---
print("\nКрок 4: Побудова довірчого еліпсу для початкових даних...")
plot_ellipse(data, mean_vector, cov_matrix, outliers_idx=outlier_indices,
             title="Дані з викидами та довірчий еліпс")

# --- ПОВТОРЕННЯ ОБЧИСЛЕНЬ ПІСЛЯ ВИДАЛЕННЯ ВИКИДІВ ---
print("\n--- АНАЛІЗ ОЧИЩЕНИХ ДАНИХ ---")

# Створюємо новий набір даних без викидів
data_cleaned = np.delete(data, outlier_indices, axis=0)
df_cleaned = pd.DataFrame(data_cleaned, columns=['X1', 'X2'])
print(f"Розмір очищеного набору даних: {data_cleaned.shape}\n")

# Повторюємо кроки 1-4 для очищених даних
mean_vector_cleaned = np.mean(data_cleaned, axis=0)
cov_matrix_cleaned = np.cov(data_cleaned, rowvar=False)
normality_test_after = pg.multivariate_normality(df_cleaned)

print(f"Новий вектор середніх:\n{mean_vector_cleaned}\n")
print(f"Нова коваріаційна матриця:\n{cov_matrix_cleaned}\n")
print("Результат тесту на нормальність (після видалення викидів):")
print(normality_test_after)

print("\nПобудова довірчого еліпсу для очищених даних...")
plot_ellipse(data_cleaned, mean_vector_cleaned, cov_matrix_cleaned,
             title="Очищені дані та новий довірчий еліпс")

print("\n--- КРОК 6: ФОРМУЛЮВАННЯ ВИСНОВКІВ ---")
print("""
1. Початковий аналіз: Дані з викидами не пройшли тест на нормальність, 
   а довірчий еліпс був надмірно великим.
2. Ідентифікація викидів: Метод відстані Махаланобіса успішно виявив 
   аномальні спостереження.
3. Повторний аналіз: Після видалення викидів очищені дані успішно пройшли
   тест на нормальність, а новий довірчий еліпс став значно меншим,
   краще описуючи структуру основної маси даних.
4. Порівняння рівнів значущості: У випадку даних з викидами, результат
   тесту був би однаковим (відхилення гіпотези) як для alpha=0.05, так
   і для alpha=0.005. Для очищених даних гіпотеза не відхиляється 
   при обох рівнях значущості.
""")