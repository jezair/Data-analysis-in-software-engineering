import pandas as pd
import numpy as np
import pingouin as pg

# --- Крок 0: Створення даних для прикладу ---
# Створимо два набори даних для демонстрації:
# 1. data_normal: дані, що відповідають багатовимірному нормальному розподілу.
# 2. data_not_normal: дані, що НЕ відповідають нормальному розподілу (використаємо експоненційний).

np.random.seed(42) # для відтворюваності результатів
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]] # Коваріаційна матриця

# Створюємо 100 зразків з 2 змінними, які є багатовимірно нормальними
data_normal = np.random.multivariate_normal(mean, cov, 100)
df_normal = pd.DataFrame(data_normal, columns=['Змінна_1', 'Змінна_2'])

# Створюємо дані, що не є нормальними
data_not_normal = np.random.exponential(size=(100, 2))
df_not_normal = pd.DataFrame(data_not_normal, columns=['Змінна_1', 'Змінна_2'])


# --- Кроки 1, 2, 3: Оцінка, визначення статистики та перевірка ---
# Виконаємо тест Мардіа для обох наборів даних.
# Рівень значущості alpha (α) встановлюємо на 0.005, як вказано в умові.
alpha = 0.005

print("--- Результати для нормально розподілених даних ---")
# Функція multivariate_normality повертає статистику та p-value для асиметрії і ексцесу
mardia_normal_results = pg.multivariate_normality(df_normal, alpha=alpha)
print(mardia_normal_results)
print("\n")


print("--- Результати для НЕ нормально розподілених даних ---")
mardia_not_normal_results = pg.multivariate_normality(df_not_normal, alpha=alpha)
print(mardia_not_normal_results)