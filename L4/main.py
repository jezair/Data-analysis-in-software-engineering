import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

data_multi = {
    'Effort': [100, 150, 500, 250, 300, 700, 450, 600, 200, 350, 400, 800],
    'FP': [200, 310, 900, 410, 590, 1300, 800, 1100, 350, 600, 750, 1500],
    'Duration': [6, 8, 24, 12, 12, 30, 18, 24, 10, 14, 16, 36]
}
df_multi = pd.DataFrame(data_multi)
df_multi['LOC_Correlated'] = df_multi['FP'] * 50 + np.random.randint(-1000, 1000, df_multi.shape[0])

df_single = df_multi[['Effort', 'FP']].copy()

print("--- 0. Ісходні дані (множинні) ---")
print(df_multi.head())
print("\n--- 0. Ісходні дані (однофакторні) ---")
print(df_single.head())

# ---------------------------------------------------------------------
print("\n--- Етап 1: Перевірка на мультиколінеарність (VIF) ---")
# ---------------------------------------------------------------------

X_multi = df_multi[['FP', 'Duration', 'LOC_Correlated']]
Y_multi = df_multi['Effort']

def calculate_vif(X_df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_df.columns
    vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(len(X_df.columns))]
    return vif_data

vif_results = calculate_vif(X_multi)
print("VIF до відкидання факторів:")
print(vif_results)

X_multi_reduced = X_multi.drop('LOC_Correlated', axis=1)

print("\nVIF після відкидання 'LOC_Correlated':")
vif_results_reduced = calculate_vif(X_multi_reduced)
print(vif_results_reduced)

# ---------------------------------------------------------------------
print("\n--- Етап 2: Визначення оцінок параметрів (Побудова моделі) ---")
# ---------------------------------------------------------------------

# --- 2.1 Однофакторна модель ---
X_single = df_single['FP']
Y_single = df_single['Effort']

X_single_const = sm.add_constant(X_single)

model_single = sm.OLS(Y_single, X_single_const).fit()

print("\n--- Параметри ОДНОФАКТОРНОЇ моделі (Effort ~ FP) ---")
print(model_single.params)

# --- 2.2 Множинна модель (зменшена) ---
X_multi_reduced_const = sm.add_constant(X_multi_reduced)

model_multi = sm.OLS(Y_multi, X_multi_reduced_const).fit()

print("\n--- Параметри МНОЖИННОЇ моделі (Effort ~ FP + Duration) ---")
print(model_multi.params)

print("\n--- Повне зведення (summary) ОДНОФАКТОРНОЇ моделі ---")
print(model_single.summary())

# ---------------------------------------------------------------------
print("\n--- Етап 3: Розрахунок метрик (R², MMRE, PRED(0.25)) ---")
# ---------------------------------------------------------------------
Y_true = Y_single
Y_pred = model_single.predict(X_single_const)

r_squared = model_single.rsquared
print(f"Коефіцієнт детермінації (R²): {r_squared:.4f}")

mask = Y_true != 0
mre = np.abs(Y_true[mask] - Y_pred[mask]) / Y_true[mask]

mmre = np.mean(mre)
print(f"Середня величина відносної похибки (MMRE): {mmre:.4f}")

pred_025 = np.mean(mre <= 0.25)
print(f"PRED(0.25): {pred_025:.4f} (тобто {pred_025 * 100:.2f} % прогнозів)")

if pred_025 >= 0.75:
    print("Якість PRED(0.25) вважається високою.")
else:
    print("Якість PRED(0.25) вважається невисокою.")

# ---------------------------------------------------------------------
print("\n--- Етап 4: Побудова лінії регресії (Однофакторний випадок) ---")
# ---------------------------------------------------------------------

plt.figure(figsize=(10, 6))
sns.regplot(x=X_single, y=Y_single, ci=None,
            line_kws={'color': 'red', 'label': 'Лінія регресії'},
            scatter_kws={'label': 'Емпіричні дані'})

plt.title('Лінія регресії та емпіричні дані (Effort vs. FP)')
plt.xlabel('Function Points (FP)')
plt.ylabel('Effort')
plt.legend()
plt.grid(True)
# plt.show() # Розкоментуйте, якщо запускаєте як .py скрипт
print("Графік буде показано (у Jupyter/Colab) або у окремому вікні.")


# ---------------------------------------------------------------------
print("\n--- Етап 5: Визначення відхилень (залишків) ---")
# ---------------------------------------------------------------------
residuals = model_single.resid

print("Відхилення (залишки) між емпіричними даними та лінією регресії:")
print(residuals)

# ---------------------------------------------------------------------
print("\n--- Етап 6: Перевірка гіпотези про нормальність відхилень ---")
# ---------------------------------------------------------------------

alpha = 0.05
shapiro_stat, shapiro_p = stats.shapiro(residuals)

print(f"Статистика тесту Шапіро-Уілка: {shapiro_stat:.4f}")
print(f"P-value: {shapiro_p:.4f}")

if shapiro_p > alpha:
    print(f"P-value ({shapiro_p:.4f}) > alpha ({alpha}).")
    print("Висновок: Немає підстав відхилити H0. Відхилення розподілені нормально.")
else:
    print(f"P-value ({shapiro_p:.4f}) <= alpha ({alpha}).")
    print("Висновок: Відхиляємо H0. Відхилення НЕ розподілені нормально.")

print("...будуємо Q-Q plot для візуальної перевірки...")
fig_qq = sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot відхилень (залишків)')

# ---------------------------------------------------------------------
print("\n--- Етап 7: Висновки ---")
# ---------------------------------------------------------------------
print("Цей етап - аналітичний. Див. коментарі в попередній відповіді.")


# ---------------------------------------------------------------------
print("\n--- Завдання для самостійної роботи: T-тест та F-тест ---")
# ---------------------------------------------------------------------

print(f"\n--- F-тест (загальна значущість) ---")
print(f"F-статистика: {model_single.fvalue:.4f}")
print(f"P-value F-тесту: {model_single.f_pvalue:.4f}")
if model_single.f_pvalue < 0.05:
    print("-> Модель в цілому є значущою.")
else:
    print("-> Модель в цілому не є значущою.")

print(f"\n--- T-тести (значущість факторів) ---")
print(model_single.pvalues)
if model_single.pvalues['FP'] < 0.05:
    print("-> Фактор 'FP' є значущим.")
else:
    print("-> Фактор 'FP' не є значущим.")

plt.show()