import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.stattools import medcouple

data = np.array([
    6020, 3570, 2496, 191, 406, 14520, 14430, 650, 225, 183,
    1080, 2464, 1041, 528, 212, 7060, 903, 4164, 633, 665,
    544, 5732, 1238, 1586, 281, 220, 851, 2391, 780, 591,
    391, 2263, 4112, 293, 894, 246, 1440, 1200, 1050, 3193,
    763, 2620, 9296, 7800, 3625, 170, 3800, 784, 1000, 500, 601, 440
])

print("--- Етап 1: Первинний аналіз (за припущенням нормальності) ---")

n = len(data)
x_mean = np.mean(data)
s2 = np.var(data, ddof=1)
s = np.sqrt(s2)

print(f"Кількість елементів у вибірці: {n}")
print(f"Середнє арифметичне: {x_mean:.3f}")
print(f"Незміщена дисперсія (S^2): {s2:.3f}")
print(f"Стандартне відхилення (S): {s:.3f}")

plt.figure(figsize=(10, 6))
plt.hist(data, bins=15, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Гістограма')
xmin, xmax = plt.xlim()
x_vals = np.linspace(xmin, xmax, 200)
p_norm = stats.norm.pdf(x_vals, x_mean, s)
plt.plot(x_vals, p_norm, 'r', linewidth=2, label='Нормальний розподіл')
plt.title("Гістограма та нормальний розподіл")
plt.xlabel("Значення")
plt.ylabel("Щільність ймовірності")
plt.legend()
plt.show()

shapiro_stat, shapiro_p = stats.shapiro(data)
print(f"\nРезультати тесту Шапіро-Уїлка: W = {shapiro_stat:.3f}, p-value = {shapiro_p:.6f}")
if shapiro_p > 0.05:
    print("Оскільки p-value > 0.05, гіпотезу про нормальність не відхиляємо.")
else:
    print("Оскільки p-value < 0.05, гіпотезу про нормальність відхилено. Розподіл не є нормальним.")

print("\nРозрахуємо 95% довірчі інтервали (за 'нормальними' формулами):")
alpha = 0.05
t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
ci_mean = (x_mean - t_crit * s / np.sqrt(n), x_mean + t_crit * s / np.sqrt(n))
print(f"Для середнього: ({ci_mean[0]:.3f}; {ci_mean[1]:.3f})")

chi2_lower = stats.chi2.ppf(alpha / 2, n - 1)
chi2_upper = stats.chi2.ppf(1 - alpha / 2, n - 1)
ci_sigma = (np.sqrt((n - 1) * s2 / chi2_upper), np.sqrt((n - 1) * s2 / chi2_lower))
print(f"Для стандартного відхилення: ({ci_sigma[0]:.3f}; {ci_sigma[1]:.3f})")

outliers_3sigma = [x for x in data if abs(x - x_mean) > 3 * s]
print(f"\nПошук викидів за правилом трьох сигм: {outliers_3sigma if outliers_3sigma else 'немає'}")


print("\n\n--- Етап 2: Аналіз з використанням робастних методів (для ненормальних даних) ---")

median = np.median(data)
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1

print(f"Медіана: {median:.3f}")
print(f"Міжквартильний розмах (IQR): {iqr:.3f}")

plt.figure(figsize=(10, 4))
plt.boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor='skyblue'))
plt.title('Діаграма "Ящик з вусами" для виявлення викидів')
plt.xlabel("Значення")
plt.grid(True)
plt.show()

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers_iqr = [x for x in data if x < lower_bound or x > upper_bound]
print(f"\nПошук викидів за методом Тьюкі (на основі IQR): {outliers_iqr if outliers_iqr else 'немає'}")
print(f"Межі для визначення викидів: [ {lower_bound:.3f} ; {upper_bound:.3f} ]")

data_cleaned = [x for x in data if not (x < lower_bound or x > upper_bound)]
print(f"\nРозмір вибірки після видалення викидів: {len(data_cleaned)}")

if outliers_iqr:
    shapiro_stat_c, shapiro_p_c = stats.shapiro(data_cleaned)
    print(f"Повторний тест Шапіро-Уїлка для очищених даних: p-value = {shapiro_p_c:.6f}")
    if shapiro_p_c > 0.05:
        print("Тепер p-value > 0.05, отже, очищені дані можна вважати нормально розподіленими.")
    else:
        print("Навіть після видалення викидів p-value < 0.05, розподіл все ще не є нормальним.")