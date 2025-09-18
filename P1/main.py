import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1) Вибірка (твої дані)
data = np.array([
    6020, 3570, 2496, 191, 406, 14520, 14430, 650, 225, 183,
    1080, 2464, 1041, 528, 212, 7060, 903, 4164, 633, 665,
    544, 5732, 1238, 1586, 281, 220, 851, 2391, 780, 591,
    391, 2263, 4112, 293, 894, 246, 1440, 1200, 1050, 3193,
    763, 2620, 9296, 7800, 3625, 170, 3800, 784, 1000, 500, 601, 440
])


# 2) Вибіркове середнє та дисперсія
n = len(data)
x_mean = np.mean(data)
s2 = np.var(data, ddof=1)  # незміщена дисперсія
s = np.sqrt(s2)

print(f"Кількість спостережень: {n}")
print(f"Вибіркове середнє: {x_mean:.3f}")
print(f"Вибіркова дисперсія: {s2:.3f}")
print(f"Середньоквадратичне відхилення: {s:.3f}")



# 3) Гістограма + нормальний розподіл
plt.hist(data, bins=15, density=True, alpha=0.6, color='skyblue', edgecolor='black')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 200)
p = stats.norm.pdf(x, x_mean, s)
plt.plot(x, p, 'r', linewidth=2)
plt.title("Гістограма та нормальний розподіл")
plt.xlabel("X")
plt.ylabel("Щільність ймовірності")
plt.show()



# 4) Перевірка нормальності (Шапіро-Уїлк)
shapiro_test = stats.shapiro(data)
print(f"Тест Шапіро-Уїлка: статистика={shapiro_test[0]:.3f}, p-value={shapiro_test[1]:.6f}")
if shapiro_test[1] > 0.05:
    print("Дані можна вважати нормально розподіленими.")
else:
    print("Дані НЕ є нормально розподіленими.")



# 5) Довірчі інтервали (95%)
alpha = 0.05

# для середнього
t_crit = stats.t.ppf(1 - alpha/2, n-1)
ci_mean = (x_mean - t_crit * s / np.sqrt(n), x_mean + t_crit * s / np.sqrt(n))

# для σ
chi2_lower = stats.chi2.ppf(alpha/2, n-1)
chi2_upper = stats.chi2.ppf(1 - alpha/2, n-1)
ci_sigma = (np.sqrt((n-1)*s2/chi2_upper), np.sqrt((n-1)*s2/chi2_lower))

print(f"95% довірчий інтервал для середнього: {ci_mean}")
print(f"95% довірчий інтервал для σ: {ci_sigma}")



# 6) Пошук викидів (метод 3σ)
outliers = [x for x in data if abs(x - x_mean) > 3*s]
print(f"Викиди: {outliers if outliers else 'не знайдено'}")


print("\nВисновок:")
print(f"- Середнє значення ≈ {x_mean:.2f}, дисперсія ≈ {s2:.2f}, σ ≈ {s:.2f}.")
if shapiro_test[1] > 0.05:
    print("- Розподіл можна вважати нормальним.")
else:
    print("- Розподіл НЕ є нормальним, тому результати інтервалів слід інтерпретувати обережно.")
if outliers:
    print(f"- Виявлено викиди: {outliers}")
else:
    print("- Викиди не виявлені.")
