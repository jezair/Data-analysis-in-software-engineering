import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# === Вхідні дані ===
x_bar = 2496       # вибіркове середнє
S2 = 6.4           # вибіркова дисперсія
n = 30             # обсяг вибірки (тут треба вказати свій!)

# === 1. Вибіркове середнє та дисперсія ===
print(f"Вибіркове середнє: {x_bar}")
print(f"Вибіркова дисперсія: {S2}")

# === 2. Стандартне відхилення ===
S = np.sqrt(S2)
print(f"Вибіркове стандартне відхилення: {S:.4f}")

# === 3. Гістограма (тут потрібні дані, але зімітуємо нормально розподілені) ===
# ⚠️ Зверни увагу: це просто імітація даних для ілюстрації
np.random.seed(42)
sample = np.random.normal(loc=x_bar, scale=S, size=n)

plt.hist(sample, bins=10, density=True, alpha=0.6, color='skyblue', edgecolor='black')

# Накладемо нормальну щільність
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
pdf = st.norm.pdf(x, x_bar, S)
plt.plot(x, pdf, 'r', linewidth=2)
plt.title("Гістограма + нормальний розподіл")
plt.show()

# === 4. Довіpчий інтервал для середнього ===
alpha = 0.05
df = n - 1
t_crit = st.t.ppf(1 - alpha/2, df)

delta = t_crit * S / np.sqrt(n)
ci_mean = (x_bar - delta, x_bar + delta)
print(f"95% довірчий інтервал для середнього: {ci_mean}")

# === 5. Довірчий інтервал для дисперсії і стандартного відхилення ===
chi2_lower = st.chi2.ppf(alpha/2, df)
chi2_upper = st.chi2.ppf(1 - alpha/2, df)

ci_var = ((df * S2) / chi2_upper, (df * S2) / chi2_lower)
ci_std = (np.sqrt(ci_var[0]), np.sqrt(ci_var[1]))

print(f"95% довірчий інтервал для дисперсії: {ci_var}")
print(f"95% довірчий інтервал для стандартного відхилення: {ci_std}")
