import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# === Крок 0: Підготовка (Створення демонстраційних даних) ===
np.random.seed(42)
N = 50

X_orig = np.sort(np.random.uniform(5, 100, N))

Y_orig = 5 * (X_orig ** 0.75) * np.exp(np.random.normal(0, 0.25, N))

Y_orig = Y_orig + np.random.normal(0, 3, N)

Y_orig[Y_orig <= 0] = 1.0


def calculate_metrics(y_true, y_pred):
    valid_indices = y_true > 1e-9
    if not np.any(valid_indices):
        return {'R2': 0, 'MMRE': np.inf, 'PRED(0.25)': 0}

    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]

    mre = np.abs(y_true_valid - y_pred_valid) / y_true_valid

    mmre = np.mean(mre)

    pred25 = np.mean(mre <= 0.25) * 100

    r2 = r2_score(y_true, y_pred)

    return {"R2": r2, "MMRE": mmre, "PRED(0.25)": pred25}


def plot_results(x_orig, y_orig, y_pred_nonlin,
                 x_norm, y_norm, y_pred_norm,
                 transform_name):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Аналіз з перетворенням: {transform_name}", fontsize=16)

    # Графік 1: Нормалізовані дані та лінійна регресія
    ax1.scatter(x_norm, y_norm, alpha=0.7, label="Нормалізовані дані")
    ax1.plot(x_norm, y_pred_norm, color='red',
             label="Лінійна регресія (в норм. просторі)")
    ax1.set_title("Крок 3: Лінійна регресія (Нормалізовані дані)")
    ax1.set_xlabel("X (нормалізований)")
    ax1.set_ylabel("Y (нормалізований)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Графік 2: Вихідні дані та нелінійна регресія
    sort_indices = np.argsort(x_orig)
    ax2.scatter(x_orig, y_orig, alpha=0.7, label="Емпіричні дані (вихідні)")
    ax2.plot(x_orig[sort_indices], y_pred_nonlin[sort_indices], color='green',
             linewidth=2, label="Нелінійна регресія (відновлена)")
    ax2.set_title("Крок 6: Нелінійна регресія (Вихідні дані)")
    ax2.set_xlabel("X (вихідні дані)")
    ax2.set_ylabel("Y (вихідні дані)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def run_analysis(x_orig, y_orig, transform_funcs, transform_name):
    print(f"\n{'=' * 60}")
    print(f"ПОЧАТОК АНАЛІЗУ З ПЕРЕТВОРЕННЯМ: {transform_name}")
    print(f"{'=' * 60}")

    # === Крок 1: Вибір перетворення ===
    print(f"Крок 1: Обрано перетворення '{transform_name}'")
    x_transform = transform_funcs['x_func']
    y_transform = transform_funcs['y_func']
    y_inverse = transform_funcs['y_inv']

    # === Крок 2: Нормалізація емпіричних даних ===
    try:
        X_norm = x_transform(x_orig)
        Y_norm = y_transform(y_orig)
        print("Крок 2: Дані успішно нормалізовано.")
    except ValueError as e:
        print(f"Крок 2: Помилка нормалізації! {e}. Спробуйте інше перетворення.")
        return None

    # === Крок 3: Побудова лінійного рівняння регресії ===
    model_norm = LinearRegression()
    model_norm.fit(X_norm.reshape(-1, 1), Y_norm)

    b0 = model_norm.intercept_
    b1 = model_norm.coef_[0]
    print("Крок 3: Побудовано лінійну регресію для нормалізованих даних.")
    print(f"       Рівняння: Y_norm = {b0:.4f} + {b1:.4f} * X_norm")

    # === Крок 4: Визначення значень відхилення (залишків) ===
    Y_norm_pred = model_norm.predict(X_norm.reshape(-1, 1))
    residuals = Y_norm - Y_norm_pred
    print(f"Крок 4: Залишки (відхилення) розраховано. (Середнє: {np.mean(residuals):.2e})")

    # === Крок 5: Перевірка гіпотези про нормальність ===
    alpha = 0.05
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print("Крок 5: Перевірка залишків на нормальність (тест Шапіро-Вілка):")
    print(f"       Статистика = {shapiro_stat:.4f}, p-value = {shapiro_p:.4f}")

    if shapiro_p <= alpha:
        print(f"       Результат: Гіпотеза про нормальність ВІДХИЛЯЄТЬСЯ (p <= {alpha}).")
        return None

    print(f"       Результат: Гіпотеза про нормальність ПРИЙМАЄТЬСЯ (p > {alpha}).")

    # === Крок 6: Побудова нелінійного рівняння регресії ===
    Y_nonlin_pred = y_inverse(Y_norm_pred)
    print("Крок 6: Побудовано нелінійне рівняння (шляхом зворотного перетворення).")

    if transform_name == 'Log-Log':
        a = np.exp(b0)
        print(f"       Форма рівняння: Y = {a:.4f} * X^{b1:.4f} (Степенева модель)")
    elif transform_name == 'Sqrt-Sqrt':
        print(f"       Форма рівняння: Y = ({b0:.4f} + {b1:.4f} * sqrt(X))^2 (Квадратична модель)")
    elif transform_name == 'Log-Linear' and transform_funcs['x_func'] == 'identity':
        a = np.exp(b0)
        b = np.exp(b1)
        print(f"       Форма рівняння: Y = {a:.4f} * ({b:.4f})^X (Експоненційна модель)")

    # === Крок 7: Визначення R2, MMRE, PRED(0,25) ===
    metrics = calculate_metrics(y_orig, Y_nonlin_pred)
    print("Крок 7: Розрахунок метрик якості для нелінійної моделі:")
    print(f"       R^2 = {metrics['R2']:.4f}")
    print(f"       MMRE = {metrics['MMRE']:.4f}")
    print(f"       PRED(0.25) = {metrics['PRED(0.25)']:.2f}%")

    # === Крок 8: Побудова лінії регресії та емпіричних даних ===
    print("Крок 8: Побудова графіків...")
    plot_results(x_orig, y_orig, Y_nonlin_pred,
                 X_norm, Y_norm, Y_norm_pred,
                 transform_name)

    metrics['model_name'] = transform_name

    return metrics


# ======================================================================
# ===                         САМЕ ЗАВДАННЯ                          ===
# ======================================================================

transforms_log_log = {
    'x_func': np.log,  # log(X)
    'y_func': np.log,  # log(Y)
    'y_inv': np.exp  # exp(Y_norm_pred)
}
results_log = run_analysis(X_orig, Y_orig, transforms_log_log, "Log-Log (Степенева)")

transforms_sqrt_sqrt = {
    'x_func': np.sqrt,  # sqrt(X)
    'y_func': np.sqrt,  # sqrt(Y)
    'y_inv': lambda y: y ** 2  # (Y_norm_pred)^2
}
results_sqrt = run_analysis(X_orig, Y_orig, transforms_sqrt_sqrt, "Sqrt-Sqrt (Квадратична)")

# --- Крок 9: Висновок про якість у порівнянні з лінійним з лр №4 ---

print(f"\n{'=' * 60}")
print("КРОК 9: ПОРІВНЯННЯ РЕЗУЛЬТАТІВ ТА ВИСНОВКИ")
print(f"{'=' * 60}")

print("Для порівняння, порахуємо спочатку просту лінійну модель (як в л.р.No4).")
baseline_model = LinearRegression()
baseline_model.fit(X_orig.reshape(-1, 1), Y_orig)
Y_baseline_pred = baseline_model.predict(X_orig.reshape(-1, 1))
results_baseline = calculate_metrics(Y_orig, Y_baseline_pred)
print("... Розрахунок завершено.")

# Виведення порівняльної таблиці
print("\nПорівняльна таблиця метрик якості моделей:")
print("-" * 75)
print(f"| {'Модель':<30} | {'R^2 (більше = краще)':>20} | {'MMRE (менше = краще)':>20} |")
print("-" * 75)
print(f"| {'Проста Лінійна (з л.р.No4)':<30} | {results_baseline['R2']:>20.4f} | {results_baseline['MMRE']:>20.4f} |")

if results_log:
    print(f"| {results_log['model_name']:<30} | {results_log['R2']:>20.4f} | {results_log['MMRE']:>20.4f} |")
else:
    print(f"| {'Log-Log (Степенева)':<30} | {'(Тест не пройдено)':>43} |")

if results_sqrt:
    print(f"| {results_sqrt['model_name']:<30} | {results_sqrt['R2']:>20.4f} | {results_sqrt['MMRE']:>20.4f} |")
else:
    print(f"| {'Sqrt-Sqrt (Квадратична)':<30} | {'(Тест не пройдено)':>43} |")
print("-" * 75)

print("\n### Висновок по роботі: ###")

print("В ході виконання роботи було побудовано кілька моделей.")
print(f"1. Базова лінійна модель (з л.р. №4) показала себе погано.")
print(
    f"   Її R^2 = {results_baseline['R2']:.3f} (дуже низький), а похибка MMRE = {results_baseline['MMRE']:.3f} (дуже висока).")
print("   Це означає, що вона погано описує дані.")

if results_log and (not results_sqrt or results_log['R2'] > results_sqrt['R2']):
    print("\n2. Модель з Log-Log перетворенням (степенева) пройшла тест Шапіро-Вілка,")
    print("   тобто залишки нормальні (p > 0.05).")
    print(f"   Ця модель дала набагато кращі результати:")
    print(f"   R^2 зріс до {results_log['R2']:.3f}, а MMRE впав до {results_log['MMRE']:.3f}.")

    if not results_sqrt:
        print("\n3. Спроба з Sqrt-Sqrt перетворенням (самостійна робота) не вдалася,")
        print("   оскільки тест на нормальність залишків провалився (p <= 0.05).")
        print("   Тому цю модель ми не можемо використовувати.")
    elif results_sqrt:
        print("\n3. Модель Sqrt-Sqrt (самостійна робота) теж пройшла тест, але її показники")
        print(f"   (R^2={results_sqrt['R2']:.3f}, MMRE={results_sqrt['MMRE']:.3f}) гірші, ніж у Log-Log.")

    print("\nЗагальний висновок: Найкращою моделлю для цих даних є нелінійна")
    print("степенева модель (Log-Log). Це доводить, що нормалізація даних")
    print("дозволяє будувати значно точніші прогнози, ніж проста лінійна регресія.")

elif results_sqrt:
    if results_log:
        print("\n2. Модель з Log-Log перетворенням пройшла тест, але показники середні.")
        print(f"   (R^2={results_log['R2']:.3f}, MMRE={results_log['MMRE']:.3f}).")
    else:
        print("\n2. Модель з Log-Log перетворенням тест не пройшла.")

    print("\n3. Модель з Sqrt-Sqrt перетворенням (самостійна робота) теж пройшла тест")
    print("   і показала найкращі результати:")
    print(f"   R^2 = {results_sqrt['R2']:.3f} (найвищий), MMRE = {results_sqrt['MMRE']:.3f} (найнижчий).")

    print("\nЗагальний висновок: Найкращою моделлю для цих даних є нелінійна")
    print("модель Sqrt-Sqrt. Це доводить, що для отримання точного прогнозу")
    print("важливо правильно підібрати нормалізуюче перетворення.")

else:
    print("\n2. Обидві спроби (Log-Log та Sqrt-Sqrt) не пройшли тест на нормальність.")
    print("   Це означає, що жодне з обраних перетворень не підходить для")
    print("   цих даних і потрібно було б шукати інше (наприклад, Box-Cox).")
    print("   Порівнювати їх з лінійною моделлю некоректно.")