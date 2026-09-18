import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. Определение модели и вспомогательных функций
# ==========================================
# Квадратичная модель: f(x) = [1, x, x^2]^T, m = 3
def f(x):
    return np.array([1.0, x, x**2])

def calc_matrices(points, weights):
    """Вычисляет информационную матрицу M и дисперсионную матрицу D."""
    M = np.zeros((3, 3))
    for x_i, p_i in zip(points, weights):
        f_vec = f(x_i).reshape(-1, 1)
        M += p_i * (f_vec @ f_vec.T)
    D = np.linalg.inv(M)
    return M, D

def calc_criteria(M, D):
    """Вычисляет 7 критериев оптимальности для заданных M и D."""
    # D-критерий: |M| (максимизируется)
    det_M = float(np.real(np.linalg.det(M)))
    
    # A-критерий: tr(D) (минимизируется)
    tr_D = float(np.real(np.trace(D)))
    
    # E-критерий: min lambda(M) (максимизируется)
    # Матрица M симметрична, поэтому eigvalsh надежнее и сразу возвращает float
    min_eig_M = float(np.min(np.linalg.eigvalsh(M)))
    
    # Phi_2-критерий: sqrt(1/m * tr(D^2)) (минимизируется)
    phi_2 = float(np.real(np.sqrt((1.0 / 3.0) * np.trace(D @ D))))
    
    # Л-критерий: sum((lambda_i - mean)^2) (минимизируется)
    eig_D = np.linalg.eigvalsh(D)
    crit_L = float(np.sum((eig_D - np.mean(eig_D))**2))
    
    # MV-критерий: max D_ii (минимизируется)
    mv_crit = float(np.max(np.diag(D)))
    
    # G-критерий: max d(x), d(x) = f(x)^T * D * f(x) (минимизируется)
    x_grid = np.linspace(-1, 1, 500)
    d_vals = [float(f(x_val).T @ D @ f(x_val)) for x_val in x_grid]
    g_crit = float(np.max(d_vals))
    
    return {
        "|M| (D-крит) ↑": det_M,
        "tr(D) (A-крит) ↓": tr_D,
        "min λ(M) (E-крит) ↑": min_eig_M,
        "Phi_2 ↓": phi_2,
        "Л-крит ↓": crit_L,
        "MV-крит ↓": mv_crit,
        "G-крит ↓": g_crit
    }

# ==========================================
# 2. Расчет критериев для Планов № 1–4
# ==========================================
plans = {
    "План 1": {"points": [-1, 0, 1], "weights": [0.2, 0.6, 0.2]},
    "План 2": {"points": [-1, 0, 1], "weights": [0.25, 0.5, 0.25]},
    "План 3": {"points": [-1, 0, 1], "weights": [0.1884, 0.6233, 0.1884]},
    "План 4": {"points": [-1, 0, 1], "weights": [1/3, 1/3, 1/3]}
}

results = {}
for name, plan in plans.items():
    M, D = calc_matrices(plan["points"], plan["weights"])
    results[name] = calc_criteria(M, D)

df_results = pd.DataFrame(results).T

# ==========================================
# 3. Ранжирование планов
# ==========================================
# Направление оптимизации (True = чем меньше, тем лучше; False = чем больше, тем лучше)
criteria_direction = {
    "|M| (D-крит) ↑": False,
    "tr(D) (A-крит) ↓": True,
    "min λ(M) (E-крит) ↑": False,
    "Phi_2 ↓": True,
    "Л-крит ↓": True,
    "MV-крит ↓": True,
    "G-крит ↓": True
}

df_ranks = pd.DataFrame(index=df_results.index)
for col, ascending in criteria_direction.items():
    df_ranks[col] = df_results[col].rank(ascending=ascending, method="min").astype(int)

print("=" * 80)
print(" ЗНАЧЕНИЯ КРИТЕРИЕВ ДЛЯ ПЛАНОВ 1-4")
print("=" * 80)
print(df_results.round(5).to_string())

print("\n" + "=" * 80)
print(" РАНГИ ПЛАНОВ (1 — лучший, 4 — худший)")
print("=" * 80)
print(df_ranks.to_string())

# ==========================================
# 4. Анализ D-оптимальности как функции от q
# ==========================================
# Спектр: [-1, 0, 1], p_1 = q, p_2 = 1 - 2q, p_3 = q
# Допустимый диапазон: q in (0, 0.5)
q_vals = np.linspace(0.01, 0.49, 500)
det_M_vals = []

for q in q_vals:
    M_q, _ = calc_matrices([-1, 0, 1], [q, 1 - 2*q, q])
    det_M_vals.append(np.linalg.det(M_q))

det_M_vals = np.array(det_M_vals)
best_idx = np.argmax(det_M_vals)
q_opt = q_vals[best_idx]
max_det = det_M_vals[best_idx]

print("\n" + "=" * 80)
print(f"Оптимум D-критерия: q* = {q_opt:.4f} с |M(q*)| = {max_det:.4f}")
print("Теоретический оптимум: q = 0.25 (соответствует Плану № 2)")
print("=" * 80)

# Построение графика
plt.figure(figsize=(9, 5), dpi=100)
plt.plot(q_vals, det_M_vals, label=r"$|M(q)|$", color="blue", linewidth=2)
plt.axvline(q_opt, color="red", linestyle="--", label=f"Максимум: $q^* \\approx {q_opt:.2f}$ (План № 2)")
plt.scatter([q_opt], [max_det], color="red", zorder=5)

# Отметка остальных табличных планов
q_table = {"План 1": 0.2, "План 2": 0.25, "План 3": 0.1884, "План 4": 1/3}
for p_name, p_q in q_table.items():
    p_M, _ = calc_matrices([-1, 0, 1], [p_q, 1 - 2*p_q, p_q])
    p_det = np.linalg.det(p_M)
    plt.scatter([p_q], [p_det], s=40, zorder=4)
    plt.annotate(p_name, (p_q, p_det), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

plt.title(r"Зависимость D-критерия $|M(q)|$ от параметра весов $q \in (0, 0.5)$", fontsize=12)
plt.xlabel("Параметр весов $q$", fontsize=11)
plt.ylabel(r"Определитель $|M(q)|$", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()