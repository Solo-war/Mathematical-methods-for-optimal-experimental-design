import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. Определение модели и вспомогательных функций
# ==========================================
# Квадратичная модель: f(x) = [1, x, x^2]^T, количество параметров модели = 3
def get_regression_basis_vector(x_value):
    """Возвращает вектор базисных функций для точки x."""
    return np.array([1.0, x_value, x_value**2])

def calculate_information_and_dispersion_matrices(design_points, design_weights):
    """Вычисляет информационную матрицу Фишера и дисперсионную матрицу (матрицу ковариаций)."""
    information_matrix = np.zeros((3, 3))
    
    for current_point, current_weight in zip(design_points, design_weights):
        basis_vector_column = get_regression_basis_vector(current_point).reshape(-1, 1)
        # Накапливаем сумму: вес точки * (вектор-столбец умноженный на вектор-строку)
        information_matrix += current_weight * (basis_vector_column @ basis_vector_column.T)
        
    dispersion_matrix = np.linalg.inv(information_matrix)
    return information_matrix, dispersion_matrix

def calculate_optimality_criteria(information_matrix, dispersion_matrix):
    """Вычисляет 7 критериев оптимальности для заданных матриц."""
    
    # D-критерий: Определитель информационной матрицы (максимизируется)
    d_criterion_value = float(np.real(np.linalg.det(information_matrix)))
    
    # A-критерий: След дисперсионной матрицы (минимизируется)
    a_criterion_value = float(np.real(np.trace(dispersion_matrix)))
    
    # E-критерий: Минимальное собственное число информационной матрицы (максимизируется)
    min_eigenvalue_of_inf_matrix = float(np.min(np.linalg.eigvalsh(information_matrix)))
    
    # Phi_2-критерий: Корень из усредненного следа квадрата дисперсионной матрицы (минимизируется)
    phi_2_criterion_value = float(np.real(np.sqrt((1.0 / 3.0) * np.trace(dispersion_matrix @ dispersion_matrix))))
    
    # Л-критерий: Разброс (сумма квадратов отклонений) собственных чисел дисперсионной матрицы (минимизируется)
    eigenvalues_of_dispersion_matrix = np.linalg.eigvalsh(dispersion_matrix)
    mean_eigenvalue = np.mean(eigenvalues_of_dispersion_matrix)
    l_criterion_value = float(np.sum((eigenvalues_of_dispersion_matrix - mean_eigenvalue)**2))
    
    # MV-критерий: Максимальная дисперсия оценки отдельного параметра (максимальный элемент на диагонали D)
    mv_criterion_value = float(np.max(np.diag(dispersion_matrix)))
    
    # G-критерий: Максимальная дисперсия прогноза функции на области планирования [-1, 1] (минимизируется)
    grid_of_x_values = np.linspace(-1, 1, 500)
    prediction_variances_on_grid = [
        float(get_regression_basis_vector(x_val).T @ dispersion_matrix @ get_regression_basis_vector(x_val)) 
        for x_val in grid_of_x_values
    ]
    g_criterion_value = float(np.max(prediction_variances_on_grid))
    
    return {
        "|M| (D-крит) ↑": d_criterion_value,
        "tr(D) (A-крит) ↓": a_criterion_value,
        "min λ(M) (E-крит) ↑": min_eigenvalue_of_inf_matrix,
        "Phi_2 ↓": phi_2_criterion_value,
        "Л-крит ↓": l_criterion_value,
        "MV-крит ↓": mv_criterion_value,
        "G-крит ↓": g_criterion_value
    }

# ==========================================
# 2. Расчет критериев для Планов № 1–4
# ==========================================
experiment_plans = {
    "План 1": {"points": [-1, 0, 1], "weights": [0.2, 0.6, 0.2]},
    "План 2": {"points": [-1, 0, 1], "weights": [0.25, 0.5, 0.25]},
    "План 3": {"points": [-1, 0, 1], "weights": [0.1884, 0.6233, 0.1884]},
    "План 4": {"points": [-1, 0, 1], "weights": [1/3, 1/3, 1/3]}
}

criteria_results_for_plans = {}
for plan_name, plan_data in experiment_plans.items():
    current_inf_matrix, current_disp_matrix = calculate_information_and_dispersion_matrices(
        design_points=plan_data["points"], 
        design_weights=plan_data["weights"]
    )
    criteria_results_for_plans[plan_name] = calculate_optimality_criteria(current_inf_matrix, current_disp_matrix)

dataframe_of_results = pd.DataFrame(criteria_results_for_plans).T

# ==========================================
# 3. Ранжирование планов
# ==========================================
# Направление оптимизации (True = чем меньше значение, тем лучше; False = чем больше значение, тем лучше)
is_smaller_better_for_criterion = {
    "|M| (D-крит) ↑": False,
    "tr(D) (A-крит) ↓": True,
    "min λ(M) (E-крит) ↑": False,
    "Phi_2 ↓": True,
    "Л-крит ↓": True,
    "MV-крит ↓": True,
    "G-крит ↓": True
}

dataframe_of_ranks = pd.DataFrame(index=dataframe_of_results.index)
for criterion_name, sort_ascending in is_smaller_better_for_criterion.items():
    dataframe_of_ranks[criterion_name] = dataframe_of_results[criterion_name].rank(
        ascending=sort_ascending, 
        method="min"
    ).astype(int)

print("=" * 80)
print(" ЗНАЧЕНИЯ КРИТЕРИЕВ ДЛЯ ПЛАНОВ 1-4")
print("=" * 80)
print(dataframe_of_results.round(5).to_string())

print("\n" + "=" * 80)
print(" РАНГИ ПЛАНОВ (1 — лучший, 4 — худший)")
print("=" * 80)
print(dataframe_of_ranks.to_string())

# ==========================================
# 4. Анализ D-оптимальности как функции от q
# ==========================================
# Спектр: [-1, 0, 1], вес левой точки = q, вес центра = 1 - 2q, вес правой точки = q
# Допустимый диапазон веса q: от 0 до 0.5
array_of_q_weights = np.linspace(0.01, 0.49, 500)
determinant_values_for_q = []

for current_q_weight in array_of_q_weights:
    inf_matrix_for_current_q, _ = calculate_information_and_dispersion_matrices(
        design_points=[-1, 0, 1], 
        design_weights=[current_q_weight, 1 - 2 * current_q_weight, current_q_weight]
    )
    determinant_values_for_q.append(np.linalg.det(inf_matrix_for_current_q))

determinant_values_for_q = np.array(determinant_values_for_q)

index_of_max_determinant = np.argmax(determinant_values_for_q)
optimal_q_weight = array_of_q_weights[index_of_max_determinant]
maximum_determinant_value = determinant_values_for_q[index_of_max_determinant]

print("\n" + "=" * 80)
print(f"Оптимум D-критерия: q* = {optimal_q_weight:.4f} с |M(q*)| = {maximum_determinant_value:.4f}")
print("Теоретический оптимум: q = 0.25 (соответствует Плану № 2)")
print("=" * 80)

# Построение графика
plt.figure(figsize=(9, 5), dpi=100)
plt.plot(array_of_q_weights, determinant_values_for_q, label=r"$|M(q)|$", color="blue", linewidth=2)
plt.axvline(
    optimal_q_weight, 
    color="red", 
    linestyle="--", 
    label=f"Максимум: $q^* \\approx {optimal_q_weight:.2f}$ (План № 2)"
)
plt.scatter([optimal_q_weight], [maximum_determinant_value], color="red", zorder=5)

# Отметка остальных табличных планов
known_plans_q_values = {"План 1": 0.2, "План 2": 0.25, "План 3": 0.1884, "План 4": 1/3}
for known_plan_name, known_plan_q in known_plans_q_values.items():
    inf_matrix_for_known_plan, _ = calculate_information_and_dispersion_matrices(
        design_points=[-1, 0, 1], 
        design_weights=[known_plan_q, 1 - 2 * known_plan_q, known_plan_q]
    )
    determinant_for_known_plan = np.linalg.det(inf_matrix_for_known_plan)
    
    plt.scatter([known_plan_q], [determinant_for_known_plan], s=40, zorder=4)
    plt.annotate(
        known_plan_name, 
        (known_plan_q, determinant_for_known_plan), 
        textcoords="offset points", 
        xytext=(0, 8), 
        ha="center", 
        fontsize=9
    )

plt.title(r"Зависимость D-критерия $|M(q)|$ от параметра весов $q \in (0, 0.5)$", fontsize=12)
plt.xlabel("Параметр весов $q$", fontsize=11)
plt.ylabel(r"Определитель $|M(q)|$", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()