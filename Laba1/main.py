import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#1 Вычисление информационной и дисперсионной матриц для заданного плана экспери-мента
# Квадратичная модель: f(x) = [1, x, x^2]^T, количество параметров модели = 3
def get_regression_basis_vector(x_value):
    """Возвращает вектор базисных функций для точки x."""
    return np.array([1.0, x_value, x_value**2])

def calculate_information_and_dispersion_matrices(design_points, de-sign_weights):
    # Вычисляет информационную матрицу Фишера и дисперсионную матрицу .
    information_matrix = np.zeros((3, 3))
    
    for current_point, current_weight in zip(design_points, design_weights):
        # Превращаем вектор в столбец
        basis_vector_column = get_regression_basis_vector(current_point).reshape(-1, 1)
        # Накапливаем сумму: вес точки * (вектор-столбец умноженный на вектор-строку)
        information_matrix += current_weight * (basis_vector_column @ ba-sis_vector_column.T)
     # Вычисляем дисперсионную матрицу как обратную информационной матрицы 
    dispersion_matrix = np.linalg.inv(information_matrix)
    return information_matrix, dispersion_matrix

def calculate_optimality_criteria(information_matrix, dispersion_matrix):
    """Вычисляет 7 критериев оптимальности для заданных матриц."""
    
    # D-критерий: Определитель информационной матрицы (максимизируется)
    d_criterion_value = float(np.real(np.linalg.det(information_matrix)))
    
    # A-критерий: След дисперсионной матрицы (минимизируется)
    a_criterion_value = float(np.real(np.trace(dispersion_matrix)))
    
    # E-критерий: Минимальное собственное число информационной матрицы (максими-зируется)
    min_eigenvalue_of_inf_matrix = float(np.min(np.linalg.eigvalsh(information_matrix)))
    
    # Phi_2-критерий: Корень из усредненного следа квадрата дисперсионной матри-цы (минимизируется)
    phi_2_criterion_value = float(np.real(np.sqrt((1.0 / 3.0) * np.trace(dispersion_matrix @ dispersion_matrix))))
    
    # Л-критерий: Разброс (сумма квадратов отклонений) собственных чисел диспер-сионной матрицы (минимизируется)
    eigenvalues_of_dispersion_matrix = np.linalg.eigvalsh(dispersion_matrix)
    mean_eigenvalue = np.mean(eigenvalues_of_dispersion_matrix)
    l_criterion_value = float(np.sum((eigenvalues_of_dispersion_matrix - mean_eigenvalue)**2))
    
    # MV-критерий: Максимальная дисперсия оценки отдельного параметра (макси-мальный элемент на диагонали D)
    mv_criterion_value = float(np.max(np.diag(dispersion_matrix)))
    
    # G-критерий: Максимальная дисперсия прогноза функции на области планирова-ния [-1, 1] (минимизируется)
    grid_of_x_values = np.linspace(-1, 1, 500)
    prediction_variances_on_grid = []

        # Проходим по каждой точке x из нашей сетки
    for x_val in grid_of_x_values:
        vector = get_regression_basis_vector(x_val)
        error_in_point = vector.T @ dispersion_matrix @ vector 
        prediction_variances_on_grid.append(float(error_in_point))
    
    # В prediction_variances_on_grid лежат ошибки для всех 500 точек.
    
    # Ищем самую большую ошибку (максимум) среди всех точек
    g_criterion_value = float(np.max(prediction_variances_on_grid))
    
    # Возвращаем результаты
    return {
        "|M| (D-крит) ↑": d_criterion_value,
        "tr(D) (A-крит) ↓": a_criterion_value,
        "min λ(M) (E-крит) ↑": min_eigenvalue_of_inf_matrix,
        "Phi_2 ↓": phi_2_criterion_value,
        "Л-крит ↓": l_criterion_value,
        "MV-крит ↓": mv_criterion_value,
        "G-крит ↓": g_criterion_value
    }

# 2. Расчет критериев для Планов № 1–4
experiment_plans = {
    "План 1": {"points": [-1, 0, 1], "weights": [0.2, 0.6, 0.2]},
    "План 2": {"points": [-1, 0, 1], "weights": [0.25, 0.5, 0.25]},
    "План 3": {"points": [-1, 0, 1], "weights": [0.1884, 0.6233, 0.1884]},
    "План 4": {"points": [-1, 0, 1], "weights": [1/3, 1/3, 1/3]}
}

criteria_results_for_plans = {}
for plan_name, plan_data in experiment_plans.items():
    current_inf_matrix, current_disp_matrix = calcu-late_information_and_dispersion_matrices(
        design_points=plan_data["points"], 
        design_weights=plan_data["weights"]
    )
    criteria_results_for_plans[plan_name] = calcu-late_optimality_criteria(current_inf_matrix, current_disp_matrix)

dataframe_of_results = pd.DataFrame(criteria_results_for_plans).T

# 3. Ранжирование планов

# Мало True - Хорошо, много False - Хорошо 
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
    dataframe_of_ranks[criterion_name] = data-frame_of_results[criterion_name].rank(
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

# 4. Анализ D-оптимальности как функции от q

# Спектр: [-1, 0, 1], вес левой точки = q, вес центра = 1 - 2q, вес правой точки = q
# Допустимый диапазон веса q: от 0 до 0.5
array_of_q_weights = np.linspace(0.01, 0.49, 500)
determinant_values_for_q = []
# Перебираем все возможные варианты веса q (доли попыток для крайних точек)
for current_q_weight in array_of_q_weights:
    
    # Вычисляем информационную матрицу 
    inf_matrix_for_current_q, _ = calculate_information_and_dispersion_matrices(
        design_points=[-1, 0, 1], # Точки измерений: левая, центр, правая
        # Задаем веса: крайним по 'q', а центру — всё, что осталось 
        design_weights=[current_q_weight, 1 - 2 * current_q_weight, cur-rent_q_weight] 
    )
    
    # Считаем определитель (оценку качества плана) и добавляем в список
    determinant_values_for_q.append(np.linalg.det(inf_matrix_for_current_q))

# Преобразуем обычный список в массив NumPy для быстрого поиска
determinant_values_for_q = np.array(determinant_values_for_q)

# Находим позицию (индекс) самого большого значения определителя
index_of_max_determinant = np.argmax(determinant_values_for_q)

# По найденному индексу достаем идеальный вес 'q', который дал этот максимум...
optimal_q_weight = array_of_q_weights[index_of_max_determinant]

# само максимальное значение определителя
maximum_determinant_value = determinant_values_for_q[index_of_max_determinant]

print("\n" + "=" * 80)
print(f"Оптимум D-критерия: q* = {optimal_q_weight:.4f} с |M(q*)| = {maxi-mum_determinant_value:.4f}")
print("Теоретический оптимум: q = 0.25 (соответствует Плану № 4)")
print("=" * 80)

# Построение графика
plt.figure(figsize=(9, 5), dpi=100)
plt.plot(array_of_q_weights, determinant_values_for_q, label=r"$|M(q)|$", col-or="blue", linewidth=2)
plt.axvline(
    optimal_q_weight, 
    color="red", 
    linestyle="--", 
    label=f"Максимум: $q^* \\approx {optimal_q_weight:.2f}$ (План № 4)"
)
plt.scatter([optimal_q_weight], [maximum_determinant_value], color="red", zorder=5)

# Отметка остальных табличных планов
known_plans_q_values = {"План 1": 0.2, "План 2": 0.25, "План 3": 0.1884, "План 4": 1/3}
for known_plan_name, known_plan_q in known_plans_q_values.items():
    inf_matrix_for_known_plan, _ = calcu-late_information_and_dispersion_matrices(
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
