import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Вычисление базисных функций, информационной и дисперсионной матриц
# Двухфакторная квадратичная модель: f(x) = [1, x1, x2, x1*x2, x1^2, x2^2]^T
# Количество параметров модели m = 6
def get_regression_basis_vector(point_2d):
    """Возвращает вектор базисных функций для двумерной точки x = (x1, x2)."""
    x1_val, x2_val = point_2d[0], point_2d[1]
    return np.array([
        1.0, 
        x1_val, 
        x2_val, 
        x1_val * x2_val, 
        x1_val**2, 
        x2_val**2
    ])


def generate_candidate_grid(grid_resolution):
    """Формирует дискретное множество X (сетку точек) на квадрате [-1, 1] x [-1, 1]."""
    axis_values = np.linspace(-1.0, 1.0, grid_resolution)
    grid_points = []
    for x1_val in axis_values:
        for x2_val in axis_values:
            grid_points.append([x1_val, x2_val])
    return np.array(grid_points)


def calculate_information_and_dispersion_matrices(design_points):
    """
    Вычисляет нормированную информационную матрицу Фишера M(e_N) 
    и дисперсионную матрицу D(e_N) для дискретного плана из N точек (веса p_i = 1/N).
    """
    number_of_observations = len(design_points)
    number_of_parameters = 6
    information_matrix = np.zeros((number_of_parameters, number_of_parameters))
    point_weight = 1.0 / number_of_observations

    for current_point in design_points:
        basis_vector_column = get_regression_basis_vector(current_point).reshape(-1, 1)
        information_matrix += point_weight * (basis_vector_column @ basis_vector_column.T)

    dispersion_matrix = np.linalg.inv(information_matrix)
    return information_matrix, dispersion_matrix


def generate_non_singular_initial_plan(candidate_grid_points, number_of_observations, random_seed=42):
    """
    Генерирует начальный невырожденный план эксперимента из N точек.
    Так как повторные наблюдения допускаются, выборка идет с возвращением.
    """
    random_generator = np.random.default_rng(random_seed)
    total_candidates = len(candidate_grid_points)

    while True:
        selected_indices = random_generator.choice(
            total_candidates, 
            size=number_of_observations, 
            replace=True
        )
        initial_plan_points = candidate_grid_points[selected_indices]
        
        # Проверяем, чтобы информационная матрица была невырожденной
        inf_matrix = np.zeros((6, 6))
        for pt in initial_plan_points:
            vec = get_regression_basis_vector(pt).reshape(-1, 1)
            inf_matrix += (1.0 / number_of_observations) * (vec @ vec.T)
            
        if np.linalg.matrix_rank(inf_matrix) == 6 and np.linalg.det(inf_matrix) > 1e-12:
            return initial_plan_points


# 2. Реализация алгоритма Федорова для синтеза D-оптимальных дискретных планов
def run_fedorov_exchange_algorithm(candidate_grid_points, number_of_observations, delta_threshold=1e-6):
    """
    Алгоритм Федорова замены точек для построения D-оптимального дискретного плана.
    Повторные наблюдения допускаются.
    """
    # Шаг 1. Выбираем невырожденный начальный план
    current_plan_points = generate_non_singular_initial_plan(
        candidate_grid_points, 
        number_of_observations
    ).copy()

    # Предварительно вычисляем базисные векторы для всех точек сетки (матрица F_grid размера K x 6)
    candidate_basis_matrix = np.array([get_regression_basis_vector(pt) for pt in candidate_grid_points])
    
    iteration_count = 0

    while True:
        # Вычисляем текущие матрицы M и D = M^(-1)
        information_matrix, dispersion_matrix = calculate_information_and_dispersion_matrices(current_plan_points)
        
        # Базисные векторы для точек текущего плана (матрица F_plan размера N x 6)
        plan_basis_matrix = np.array([get_regression_basis_vector(pt) for pt in current_plan_points])

        # Вычисляем дисперсии прогноза d(x, e_N) = f^T(x) * D * f(x) для всех кандидатов сетки
        d_candidates = np.sum((candidate_basis_matrix @ dispersion_matrix) * candidate_basis_matrix, axis=1)
        
        # Вычисляем d(x_j, e_N) для всех точек текущего плана
        d_plan = np.sum((plan_basis_matrix @ dispersion_matrix) * plan_basis_matrix, axis=1)
        
        # Вычисляем взаимные дисперсии d(x_j, x, e_N) = f^T(x_j) * D * f(x) (размер N x K)
        d_cross = plan_basis_matrix @ dispersion_matrix @ candidate_basis_matrix.T

        # Шаг 2. Вычисляем матрицу приращений Delta(x_j, x) для всех пар (x_j из плана, x из сетки)
        # Формула: Delta(x_j, x) = 1/N * (d(x) - d(x_j)) - 1/N^2 * (d(x)*d(x_j) - d^2(x, x_j))
        n_float = float(number_of_observations)
        delta_matrix = (
            (1.0 / n_float) * (d_candidates[None, :] - d_plan[:, None])
            - (1.0 / (n_float ** 2)) * (d_plan[:, None] * d_candidates[None, :] - d_cross ** 2)
        )

        # Находим пару индексов (j*, k*), дающую максимальное приращение Delta
        flat_max_index = np.argmax(delta_matrix)
        best_plan_idx, best_candidate_idx = np.unravel_index(flat_max_index, delta_matrix.shape)
        max_delta_value = float(delta_matrix[best_plan_idx, best_candidate_idx])

        # Шаг 3. Проверка условия останова: если максимальное приращение меньше порога delta
        if max_delta_value < delta_threshold:
            break

        # Шаг 4. Заменяем точку x_j в плане на лучшую точку x из сетки
        current_plan_points[best_plan_idx] = candidate_grid_points[best_candidate_idx]
        iteration_count += 1

    final_inf_matrix, final_disp_matrix = calculate_information_and_dispersion_matrices(current_plan_points)
    
    # Вычисляем характеристики полученного плана
    d_criterion_det = float(np.real(np.linalg.det(final_inf_matrix)))
    a_criterion_trace = float(np.real(np.trace(final_disp_matrix)))
    
    # Максимальная дисперсия прогноза на сетке (G-критерий)
    final_d_candidates = np.sum((candidate_basis_matrix @ final_disp_matrix) * candidate_basis_matrix, axis=1)
    g_criterion_max_d = float(np.max(final_d_candidates))

    return {
        "plan_points": current_plan_points,
        "det_M": d_criterion_det,
        "trace_D": a_criterion_trace,
        "max_d": g_criterion_max_d,
        "iterations": iteration_count
    }


def group_repeated_points_in_plan(plan_points):
    """Группирует совпадающие точки плана и подсчитывает кратность (число повторений r_i)."""
    rounded_points = np.round(plan_points, 4)
    unique_points, counts = np.unique(rounded_points, axis=0, return_counts=True)
    
    # Сортируем по убыванию числа наблюдений в точке
    sort_order = np.argsort(-counts)
    unique_points = unique_points[sort_order]
    counts = counts[sort_order]
    
    plan_table = pd.DataFrame({
        "x1": unique_points[:, 0],
        "x2": unique_points[:, 1],
        "r_i (наблюдений)": counts,
        "p_i (вес)": np.round(counts / len(plan_points), 4)
    })
    return plan_table


# 3. Проведение исследований для N = 20, 25, 30, 35, 40 на сетках 10x10 и 20x20
observation_numbers_list = [20, 25, 30, 35, 40]
grid_resolutions_dict = {
    "Сетка 10x10": 10,
    "Сетка 20x20": 20
}

all_experiments_results = {}
summary_comparison_rows = []

for grid_name, resolution in grid_resolutions_dict.items():
    all_experiments_results[grid_name] = {}
    candidate_grid = generate_candidate_grid(resolution)
    
    for n_obs in observation_numbers_list:
        result = run_fedorov_exchange_algorithm(
            candidate_grid_points=candidate_grid, 
            number_of_observations=n_obs
        )
        all_experiments_results[grid_name][n_obs] = result

# Формируем сводную таблицу сравнения планов и выбираем лучший для каждого N
for n_obs in observation_numbers_list:
    res_10 = all_experiments_results["Сетка 10x10"][n_obs]
    res_20 = all_experiments_results["Сетка 20x20"][n_obs]
    
    best_grid_name = "Сетка 20x20" if res_20["det_M"] >= res_10["det_M"] else "Сетка 10x10"
    
    summary_comparison_rows.append({
        "N": n_obs,
        "|M| (10x10) ↑": res_10["det_M"],
        "max d(x) (10x10) ↓": res_10["max_d"],
        "Итераций (10x10)": res_10["iterations"],
        "|M| (20x20) ↑": res_20["det_M"],
        "max d(x) (20x20) ↓": res_20["max_d"],
        "Итераций (20x20)": res_20["iterations"],
        "Лучший план": best_grid_name
    })

dataframe_summary = pd.DataFrame(summary_comparison_rows).set_index("N")

print("=" * 105)
print(" СРАВНЕНИЕ D-ОПТИМАЛЬНЫХ ДИСКРЕТНЫХ ПЛАНОВ (АЛГОРИТМ ФЕДОРОВА, ВАРИАНТ 1)")
print("=" * 105)
print(dataframe_summary.round(6).to_string())


# 4. Детальный вывод спектров лучших планов для каждого числа наблюдений N
print("\n" + "=" * 105)
print(" СПЕКТРЫ И ВЕСА ЛУЧШИХ ДИСКРЕТНЫХ ПЛАНОВ ДЛЯ КАЖДОГО N")
print("=" * 105)

for n_obs in observation_numbers_list:
    best_grid_name = dataframe_summary.loc[n_obs, "Лучший план"]
    best_result = all_experiments_results[best_grid_name][n_obs]
    grouped_plan_df = group_repeated_points_in_plan(best_result["plan_points"])
    
    print(f"\n---> Число наблюдений N = {n_obs} | Победитель: {best_grid_name} | |M| = {best_result['det_M']:.6f}")
    print(grouped_plan_df.to_string(index=False))


# 5. Визуализация: график роста D-критерия и расположение точек на факторном пространстве
fig = plt.figure(figsize=(14, 9), dpi=100)

# График 1: Сравнение |M| для сеток 10x10 и 20x20 в зависимости от N
ax_crit = plt.subplot2grid((2, 3), (0, 0), colspan=3)
det_10_values = [all_experiments_results["Сетка 10x10"][n]["det_M"] for n in observation_numbers_list]
det_20_values = [all_experiments_results["Сетка 20x20"][n]["det_M"] for n in observation_numbers_list]

ax_crit.plot(observation_numbers_list, det_10_values, marker="o", linewidth=2, label="Сетка 10x10", color="royalblue")
ax_crit.plot(observation_numbers_list, det_20_values, marker="s", linewidth=2, linestyle="--", label="Сетка 20x20", color="crimson")
ax_crit.axhline(0.011432, color="green", linestyle=":", alpha=0.8, label="Теоретический предел непрерывного плана (|M*| ≈ 0.01143)")
ax_crit.set_title("Зависимость определителя информационной матрицы |M(e_N)| от числа наблюдений N", fontsize=12)
ax_crit.set_xlabel("Число наблюдений N", fontsize=11)
ax_crit.set_ylabel("Определитель |M| (D-критерий)", fontsize=11)
ax_crit.set_xticks(observation_numbers_list)
ax_crit.grid(True, linestyle=":", alpha=0.6)
ax_crit.legend()

# Графики 2-4: Спектры лучших планов для N = 20, 30, 40
sample_n_for_plots = [20, 30, 40]
for idx, n_obs in enumerate(sample_n_for_plots):
    ax_plan = plt.subplot2grid((2, 3), (1, idx))
    best_grid_name = dataframe_summary.loc[n_obs, "Лучший план"]
    best_plan_points = all_experiments_results[best_grid_name][n_obs]["plan_points"]
    grouped_df = group_repeated_points_in_plan(best_plan_points)
    
    # Отрисовка узлов сетки
    res = grid_resolutions_dict[best_grid_name]
    grid_pts = generate_candidate_grid(res)
    ax_plan.scatter(grid_pts[:, 0], grid_pts[:, 1], color="lightgray", s=10, label="Узлы сетки", zorder=1)
    
    # Отрисовка точек плана (размер маркера пропорционален кратности наблюдений r_i)
    scatter_sizes = grouped_df["r_i (наблюдений)"] * 45
    ax_plan.scatter(grouped_df["x1"], grouped_df["x2"], color="darkorange", edgecolors="black", s=scatter_sizes, zorder=3)
    
    # Подписи кратности наблюдений r_i возле каждой точки
    for _, row in grouped_df.iterrows():
        ax_plan.annotate(
            f"r={int(row['r_i (наблюдений)'])}",
            (row["x1"], row["x2"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            fontweight="bold"
        )
        
    ax_plan.set_title(f"Лучший план при N = {n_obs}\n({best_grid_name}, |M|={all_experiments_results[best_grid_name][n_obs]['det_M']:.5f})", fontsize=10)
    ax_plan.set_xlabel("$x_1$", fontsize=10)
    ax_plan.set_ylabel("$x_2$", fontsize=10)
    ax_plan.set_xlim(-1.2, 1.2)
    ax_plan.set_ylim(-1.2, 1.2)
    ax_plan.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.show()