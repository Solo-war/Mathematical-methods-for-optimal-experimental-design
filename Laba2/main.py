import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Определение базисных функций и расчет матриц плана
# Двухфакторная полная квадратичная модель на квадрате [-1, 1]^2:
# f(x) = [1, x1, x2, x1^2, x1*x2, x2^2]^T, число неизвестных параметров m = 6
def get_regression_basis_vector(x1_value, x2_value):
    """Возвращает вектор базисных функций для точки плана (x1, x2)."""
    return np.array([1.0, x1_value, x2_value, x1_value**2, x1_value * x2_value, x2_value**2])


def calculate_information_and_dispersion_matrices(design_points, design_weights):
    """
    Вычисляет нормированную информационную матрицу Фишера M(epsilon)
    и дисперсионную матрицу D(epsilon) = M^(-1).
    """
    number_of_parameters = 6
    information_matrix = np.zeros((number_of_parameters, number_of_parameters))

    for (current_x1, current_x2), current_weight in zip(design_points, design_weights):
        # Превращаем вектор базисных функций в столбец
        basis_vector_column = get_regression_basis_vector(current_x1, current_x2).reshape(-1, 1)
        # Накапливаем сумму: p_i * (f(x_i) * f(x_i)^T)
        information_matrix += current_weight * (basis_vector_column @ basis_vector_column.T)

    # Дисперсионная матрица как обратная к информационной
    dispersion_matrix = np.linalg.inv(information_matrix)
    return information_matrix, dispersion_matrix


def calculate_dispersion_function_values(grid_of_points, dispersion_matrix):
    """
    Вычисляет значение функции дисперсии d(x, epsilon) = f^T(x) * D * f(x)
    на всей исследуемой координатной сетке.
    """
    grid_x1_coords = grid_of_points[:, 0]
    grid_x2_coords = grid_of_points[:, 1]

    # Формируем матрицу базисных функций F для всех точек сетки сразу
    basis_functions_matrix = np.column_stack([
        np.ones_like(grid_x1_coords),
        grid_x1_coords,
        grid_x2_coords,
        grid_x1_coords**2,
        grid_x1_coords * grid_x2_coords,
        grid_x2_coords**2
    ])

    # Квадратичная форма: d(x) = diag(F @ D @ F^T) = sum((F @ D) * F, axis=1)
    dispersion_values = np.sum((basis_functions_matrix @ dispersion_matrix) * basis_functions_matrix, axis=1)
    return dispersion_values


# 2. Процедура очистки синтезированного плана
def clean_synthesized_plan(design_points, design_weights, distance_tolerance=0.08, min_weight_threshold=0.015):
    """
    Процедура очистки плана:
    1. Объединяет близкие точки, тяготеющие друг к другу (на расстоянии <= distance_tolerance).
    2. Отбрасывает точки с пренебрежимо малыми весами (< min_weight_threshold).
    3. Нормирует веса оставшихся точек так, чтобы sum(p_i) = 1.
    """
    point_clusters = []  # Список кортежей: (список точек кластера, список весов)

    for current_point, current_weight in zip(design_points, design_weights):
        is_point_added_to_cluster = False
        for cluster_points, cluster_weights in point_clusters:
            # Считаем текущий центр масс кластера
            cluster_center = np.average(cluster_points, axis=0, weights=cluster_weights)
            # Проверяем расстояние между текущей точкой и центром кластера
            if np.linalg.norm(current_point - cluster_center) <= distance_tolerance:
                cluster_points.append(current_point)
                cluster_weights.append(current_weight)
                is_point_added_to_cluster = True
                break
        
        # Если точка не подошла ни к одному кластеру, создаем новый
        if not is_point_added_to_cluster:
            point_clusters.append(([current_point], [current_weight]))

    cleaned_points_list = []
    cleaned_weights_list = []

    # Сворачиваем каждый кластер в его центр масс и отфильтровываем паразитные точки
    for cluster_points, cluster_weights in point_clusters:
        total_cluster_weight = sum(cluster_weights)
        if total_cluster_weight >= min_weight_threshold:
            cluster_center = np.average(cluster_points, axis=0, weights=cluster_weights)
            # Округляем координаты до 4 знаков
            cleaned_points_list.append(np.round(cluster_center, 4))
            cleaned_weights_list.append(total_cluster_weight)

    cleaned_points_array = np.array(cleaned_points_list)
    cleaned_weights_array = np.array(cleaned_weights_list)

    # Нормировка суммарных весов к 1с
    cleaned_weights_array /= np.sum(cleaned_weights_array)

    return cleaned_points_array, cleaned_weights_array


# 3. Последовательный алгоритм построения непрерывного D-оптимального плана
def run_sequential_d_optimal_algorithm(
    tolerance_delta=0.01, 
    maximum_iterations=400, 
    logging_step_interval=10, 
    grid_density_per_axis=81
):
    """
    Последовательный алгоритм (алгоритм Федорова-Винна):
    1. Инициализирует начальный невырожденный план ПФЭ 5x5 (25 точек, равные веса).
    2. Находит точку глобального максимума d(x, epsilon).
    3. Проверяет условие Кифера-Вольфовица: max d(x) - m <= delta.
    4. Добавляет новую точку с пересчетом весов и дроблением шага alpha при необходимости.
    5. Выполняет очистку плана и формирует итоговый протокол.
    """
    number_of_model_parameters = 6

    # Формирование начального плана: полный факторный эксперимент из 25 точек
    coordinate_levels = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    mesh_x1, mesh_x2 = np.meshgrid(coordinate_levels, coordinate_levels)
    initial_design_points = np.column_stack([mesh_x1.ravel(), mesh_x2.ravel()])
    initial_design_weights = np.full(len(initial_design_points), 1.0 / len(initial_design_points))

    current_points = initial_design_points.copy()
    current_weights = initial_design_weights.copy()

    # Сетка в области [-1, 1] x [-1, 1] для поиска максимума d(x, epsilon)
    search_levels = np.linspace(-1.0, 1.0, grid_density_per_axis)
    search_mesh_x1, search_mesh_x2 = np.meshgrid(search_levels, search_levels)
    search_grid_points = np.column_stack([search_mesh_x1.ravel(), search_mesh_x2.ravel()])

    # Список для формирования таблицы-протокола итераций
    iteration_log_records = []

    print("=" * 80)
    print(" ЛАБОРАТОРНАЯ РАБОТА № 2. ВАРИАНТ № 1")
    print(" Построение непрерывного D-оптимального плана (Последовательный алгоритм)")
    print(f" Модель: полная квадратичная на квадрате [-1, 1] x [-1, 1] (m = {number_of_model_parameters})")
    print(f" Начальный план: 25 точек (ПФЭ 5x5), равные веса p_i = 1/25")
    print(f" Точность останова: max d(x) - m <= {tolerance_delta}")
    print("=" * 80)

    for iteration_index in range(maximum_iterations):
        # Шаг 1: Вычисление матриц и определителя
        current_inf_matrix, current_disp_matrix = calculate_information_and_dispersion_matrices(
            current_points, current_weights
        )
        current_determinant = np.linalg.det(current_inf_matrix)
        current_ln_determinant = np.log(current_determinant) if current_determinant > 0 else -np.inf

        # Шаг 2: Поиск точки максимума функции дисперсии d(x, epsilon)
        dispersion_values_on_grid = calculate_dispersion_function_values(search_grid_points, current_disp_matrix)
        index_of_max_dispersion = np.argmax(dispersion_values_on_grid)
        maximum_dispersion_value = dispersion_values_on_grid[index_of_max_dispersion]
        best_extreme_point = search_grid_points[index_of_max_dispersion]

        # Невязка по теореме эквивалентности Кифера-Вольфовица: max d(x) - m
        criterion_discrepancy = maximum_dispersion_value - number_of_model_parameters

        # Запись в протокол (с заданной дискретностью либо на завершающих шагах)
        current_step_alpha = 1.0 / (len(current_points) + 1)
        if (
            iteration_index % logging_step_interval == 0
            or criterion_discrepancy <= tolerance_delta
            or iteration_index == maximum_iterations - 1
        ):
            iteration_log_records.append({
                "Итерация s": iteration_index,
                "|M(ε_s)|": current_determinant,
                "ln|M(ε_s)|": current_ln_determinant,
                "max d(x)": maximum_dispersion_value,
                "Невязка (d - m)": criterion_discrepancy,
                "Точка x_s": f"({best_extreme_point[0]:.2f}, {best_extreme_point[1]:.2f})"
            })

        # Шаг 3: Проверка критерия останова
        if criterion_discrepancy <= tolerance_delta:
            break

        # Шаг 4: Пересчет весов и добавление точки x_s с процедурой дробления шага
        gamma_reduction_factor = 2.0
        trial_alpha_step = current_step_alpha

        while True:
            trial_weights = (1.0 - trial_alpha_step) * current_weights
            trial_points = np.vstack([current_points, best_extreme_point])
            trial_weights = np.append(trial_weights, trial_alpha_step)

            trial_inf_matrix, _ = calculate_information_and_dispersion_matrices(trial_points, trial_weights)
            trial_determinant = np.linalg.det(trial_inf_matrix)

            # Проверяем возрастание функционала |M|
            if trial_determinant > current_determinant or trial_alpha_step < 1e-6:
                current_points = trial_points
                current_weights = trial_weights
                break
            else:
                trial_alpha_step /= gamma_reduction_factor

    # Очистка плана от близких точек и малых весов
    cleaned_points, cleaned_weights = clean_synthesized_plan(
        current_points, current_weights, distance_tolerance=0.1, min_weight_threshold=0.015
    )

    cleaned_inf_matrix, cleaned_disp_matrix = calculate_information_and_dispersion_matrices(
        cleaned_points, cleaned_weights
    )
    cleaned_determinant = np.linalg.det(cleaned_inf_matrix)
    cleaned_dispersion_values = calculate_dispersion_function_values(search_grid_points, cleaned_disp_matrix)
    cleaned_max_dispersion = np.max(cleaned_dispersion_values)

    # 4. Вывод протокола решения и результатов в консоль
    dataframe_of_iterations = pd.DataFrame(iteration_log_records)
    print("\nПРОТОКОЛ ВЫЧИСЛИТЕЛЬНОГО ПРОЦЕССА:")
    print(dataframe_of_iterations.round(5).to_string(index=False))

    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ СИНТЕЗА И ОЧИСТКИ ПЛАНА:")
    print(f"Число точек спектра в итоговом плане: {len(cleaned_points)}")
    print(f"Определитель информационной матрицы |M(ε*)| = {cleaned_determinant:.6f}")
    print(f"Значение критерия ln |M(ε*)|               = {np.log(cleaned_determinant):.6f}")
    print(f"Максимальная дисперсия max d(x, ε*)         = {cleaned_max_dispersion:.5f} (теоретический = 6.00000)")
    print(f"Невязка теоремы эквивалентности             = {abs(cleaned_max_dispersion - number_of_model_parameters):.5f}")
    print("=" * 80)

    dataframe_of_final_plan = pd.DataFrame({
        "Точка x1": cleaned_points[:, 0],
        "Точка x2": cleaned_points[:, 1],
        "Вес p_i": cleaned_weights
    })
    dataframe_of_final_plan.index = [f"Точка {i+1}" for i in range(len(cleaned_points))]
    print("\nИТОГОВЫЙ СПЕКТР И ВЕСА ОПТИМАЛЬНОГО ПЛАНА ε*:")
    print(dataframe_of_final_plan.round(4).to_string())

    # 5. Построение графиков
    fig = plt.figure(figsize=(15, 4.8), dpi=100)

    # График 1: Начальный план эксперимента (25 точек)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("Начальный план $\\varepsilon_0$ (25 точек)", fontsize=11)
    initial_marker_sizes = initial_design_weights * 2500
    ax1.scatter(
        initial_design_points[:, 0], 
        initial_design_points[:, 1], 
        s=initial_marker_sizes, 
        color="royalblue", 
        alpha=0.6, 
        edgecolors="black"
    )
    ax1.set_xlim(-1.25, 1.25)
    ax1.set_ylim(-1.25, 1.25)
    ax1.set_xlabel("$x_1$", fontsize=10)
    ax1.set_ylabel("$x_2$", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # График 2: Итоговый оптимальный план после очистки
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Оптимальный план $\\varepsilon^*$", fontsize=11)
    cleaned_marker_sizes = cleaned_weights * 2500
    ax2.scatter(
        cleaned_points[:, 0], 
        cleaned_points[:, 1], 
        s=cleaned_marker_sizes, 
        color="crimson", 
        alpha=0.7, 
        edgecolors="black"
    )
    for (pt_x1, pt_x2), pt_w in zip(cleaned_points, cleaned_weights):
        ax2.annotate(
            f"{pt_w:.3f}", 
            (pt_x1, pt_x2), 
            textcoords="offset points", 
            xytext=(0, 8), 
            ha="center", 
            fontsize=8, 
            fontweight="bold"
        )
    ax2.set_xlim(-1.25, 1.25)
    ax2.set_ylim(-1.25, 1.25)
    ax2.set_xlabel("$x_1$", fontsize=10)
    ax2.set_ylabel("$x_2$", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # График 3: Поверхность и изолинии функции дисперсии d(x, epsilon*)
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title("Функция дисперсии $d(x, \\varepsilon^*)$", fontsize=11)
    dispersion_contour_map = cleaned_dispersion_values.reshape(search_mesh_x1.shape)
    contour_fill = ax3.contourf(search_mesh_x1, search_mesh_x2, dispersion_contour_map, levels=18, cmap="viridis")
    plt.colorbar(contour_fill, ax=ax3, label=r"$d(x, \varepsilon^*)$")
    ax3.scatter(
        cleaned_points[:, 0], 
        cleaned_points[:, 1], 
        color="white", 
        edgecolors="red", 
        linewidths=1.5, 
        marker="o", 
        s=50, 
        label="Узлы $\\varepsilon^*$"
    )
    ax3.set_xlim(-1.0, 1.0)
    ax3.set_ylim(-1.0, 1.0)
    ax3.set_xlabel("$x_1$", fontsize=10)
    ax3.set_ylabel("$x_2$", fontsize=10)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_sequential_d_optimal_algorithm()