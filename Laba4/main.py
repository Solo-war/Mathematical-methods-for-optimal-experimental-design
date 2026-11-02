import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# ЧИСТЫЙ КОД ЛАБОРАТОРНОЙ РАБОТЫ № 4 (ВАРИАНТ 1)
# ==============================================================================

# 1. Имитационная модель Кобба-Дугласа и вектор чувствительности f(x, theta)
# Модель: y = theta_0 * (X1 ^ theta_1) * (X2 ^ theta_2) + e, число параметров m = 3
# Вариант 1: два ресурса X1, X2 в [1, 10], постоянная отдача от масштаба (theta_1 + theta_2 = 1)
def cobb_douglas_response(points_2d, theta_params):
    """Вычисляет отклик функции Кобба-Дугласа eta(x, theta) = theta_0 * X1^theta_1 * X2^theta_2."""
    theta_0, theta_1, theta_2 = theta_params[0], theta_params[1], theta_params[2]
    pts = np.atleast_2d(points_2d)
    x1_vals, x2_vals = pts[:, 0], pts[:, 1]
    response = theta_0 * (x1_vals ** theta_1) * (x2_vals ** theta_2)
    return response if len(np.shape(points_2d)) > 1 else float(response[0])


def get_sensitivity_vector(point_2d, theta_params):
    """
    Возвращает вектор частных производных f(x, theta) = d(eta)/d(theta)
    в точке x = (X1, X2) при заданном векторе оценок параметров theta.
    """
    theta_0, theta_1, theta_2 = theta_params[0], theta_params[1], theta_params[2]
    x1_val, x2_val = float(point_2d[0]), float(point_2d[1])
    base_power = (x1_val ** theta_1) * (x2_val ** theta_2)

    d_eta_d_theta0 = base_power
    d_eta_d_theta1 = theta_0 * base_power * np.log(x1_val)
    d_eta_d_theta2 = theta_0 * base_power * np.log(x2_val)
    return np.array([d_eta_d_theta0, d_eta_d_theta1, d_eta_d_theta2])


def get_sensitivity_matrix(design_points, theta_params):
    """Формирует матрицу чувствительности (Якоби) F размера N x 3."""
    return np.array([get_sensitivity_vector(pt, theta_params) for pt in design_points])


def calculate_local_information_and_dispersion_matrices(design_points, theta_params):
    """
    Вычисляет нормированную информационную матрицу Фишера M(e_N, theta)
    и дисперсионную матрицу D(e_N, theta) = M^(-1) для дискретного плана из N точек.
    """
    basis_matrix = get_sensitivity_matrix(design_points, theta_params)
    number_of_observations = len(design_points)
    information_matrix = (basis_matrix.T @ basis_matrix) / float(number_of_observations)
    dispersion_matrix = np.linalg.inv(information_matrix)
    return information_matrix, dispersion_matrix


def simulate_noisy_observations(design_points, true_theta, noise_power_ratio, rng):
    """
    Моделирует зашумленные наблюдения y = eta(x, theta_true) + e.
    Мощность шума составляет noise_power_ratio (15%) от средней мощности полезного сигнала.
    """
    clean_signal = cobb_douglas_response(design_points, true_theta)
    signal_power = np.mean(clean_signal ** 2)
    noise_std_sigma = np.sqrt(noise_power_ratio * signal_power)

    noise_vector = rng.normal(loc=0.0, scale=noise_std_sigma, size=len(design_points))
    noisy_y = np.maximum(clean_signal + noise_vector, 1e-3)
    return clean_signal, noisy_y, float(noise_std_sigma)


# 2. Оценивание параметров модели (линейный МНК после логарифмирования и нелинейный МНК)
def estimate_parameters_log_ols(design_points, observed_y):
    """
    Оценка параметров через логарифмическое представление модели наблюдения:
    ln(y) = ln(theta_0) + theta_1 * ln(X1) + theta_2 * ln(X2).
    """
    log_x1 = np.log(design_points[:, 0])
    log_x2 = np.log(design_points[:, 1])
    linear_matrix_u = np.column_stack([np.ones(len(design_points)), log_x1, log_x2])
    log_y = np.log(observed_y)

    beta_hat, _, _, _ = np.linalg.lstsq(linear_matrix_u, log_y, rcond=None)
    return np.array([float(np.exp(beta_hat[0])), float(beta_hat[1]), float(beta_hat[2])])


def estimate_parameters_nonlinear_ls(design_points, observed_y, max_iter=50, tol=1e-7):
    """
    Нелинейный МНК (метод Гаусса-Ньютона) для минимизации S(theta) = sum((y_j - eta(x_j, theta))^2),
    использующий логарифмическую оценку МНК в качестве начального приближения.
    """
    current_theta = estimate_parameters_log_ols(design_points, observed_y)

    for _ in range(max_iter):
        predicted_y = cobb_douglas_response(design_points, current_theta)
        residuals = observed_y - predicted_y
        current_sse = float(np.sum(residuals ** 2))

        jacobian_f = get_sensitivity_matrix(design_points, current_theta)
        step_delta, _, _, _ = np.linalg.lstsq(jacobian_f, residuals, rcond=None)

        step_scale = 1.0
        while step_scale > 1e-4:
            candidate_theta = current_theta + step_scale * step_delta
            if candidate_theta[0] > 0.05 and -0.5 < candidate_theta[1] < 2.5 and -0.5 < candidate_theta[2] < 2.5:
                candidate_sse = float(np.sum((observed_y - cobb_douglas_response(design_points, candidate_theta)) ** 2))
                if candidate_sse < current_sse:
                    current_theta = candidate_theta
                    break
            step_scale *= 0.5

        if np.linalg.norm(step_scale * step_delta) < tol:
            break

    return current_theta


# 3. Построение локально-D-оптимального дискретного плана (алгоритм Федорова)
def generate_resource_candidate_grid(grid_resolution=20, min_val=1.0, max_val=10.0):
    """Формирует дискретную сетку ресурсов на квадрате [1, 10] x [1, 10]."""
    axis_values = np.linspace(min_val, max_val, grid_resolution)
    mesh_x1, mesh_x2 = np.meshgrid(axis_values, axis_values)
    return np.column_stack([mesh_x1.ravel(), mesh_x2.ravel()])


def run_local_d_optimal_fedorov_algorithm(candidate_grid, number_of_observations, local_theta, delta_threshold=1e-6, seed=42):
    """
    Алгоритм Федорова синтеза локально-D-оптимального дискретного плана из N наблюдений
    в окрестности начальной оценки параметров local_theta. Повторные наблюдения допускаются.
    """
    rng = np.random.default_rng(seed)
    number_of_parameters = 3

    while True:
        selected_indices = rng.choice(len(candidate_grid), size=number_of_observations, replace=True)
        current_plan_points = candidate_grid[selected_indices].copy()
        f_mat = get_sensitivity_matrix(current_plan_points, local_theta)
        inf_mat = (f_mat.T @ f_mat) / float(number_of_observations)
        if np.linalg.matrix_rank(inf_mat) == number_of_parameters and np.linalg.det(inf_mat) > 1e-10:
            break

    candidate_basis_matrix = get_sensitivity_matrix(candidate_grid, local_theta)
    iteration_count = 0

    while True:
        inf_matrix, disp_matrix = calculate_local_information_and_dispersion_matrices(current_plan_points, local_theta)
        plan_basis_matrix = get_sensitivity_matrix(current_plan_points, local_theta)

        d_candidates = np.sum((candidate_basis_matrix @ disp_matrix) * candidate_basis_matrix, axis=1)
        d_plan = np.sum((plan_basis_matrix @ disp_matrix) * plan_basis_matrix, axis=1)
        d_cross = plan_basis_matrix @ disp_matrix @ candidate_basis_matrix.T

        n_float = float(number_of_observations)
        delta_matrix = (
            (1.0 / n_float) * (d_candidates[None, :] - d_plan[:, None])
            - (1.0 / (n_float ** 2)) * (d_plan[:, None] * d_candidates[None, :] - d_cross ** 2)
        )

        flat_max_idx = np.argmax(delta_matrix)
        best_plan_idx, best_cand_idx = np.unravel_index(flat_max_idx, delta_matrix.shape)
        max_delta = float(delta_matrix[best_plan_idx, best_cand_idx])

        if max_delta < delta_threshold:
            break

        current_plan_points[best_plan_idx] = candidate_grid[best_cand_idx]
        iteration_count += 1

    final_inf, final_disp = calculate_local_information_and_dispersion_matrices(current_plan_points, local_theta)
    final_d_candidates = np.sum((candidate_basis_matrix @ final_disp) * candidate_basis_matrix, axis=1)

    return {
        "plan_points": current_plan_points,
        "det_M": float(np.linalg.det(final_inf)),
        "trace_D": float(np.trace(final_disp)),
        "max_d": float(np.max(final_d_candidates)),
        "iterations": iteration_count
    }


def group_discrete_plan_points(plan_points, local_theta):
    """Группирует совпадающие точки плана и подсчитывает кратность наблюдений r_i и вес p_i."""
    _, disp_matrix = calculate_local_information_and_dispersion_matrices(plan_points, local_theta)
    rounded_pts = np.round(plan_points, 4)
    unique_pts, counts = np.unique(rounded_pts, axis=0, return_counts=True)

    sort_order = np.argsort(-counts)
    unique_pts = unique_pts[sort_order]
    counts = counts[sort_order]

    d_vals = [float(get_sensitivity_vector(pt, local_theta).T @ disp_matrix @ get_sensitivity_vector(pt, local_theta)) for pt in unique_pts]

    return pd.DataFrame({
        "X1": unique_pts[:, 0],
        "X2": unique_pts[:, 1],
        "r_i (наблюдений)": counts.astype(int),
        "p_i (вес)": np.round(counts / len(plan_points), 4),
        "d(x_i, e*)": np.round(d_vals, 4)
    })

# ==============================================================================
# 4. ВЫПОЛНЕНИЕ ВЫЧИСЛИТЕЛЬНОГО ЭКСПЕРИМЕНТА (Пункты 2–6 задания)
# ==============================================================================
if __name__ == "__main__":
    true_theta_parameters = np.array([2.50, 0.60, 0.40])  # Постоянная отдача от масштаба: 0.6 + 0.4 = 1.0
    noise_level_ratio = 0.15  # Уровень шума 15% от мощности полезного сигнала
    master_rng = np.random.default_rng(42)

    # Затравочный эксперимент из N_0 = 6 точек
    seed_experiment_points = np.array([
        [1.0, 1.0], [10.0, 1.0], [1.0, 10.0], 
        [10.0, 10.0], [5.0, 5.0], [3.0, 7.0]
    ])
    seed_clean_y, seed_noisy_y, seed_sigma = simulate_noisy_observations(
        seed_experiment_points, true_theta_parameters, noise_level_ratio, master_rng
    )
    seed_theta_estimate = estimate_parameters_log_ols(seed_experiment_points, seed_noisy_y)
    seed_estimation_error_norm = float(np.linalg.norm(seed_theta_estimate - true_theta_parameters))

    # Синтез локально-D-оптимального плана для N = 15 (в 5 раз превышает m = 3)
    number_of_main_observations = 15
    candidate_grid_20x20 = generate_resource_candidate_grid(grid_resolution=20, min_val=1.0, max_val=10.0)
    local_optimal_result = run_local_d_optimal_fedorov_algorithm(
        candidate_grid=candidate_grid_20x20,
        number_of_observations=number_of_main_observations,
        local_theta=seed_theta_estimate
    )
    optimal_plan_points = local_optimal_result["plan_points"]
    df_optimal_plan_grouped = group_discrete_plan_points(optimal_plan_points, seed_theta_estimate)

    # Серия из 100 вычислительных экспериментов: локально-D-оптимальный план против фиксированного случайного плана
    fixed_random_plan_points = master_rng.uniform(low=1.0, high=10.0, size=(number_of_main_observations, 2))

    inf_opt_true, _ = calculate_local_information_and_dispersion_matrices(optimal_plan_points, true_theta_parameters)
    inf_rand_true, _ = calculate_local_information_and_dispersion_matrices(fixed_random_plan_points, true_theta_parameters)
    det_opt_true = float(np.linalg.det(inf_opt_true))
    det_rand_true = float(np.linalg.det(inf_rand_true))

    num_monte_carlo_runs = 100
    opt_norms_nls, rand_norms_nls = [], []
    opt_norms_log, rand_norms_log = [], []
    opt_estimates_list, rand_estimates_list = [], []

    for _ in range(num_monte_carlo_runs):
        # Эксперименты на оптимальном плане
        _, y_opt_sim, _ = simulate_noisy_observations(optimal_plan_points, true_theta_parameters, noise_level_ratio, master_rng)
        theta_hat_opt_log = estimate_parameters_log_ols(optimal_plan_points, y_opt_sim)
        theta_hat_opt_nls = estimate_parameters_nonlinear_ls(optimal_plan_points, y_opt_sim)
        opt_norms_log.append(float(np.linalg.norm(theta_hat_opt_log - true_theta_parameters)))
        opt_norms_nls.append(float(np.linalg.norm(theta_hat_opt_nls - true_theta_parameters)))
        opt_estimates_list.append(theta_hat_opt_nls)

        # Эксперименты на случайном плане
        _, y_rand_sim, _ = simulate_noisy_observations(fixed_random_plan_points, true_theta_parameters, noise_level_ratio, master_rng)
        theta_hat_rand_log = estimate_parameters_log_ols(fixed_random_plan_points, y_rand_sim)
        theta_hat_rand_nls = estimate_parameters_nonlinear_ls(fixed_random_plan_points, y_rand_sim)
        rand_norms_log.append(float(np.linalg.norm(theta_hat_rand_log - true_theta_parameters)))
        rand_norms_nls.append(float(np.linalg.norm(theta_hat_rand_nls - true_theta_parameters)))
        rand_estimates_list.append(theta_hat_rand_nls)

    opt_norms_nls = np.array(opt_norms_nls)
    rand_norms_nls = np.array(rand_norms_nls)
    opt_norms_log = np.array(opt_norms_log)
    rand_norms_log = np.array(rand_norms_log)
    mean_opt_theta = np.mean(opt_estimates_list, axis=0)
    mean_rand_theta = np.mean(rand_estimates_list, axis=0)

    # --- Подготовка таблиц вывода ---
    df_seed_data = pd.DataFrame({
        "№ опыта": np.arange(1, len(seed_experiment_points) + 1),
        "X1": seed_experiment_points[:, 0],
        "X2": seed_experiment_points[:, 1],
        "Чистый отклик η(x)": np.round(seed_clean_y, 4),
        "Зашумленный выход y": np.round(seed_noisy_y, 4)
    })

    df_seed_estimates = pd.DataFrame({
        "Параметр": ["θ₀ (масштаб)", "θ₁ (эластичность по X₁)", "θ₂ (эластичность по X₂)"],
        "Истинное значение θ_true": [f"{true_theta_parameters[0]:.4f}", f"{true_theta_parameters[1]:.4f}", f"{true_theta_parameters[2]:.4f}"],
        "Оценка по затравке θ̂⁰ (лог. МНК)": [
            f"{seed_theta_estimate[0]:.4f}", f"{seed_theta_estimate[1]:.4f}", f"{seed_theta_estimate[2]:.4f}"
        ],
        "Абсолютная ошибка |θ̂_i⁰ - θ_i|": [
            f"{abs(seed_theta_estimate[0] - true_theta_parameters[0]):.4f}",
            f"{abs(seed_theta_estimate[1] - true_theta_parameters[1]):.4f}",
            f"{abs(seed_theta_estimate[2] - true_theta_parameters[2]):.4f}"
        ]
    })

    df_monte_carlo_summary = pd.DataFrame({
        "Характеристика качества (по 100 сериям, N = 15)": [
            "Определитель |M(ε, θ_true)| ↑",
            "Средняя норма ошибки ||θ̂ - θ_true||₂ (нелин. МНК) ↓",
            "Средняя норма ошибки ||θ̂ - θ_true||₂ (лог. МНК) ↓",
            "СКО нормы ошибки σ(||θ̂ - θ_true||₂) ↓",
            "Минимальная норма ошибки min ||θ̂ - θ_true||₂ ↓",
            "Максимальная норма ошибки max ||θ̂ - θ_true||₂ ↓"
        ],
        "Локально-D-оптимальный план ε*": [
            f"{det_opt_true:.2f}", f"{np.mean(opt_norms_nls):.4f}", f"{np.mean(opt_norms_log):.4f}",
            f"{np.std(opt_norms_nls):.4f}", f"{np.min(opt_norms_nls):.4f}", f"{np.max(opt_norms_nls):.4f}"
        ],
        "Случайный фиксированный план ε_rand": [
            f"{det_rand_true:.2f}", f"{np.mean(rand_norms_nls):.4f}", f"{np.mean(rand_norms_log):.4f}",
            f"{np.std(rand_norms_nls):.4f}", f"{np.min(rand_norms_nls):.4f}", f"{np.max(rand_norms_nls):.4f}"
        ],
        "Выигрыш оптимального плана": [
            f"в {det_opt_true / det_rand_true:.1f} раза выше",
            f"в {np.mean(rand_norms_nls) / np.mean(opt_norms_nls):.2f} раза точнее",
            f"в {np.mean(rand_norms_log) / np.mean(opt_norms_log):.2f} раза точнее",
            f"в {np.std(rand_norms_nls) / np.std(opt_norms_nls):.2f} раза стабильнее",
            f"в {np.min(rand_norms_nls) / np.min(opt_norms_nls):.2f} раза ниже",
            f"в {np.max(rand_norms_nls) / np.max(opt_norms_nls):.2f} раза ниже"
        ]
    })

    print("=" * 95)
    print(" ЛАБОРАТОРНАЯ РАБОТА № 4. ВАРИАНТ № 1 (ФУНКЦИЯ КОББА-ДУГЛАСА)")
    print("=" * 95)
    print("\n1. Данные затравочного эксперимента:")
    print(df_seed_data.to_string(index=False))
    print(f"\n2. Оценки по затравочному эксперименту (Норма ошибки: {seed_estimation_error_norm:.4f}):")
    print(df_seed_estimates.to_string(index=False))
    print("\n3. Спектр локально-D-оптимального плана (N=15):")
    print(df_optimal_plan_grouped.to_string(index=False))
    print("\n4. Сравнение планов по результатам 100 серий экспериментов:")
    print(df_monte_carlo_summary.to_string(index=False))

    # ==============================================================================
    # 5. ВИЗУАЛИЗАЦИЯ
    # ==============================================================================
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=120)

    # График 1: Изолинии отклика и точки планов
    ax1 = axes[0, 0]
    grid_axis = np.linspace(1.0, 10.0, 60)
    gx1, gx2 = np.meshgrid(grid_axis, grid_axis)
    g_pts = np.column_stack([gx1.ravel(), gx2.ravel()])
    eta_surface = cobb_douglas_response(g_pts, true_theta_parameters).reshape(gx1.shape)

    cf1 = ax1.contourf(gx1, gx2, eta_surface, levels=15, cmap="YlGnBu", alpha=0.75)
    plt.colorbar(cf1, ax=ax1, label=r"Отклик $\eta(X_1, X_2, \theta_{true})$")
    ax1.scatter(fixed_random_plan_points[:, 0], fixed_random_plan_points[:, 1], color="royalblue", marker="^", s=55, edgecolors="black", label="Случайный план (N=15)", zorder=3)
    ax1.scatter(seed_experiment_points[:, 0], seed_experiment_points[:, 1], color="limegreen", marker="s", s=50, edgecolors="black", label="Затравочный план (N₀=6)", zorder=4)
    ax1.scatter(df_optimal_plan_grouped["X1"], df_optimal_plan_grouped["X2"], color="crimson", marker="o", s=df_optimal_plan_grouped["r_i (наблюдений)"] * 45, edgecolors="black", label="Локально-D-опт. план (N=15)", zorder=5)
    
    for _, row in df_optimal_plan_grouped.iterrows():
        ax1.annotate(f"r={int(row['r_i (наблюдений)'])}", (row["X1"], row["X2"]), textcoords="offset points", xytext=(-12, -14), fontsize=9, fontweight="bold", color="darkred", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
    
    ax1.set_title("Планы эксперимента на области ресурсов [1; 10] × [1; 10]", fontsize=11)
    ax1.set_xlabel("Ресурс $X_1$", fontsize=10)
    ax1.set_ylabel("Ресурс $X_2$", fontsize=10)
    ax1.set_xlim(0.5, 10.7)
    ax1.set_ylim(0.5, 10.7)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=8)

    # График 2: Дисперсия прогноза
    ax2 = axes[0, 1]
    _, disp_opt_local = calculate_local_information_and_dispersion_matrices(optimal_plan_points, seed_theta_estimate)
    f_grid_all = get_sensitivity_matrix(g_pts, seed_theta_estimate)
    d_surface = np.sum((f_grid_all @ disp_opt_local) * f_grid_all, axis=1).reshape(gx1.shape)
    
    cf2 = ax2.contourf(gx1, gx2, d_surface, levels=18, cmap="plasma")
    plt.colorbar(cf2, ax=ax2, label=r"$d(x, \varepsilon_{15}^*, \hat{\theta}^{(0)})$")
    ax2.scatter(df_optimal_plan_grouped["X1"], df_optimal_plan_grouped["X2"], color="white", edgecolors="black", s=90, linewidths=1.5, label="Опорные узлы (d = m = 3)")
    ax2.set_title(r"Дисперсия прогноза $d(x, \varepsilon_{15}^*, \hat{\theta}^{(0)})$ (максимум = 3.0)", fontsize=11)
    ax2.set_xlabel("Ресурс $X_1$", fontsize=10)
    ax2.set_ylabel("Ресурс $X_2$", fontsize=10)
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # График 3: Нормы ошибок в сериях
    ax3 = axes[1, 0]
    runs_axis = np.arange(1, num_monte_carlo_runs + 1)
    ax3.plot(runs_axis, rand_norms_nls, color="royalblue", alpha=0.65, linewidth=1.2, label="Случайный план (по сериям)")
    ax3.plot(runs_axis, opt_norms_nls, color="crimson", alpha=0.85, linewidth=1.5, label="Локально-D-опт. план (по сериям)")
    ax3.axhline(np.mean(rand_norms_nls), color="navy", linestyle="--", linewidth=2, label=f"Среднее случ. плана: {np.mean(rand_norms_nls):.4f}")
    ax3.axhline(np.mean(opt_norms_nls), color="darkred", linestyle="-.", linewidth=2, label=f"Среднее D-опт. плана: {np.mean(opt_norms_nls):.4f}")
    ax3.set_title("Норма ошибки $||\\hat{\\theta} - \\theta_{true}||_2$ в 100 сериях экспериментов", fontsize=11)
    ax3.set_xlabel("Номер вычислительного эксперимента (серии)", fontsize=10)
    ax3.set_ylabel(r"Норма отклонения $||\hat{\theta} - \theta_{true}||_2$", fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(fontsize=8)

    # График 4: Распределение норм ошибок
    ax4 = axes[1, 1]
    bins = np.linspace(0, max(np.max(rand_norms_nls), np.max(opt_norms_nls)) * 1.05, 22)
    ax4.hist(rand_norms_nls, bins=bins, color="royalblue", alpha=0.55, edgecolor="black", label="Случайный план")
    ax4.hist(opt_norms_nls, bins=bins, color="crimson", alpha=0.65, edgecolor="black", label="Локально-D-опт. план")
    ax4.axvline(np.mean(rand_norms_nls), color="navy", linestyle="--", linewidth=2)
    ax4.axvline(np.mean(opt_norms_nls), color="darkred", linestyle="-.", linewidth=2)
    ax4.set_title("Распределение норм ошибок оценивания (100 серий)", fontsize=11)
    ax4.set_xlabel(r"Норма ошибки $||\hat{\theta} - \theta_{true}||_2$", fontsize=10)
    ax4.set_ylabel("Частота (число серий)", fontsize=10)
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(fontsize=9)

    plt.tight_layout()
    plt.show()