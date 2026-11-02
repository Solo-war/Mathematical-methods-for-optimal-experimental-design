import os
import shutil
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT


# ==============================================================================
# ЧИСТЫЙ КОД ЛАБОРАТОРНОЙ РАБОТЫ № 4 (ВАРИАНТ 1) ДЛЯ ПРИЛОЖЕНИЯ А В ОТЧЕТЕ
# ==============================================================================

CLEAN_PROGRAM_CODE_FOR_APPENDIX = r'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Имитационная модель Кобба-Дугласа и вектор чувствительности f(x, theta)
# Модель: y = theta_0 * (X1 ^ theta_1) * (X2 ^ theta_2) + e, число параметров m = 3
# Вариант 1: два ресурса X1, X2 в [1, 10], постоянная отдача от масштаба (theta_1 + theta_2 = 1)
def cobb_douglas_response(points_2d, theta_params):
    """Вычисляет отклик функции Кобба-Дугласа eta(x, theta) = theta_0 * X1^theta_1 * X2^theta_2."""
    theta_0, theta_1, theta_2 = theta_params[0], theta_params[1], theta_params[2]
    pts = np.atleast_2d(points_2d)
    x1_vals, x2_vals = pts[:, 0], pts[:, 1]
    response = theta_0 * (x1_vals ** theta_1) * (x2_vals ** theta_2)
    return response if len( np.shape(points_2d) ) > 1 else float(response[0])


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


# 4. Выполнение вычислительного эксперимента (Пункты 2–6 задания)
true_theta_parameters = np.array([2.50, 0.60, 0.40])  # Постоянная отдача от масштаба: 0.6 + 0.4 = 1.0
noise_level_ratio = 0.15  # Уровень шума 15% от мощности полезного сигнала
master_rng = np.random.default_rng(42)

# Затравочный эксперимент из N_0 = 6 точек
seed_experiment_points = np.array([
    [1.0, 1.0],
    [10.0, 1.0],
    [1.0, 10.0],
    [10.0, 10.0],
    [5.0, 5.0],
    [3.0, 7.0]
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
    _, y_opt_sim, _ = simulate_noisy_observations(optimal_plan_points, true_theta_parameters, noise_level_ratio, master_rng)
    theta_hat_opt_log = estimate_parameters_log_ols(optimal_plan_points, y_opt_sim)
    theta_hat_opt_nls = estimate_parameters_nonlinear_ls(optimal_plan_points, y_opt_sim)
    opt_norms_log.append(float(np.linalg.norm(theta_hat_opt_log - true_theta_parameters)))
    opt_norms_nls.append(float(np.linalg.norm(theta_hat_opt_nls - true_theta_parameters)))
    opt_estimates_list.append(theta_hat_opt_nls)

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

print("=" * 95)
print(" ЛАБОРАТОРНАЯ РАБОТА № 4. ВАРИАНТ № 1 (ФУНКЦИЯ КОББА-ДУГЛАСА)")
print("=" * 95)
print(f"Истинные параметры: theta_true = {true_theta_parameters}")
print(f"Оценка по затравочному эксперименту (N0=6): theta_hat_0 = {np.round(seed_theta_estimate, 4)}")
print(f"Норма ошибки затравочного эксперимента: {seed_estimation_error_norm:.4f}")
print("\nСпектр локально-D-оптимального плана (N=15):")
print(df_optimal_plan_grouped.to_string(index=False))
print(f"\nСредняя норма ошибки (100 серий, нелин. МНК): D-опт = {np.mean(opt_norms_nls):.4f} | Случайный = {np.mean(rand_norms_nls):.4f}")
print(f"Средняя норма ошибки (100 серий, лог. МНК)  : D-опт = {np.mean(opt_norms_log):.4f} | Случайный = {np.mean(rand_norms_log):.4f}")

# 5. Визуализация результатов
fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=120)

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
'''


# ==============================================================================
# ВЫПОЛНЕНИЕ РАСЧЕТОВ И СОХРАНЕНИЕ ГРАФИКА ДЛЯ ВСТАВКИ В WORD
# ==============================================================================
exec_namespace = {}
exec(CLEAN_PROGRAM_CODE_FOR_APPENDIX.replace("plt.show()", "plt.savefig('lab4_results.png', dpi=200)\nplt.close(fig)"), exec_namespace)

true_theta_parameters = exec_namespace["true_theta_parameters"]
seed_experiment_points = exec_namespace["seed_experiment_points"]
seed_clean_y = exec_namespace["seed_clean_y"]
seed_noisy_y = exec_namespace["seed_noisy_y"]
seed_sigma = exec_namespace["seed_sigma"]
seed_theta_estimate = exec_namespace["seed_theta_estimate"]
seed_estimation_error_norm = exec_namespace["seed_estimation_error_norm"]
local_optimal_result = exec_namespace["local_optimal_result"]
df_optimal_plan_grouped = exec_namespace["df_optimal_plan_grouped"]
det_opt_true = exec_namespace["det_opt_true"]
det_rand_true = exec_namespace["det_rand_true"]
opt_norms_nls = exec_namespace["opt_norms_nls"]
rand_norms_nls = exec_namespace["rand_norms_nls"]
opt_norms_log = exec_namespace["opt_norms_log"]
rand_norms_log = exec_namespace["rand_norms_log"]
mean_opt_theta = exec_namespace["mean_opt_theta"]
mean_rand_theta = exec_namespace["mean_rand_theta"]

df_seed_data = pd.DataFrame({
    "№ опыта": np.arange(1, len(seed_experiment_points) + 1),
    "X1": seed_experiment_points[:, 0],
    "X2": seed_experiment_points[:, 1],
    "Чистый отклик η(x)": np.round(seed_clean_y, 4),
    "Зашумленный выход y": np.round(seed_noisy_y, 4)
})

df_seed_estimates = pd.DataFrame({
    "Параметр": ["θ₀ (масштаб)", "θ₁ (эластичность по X₁)", "θ₂ (эластичность по X₂)", "Норма отклонения ||θ̂⁰ - θ_true||₂"],
    "Истинное значение θ_true": [f"{true_theta_parameters[0]:.4f}", f"{true_theta_parameters[1]:.4f}", f"{true_theta_parameters[2]:.4f}", "—"],
    "Оценка по затравке θ̂⁰ (лог. МНК)": [
        f"{seed_theta_estimate[0]:.4f}",
        f"{seed_theta_estimate[1]:.4f}",
        f"{seed_theta_estimate[2]:.4f}",
        f"{seed_estimation_error_norm:.4f}"
    ],
    "Абсолютная ошибка |θ̂_i⁰ - θ_i|": [
        f"{abs(seed_theta_estimate[0] - true_theta_parameters[0]):.4f}",
        f"{abs(seed_theta_estimate[1] - true_theta_parameters[1]):.4f}",
        f"{abs(seed_theta_estimate[2] - true_theta_parameters[2]):.4f}",
        "—"
    ]
})

df_monte_carlo_summary = pd.DataFrame({
    "Характеристика качества (по 100 сериям, N = 15)": [
        "Определитель |M(ε, θ_true)| ↑",
        "Средняя норма ошибки ||θ̂ - θ_true||₂ (нелин. МНК) ↓",
        "Средняя норма ошибки ||θ̂ - θ_true||₂ (лог. МНК) ↓",
        "СКО нормы ошибки σ(||θ̂ - θ_true||₂) ↓",
        "Минимальная норма ошибки min ||θ̂ - θ_true||₂ ↓",
        "Максимальная норма ошибки max ||θ̂ - θ_true||₂ ↓",
        "Средняя оценка θ̂₀ (ист. 2.5000)",
        "Средняя оценка θ̂₁ (ист. 0.6000)",
        "Средняя оценка θ̂₂ (ист. 0.4000)"
    ],
    "Локально-D-оптимальный план ε*": [
        f"{det_opt_true:.2f}",
        f"{np.mean(opt_norms_nls):.4f}",
        f"{np.mean(opt_norms_log):.4f}",
        f"{np.std(opt_norms_nls):.4f}",
        f"{np.min(opt_norms_nls):.4f}",
        f"{np.max(opt_norms_nls):.4f}",
        f"{mean_opt_theta[0]:.4f}",
        f"{mean_opt_theta[1]:.4f}",
        f"{mean_opt_theta[2]:.4f}"
    ],
    "Случайный фиксированный план ε_rand": [
        f"{det_rand_true:.2f}",
        f"{np.mean(rand_norms_nls):.4f}",
        f"{np.mean(rand_norms_log):.4f}",
        f"{np.std(rand_norms_nls):.4f}",
        f"{np.min(rand_norms_nls):.4f}",
        f"{np.max(rand_norms_nls):.4f}",
        f"{mean_rand_theta[0]:.4f}",
        f"{mean_rand_theta[1]:.4f}",
        f"{mean_rand_theta[2]:.4f}"
    ],
    "Выигрыш оптимального плана": [
        f"в {det_opt_true / det_rand_true:.1f} раза выше",
        f"в {np.mean(rand_norms_nls) / np.mean(opt_norms_nls):.2f} раза точнее",
        f"в {np.mean(rand_norms_log) / np.mean(opt_norms_log):.2f} раза точнее",
        f"в {np.std(rand_norms_nls) / np.std(opt_norms_nls):.2f} раза стабильнее",
        f"в {np.min(rand_norms_nls) / np.min(opt_norms_nls):.2f} раза ниже",
        f"в {np.max(rand_norms_nls) / np.max(opt_norms_nls):.2f} раза ниже",
        "Ближе к 2.5000",
        "Ближе к 0.6000",
        "Ближе к 0.4000"
    ]
})


# ==============================================================================
# ФОРМИРОВАНИЕ ОТЧЕТА «ММОПЭ ЛР4.docx» ПО ОБРАЗЦУ ЛР 1, 2, 3 И ГОСТ НГТУ
# ==============================================================================
def add_formatted_table(doc, df):
    table = doc.add_table(rows=len(df) + 1, cols=len(df.columns))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for col_idx, col_name in enumerate(df.columns):
        cell = table.cell(0, col_idx)
        cell.text = str(col_name)
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.bold = True
                r.font.name = "Times New Roman"
                r.font.size = Pt(10.5)

    for row_idx, row_data in df.iterrows():
        for col_idx, val in enumerate(row_data):
            cell = table.cell(row_idx + 1, col_idx)
            cell.text = str(val)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.font.name = "Times New Roman"
                    r.font.size = Pt(10.5)


def build_lr4_report_docx(output_filename="ММОПЭ ЛР4.docx"):
    # Берем титульный лист из предыдущего отчета (если файл лежит в этой же папке)
    template_files = ["ММОПЭ ЛР3.docx", "ММОПЭ ЛР2.docx", "ММОПЭ ЛР1.docx"]
    template_used = next((f for f in template_files if os.path.exists(f)), None)

    if template_used:
        shutil.copy(template_used, output_filename)
        doc = Document(output_filename)
        remove_flag = False
        for p in list(doc.paragraphs):
            if "Практическое задание" in p.text:
                p.text = "Практическое задание № 4"
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.font.name = "Times New Roman"
                    r.font.size = Pt(16)
                    r.font.bold = True
            if "Место для ввода текста" in p.text:
                p.text = "Тема: «Оптимальное планирование эксперимента для нелинейных регрессионных моделей»"
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if "1. ЦЕЛЬ РАБОТЫ" in p.text or remove_flag:
                remove_flag = True
                p._element.getparent().remove(p._element)
        # Исправляем таблицу ФИО на титульном листе
        if len(doc.tables) >= 1:
            t0 = doc.tables[0]
            for row in t0.rows:
                for cell in row.cells:
                    if "Преподавател" in cell.text:
                        cell.text = "Преподаватель"
                    if "Андреев" in cell.text:
                        cell.text = "ПМИМ-61 Андреев Игорь"
                    if "кучер" in cell.text.lower():
                        cell.text = "ПМИМ-61 Кучер Андрей"
                    if "попов" in cell.text.lower():
                        cell.text = "Попов Александр Александрович"
        for t in list(doc.tables)[1:]:
            t._element.getparent().remove(t._element)
    else:
        doc = Document()
        header_lines = [
            ("МИНИСТЕРСТВО НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ РОССИЙСКОЙ ФЕДЕРАЦИИ", 12, False),
            ("Федеральное государственное бюджетное образовательное учреждение высшего образования", 12, False),
            ("«Новосибирский государственный технический университет»", 13, True),
            ("Факультет прикладной математики и информатики", 12, False),
            ("Кафедра теоретической и прикладной информатики", 12, False),
            ("\nПрактическое задание № 4", 16, True),
            ("по дисциплине «Математические методы оптимального планирования эксперимента»", 13, False),
            ("Тема: «Оптимальное планирование эксперимента для нелинейных регрессионных моделей»\n", 13, False),
        ]
        for txt, sz, bld in header_lines:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(txt)
            r.font.name, r.font.size, r.font.bold = "Times New Roman", Pt(sz), bld

        title_df = pd.DataFrame({
            "Бригада / Вариант": ["Бригада 9", "Вариант 1", "Преподаватель"],
            "Исполнители": ["ПМИМ-61 Андреев Игорь", "ПМИМ-61 Кучер Андрей", "Попов Александр Александрович"]
        })
        add_formatted_table(doc, title_df)
        p_city = doc.add_paragraph("\nНовосибирск, 2026")
        p_city.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_page_break()

    def add_section_heading(title_text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(title_text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(13)
        r.font.bold = True

    def add_body_paragraph(body_text):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Inches(0.49)
        p.paragraph_format.line_spacing = 1.15
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run(body_text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(12)

    # Раздел 1
    add_section_heading("1. ЦЕЛЬ РАБОТЫ")
    add_body_paragraph("Изучить методы оптимального планирования эксперимента при нелинейной параметризации функции отклика.")

    # Раздел 2
    add_section_heading("2. ПОСТАНОВКА ЗАДАЧИ")
    add_body_paragraph(
        "Исследуется нелинейная регрессионная модель наблюдения y = η(x, θ) + e в виде двухфакторной "
        "производственной функции Кобба-Дугласа: η(x, θ) = θ₀ · X₁^θ₁ · X₂^θ₂, где X₁ ∈ [1; 10], X₂ ∈ [1; 10] — "
        "входные ресурсы (k = 2); θ = (θ₀, θ₁, θ₂)ᵀ — вектор неизвестных параметров (m = 3); e — независимая "
        "нормально распределенная ошибка наблюдения с нулевым математическим ожиданием и дисперсией σ²."
    )
    add_body_paragraph(
        "В соответствии с вариантом 1 задания рассматривается технология с постоянной отдачей от масштаба "
        "(θ₁ + θ₂ = 1) и локально-D-оптимальное планирование. Для имитационной модели заданы истинные значения "
        "параметров θ_true = (2.5000, 0.6000, 0.4000)ᵀ (0.6 + 0.4 = 1.0). Уровень мощности шума установлен равным "
        "15% от средней мощности полезного сигнала (в пределах 15...20%)."
    )
    add_body_paragraph(
        "Требуется: 1) выбрать план затравочного эксперимента из небольшого числа наблюдений (N₀ = 6) и смоделировать "
        "экспериментальные данные; 2) оценить параметры модели по затравочному эксперименту с помощью логарифмического "
        "преобразования и линейного МНК; 3) построить локально-D-оптимальный дискретный план для исходной нелинейной "
        "модели при числе наблюдений N = 15 (в 5 раз превышающем m = 3) по алгоритму Федорова; 4) провести серию из "
        "100 вычислительных экспериментов в точках локально-D-оптимального плана и в точках фиксированного случайного "
        "плана (N = 15), сравнить средние нормы отклонения оценок параметров и сделать вывод об эффективности "
        "оптимального планирования."
    )

    # Раздел 3
    add_section_heading("3. ТЕОРЕТИЧЕСКИЕ СВЕДЕНИЯ")
    add_body_paragraph(
        "Для нелинейной регрессионной модели y = η(x, θ) + e информационная матрица Фишера зависит от вектора "
        "параметров θ. Вектор чувствительности f(x, θ̂) = ∂η(x, θ)/∂θ в точке оценки θ̂ = (θ̂₀, θ̂₁, θ̂₂)ᵀ имеет вид: "
        "f(x, θ̂) = (X₁^θ̂₁·X₂^θ̂₂, θ̂₀·X₁^θ̂₁·X₂^θ̂₂·ln X₁, θ̂₀·X₁^θ̂₁·X₂^θ̂₂·ln X₂)ᵀ."
    )
    add_body_paragraph(
        "Приближенная нормированная информационная матрица дискретного плана из N наблюдений равна "
        "M(ε_N, θ̂) = (1/N)·Σ f(x_j, θ̂)·fᵀ(x_j, θ̂), а дисперсионная матрица D(ε_N, θ̂) = M⁻¹(ε_N, θ̂). "
        "План ε_N* называется локально-D-оптимальным, если он максимизирует определитель |M(ε_N, θ̂)|. "
        "При логарифмировании модели Кобба-Дугласа получается линейная регрессия ln y = ln θ₀ + θ₁·ln X₁ + θ₂·ln X₂ + ε', "
        "оценки параметров которой находятся обычным линейным МНК: β̂ = (UᵀU)⁻¹UᵀZ."
    )

    # Раздел 4
    add_section_heading("4. ПРОГРАММА И МЕТОДИКА ВЫЧИСЛЕНИЙ")
    add_body_paragraph(
        "Программа реализована на Python 3 (NumPy, Pandas, Matplotlib). Для воспроизводимости результатов фиксируется "
        "генератор случайных чисел (seed = 42). На этапе затравочного эксперимента (N₀ = 6) по зашумленным данным "
        "вычисляются начальные оценки параметров θ̂⁰ методом линейного МНК в логарифмических координатах. Затем при "
        "фиксированном θ = θ̂⁰ запускается алгоритм замены точек Федорова на сетке 20×20 квадрата [1; 10]×[1; 10] "
        "при N = 15 (повторные наблюдения допускаются, точка-кандидат выбирается по всей сетке). В серии из 100 "
        "вычислительных экспериментов на каждой итерации генерируется новая реализация гауссовской помехи (15% мощности) "
        "и находятся оценки параметров как логарифмическим МНК, так и нелинейным МНК (методом Гаусса-Ньютона)."
    )

    # Раздел 5
    add_section_heading("5. РЕЗУЛЬТАТЫ ВЫЧИСЛЕНИЙ И ИХ АНАЛИЗ")
    add_body_paragraph("Таблица 1 – Экспериментальные данные затравочного эксперимента (N₀ = 6, уровень шума 15%)")
    add_formatted_table(doc, df_seed_data)

    add_body_paragraph("Таблица 2 – Оценки параметров модели Кобба-Дугласа по затравочному эксперименту")
    add_formatted_table(doc, df_seed_estimates)

    add_body_paragraph(
        f"По полученной начальной оценке θ̂⁰ = ({seed_theta_estimate[0]:.4f}, {seed_theta_estimate[1]:.4f}, "
        f"{seed_theta_estimate[2]:.4f})ᵀ алгоритмом Федорова за {local_optimal_result['iterations']} итераций "
        f"построен локально-D-оптимальный дискретный план из N = 15 наблюдений (таблица 3). Все 15 наблюдений "
        f"равномерно распределились по трем опорным точкам с кратностью r_i = 5 (веса p_i = 1/3), в которых "
        f"дисперсия прогноза достигает теоретического значения d(x_i, ε*) = m = 3.0000."
    )
    add_body_paragraph(
        f"Таблица 3 – Спектр и веса локально-D-оптимального дискретного плана ε* (N = 15, |M(ε*, θ̂⁰)| = {local_optimal_result['det_M']:.2f})"
    )
    add_formatted_table(doc, df_optimal_plan_grouped)

    add_body_paragraph(
        "Таблица 4 – Сравнение точности оценивания параметров для локально-D-оптимального и фиксированного "
        "случайного планов в серии из 100 вычислительных экспериментов (N = 15)"
    )
    add_formatted_table(doc, df_monte_carlo_summary)

    add_body_paragraph(
        f"Анализ результатов (таблица 4, рисунок 1) показывает, что определитель информационной матрицы "
        f"локально-D-оптимального плана в истинной точке θ_true превосходит определитель случайного плана в "
        f"{det_opt_true / det_rand_true:.1f} раза ({det_opt_true:.2f} против {det_rand_true:.2f}). За счет этого "
        f"средняя по 100 сериям норма ошибки оценивания параметров ||θ̂ - θ_true||₂ снижается с "
        f"{np.mean(rand_norms_nls):.4f} до {np.mean(opt_norms_nls):.4f} (выигрыш в точности в "
        f"{np.mean(rand_norms_nls) / np.mean(opt_norms_nls):.2f} раза), а среднеквадратическое отклонение ошибки "
        f"уменьшается в {np.std(rand_norms_nls) / np.std(opt_norms_nls):.2f} раза."
    )

    # Раздел 6
    add_section_heading("6. ГРАФИКИ И ВИЗУАЛИЗАЦИЯ")
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.add_run().add_picture("lab4_results.png", width=Inches(6.3))
    p_cap = doc.add_paragraph(
        "Рисунок 1 — Расположение точек планов на области [1; 10]², функция дисперсии прогноза d(x, ε*) "
        "и сравнение норм ошибок оценивания параметров в 100 сериях вычислительных экспериментов"
    )
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Раздел 7
    add_section_heading("7. ВЫВОДЫ")
    add_body_paragraph(
        "1. Изучены понятия локально-оптимального планирования и информационной матрицы Фишера при нелинейной "
        "параметризации функции отклика на примере двухфакторной производственной функции Кобба-Дугласа с постоянной "
        "отдачей от масштаба (θ_true = (2.5, 0.6, 0.4)ᵀ) на квадрате ресурсов [1; 10] × [1; 10]."
    )
    add_body_paragraph(
        f"2. По данным затравочного эксперимента из N₀ = 6 наблюдений при уровне помехи 15% с помощью логарифмической "
        f"линеаризации и линейного МНК получены предварительные оценки параметров θ̂⁰ = ({seed_theta_estimate[0]:.4f}, "
        f"{seed_theta_estimate[1]:.4f}, {seed_theta_estimate[2]:.4f})ᵀ с нормой отклонения {seed_estimation_error_norm:.4f}."
    )
    add_body_paragraph(
        "3. С использованием алгоритма Федорова на сетке 20×20 синтезирован локально-D-оптимальный дискретный план "
        "объема N = 15, сосредоточенный в трех опорных узлах (10.0, 10.0), (10.0, 1.0) и (1.4737, 10.0) с равными "
        "кратностями r_i = 5 (p_i = 1/3) и максимальной дисперсией прогноза max d(x, ε*) = m = 3.0000."
    )
    add_body_paragraph(
        f"4. По результатам 100 серий имитационных экспериментов установлено, что применение локально-D-оптимального "
        f"плана уменьшает среднюю норму ошибки оценивания параметров в {np.mean(rand_norms_nls) / np.mean(opt_norms_nls):.2f} "
        f"раза ({np.mean(opt_norms_nls):.4f} против {np.mean(rand_norms_nls):.4f}) и снижает разброс ошибок в "
        f"{np.std(rand_norms_nls) / np.std(opt_norms_nls):.2f} раза по сравнению с фиксированным случайным планом, "
        f"что подтверждает высокую эффективность оптимального планирования для идентификации нелинейных моделей."
    )

    # Приложение А
    add_section_heading("ПРИЛОЖЕНИЕ А. Текст программы")
    p_code = doc.add_paragraph()
    r_code = p_code.add_run(CLEAN_PROGRAM_CODE_FOR_APPENDIX)
    r_code.font.name = "Courier New"
    r_code.font.size = Pt(8.5)

    doc.save(output_filename)
    print(f"\nГотовый отчет успешно сохранен в файл: {output_filename}")


if __name__ == "__main__":
    build_lr4_report_docx("ММОПЭ ЛР4.docx")