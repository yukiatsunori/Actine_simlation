import numpy as np


# F-アクチン
# サブユニットの数と位置
def generate_actin_subunits(F_actin_length, G_actin_length):
    num_subunits = int(F_actin_length / G_actin_length)
    return np.linspace(0, F_actin_length, num_subunits)


# サブユニットに結合する確率
def calculate_binding_probability(k_binding, C_sbp, Dt):
    return k_binding * C_sbp * Dt


# C_sbpをサブユニットごとに設定（空間的な分布を仮定）
def generate_spatial_distribution(num_subunits, mean_concentration):
    return mean_concentration * (
        1 + 0.2 * np.sin(np.linspace(0, 2 * np.pi, num_subunits))
    )


# サブユニットの結合イベントをサンプリング
def sample_binding_sites(binding_probabilities):
    cumulative_probabilities = np.cumsum(binding_probabilities)
    cumulative_probabilities /= cumulative_probabilities[-1]
    random_values = np.random.rand(len(binding_probabilities))
    return [np.searchsorted(cumulative_probabilities, rv) for rv in random_values]


# F-アクチンの動態をシミュレート
def simulate_F_actin_dynamics(
    F_actin_length, G_actin_length, k_binding, C_sbp, Dt, steps
):
    subunits = generate_actin_subunits(F_actin_length, G_actin_length)
    binding_probabilities = np.array(
        [calculate_binding_probability(k_binding, C_sbp, Dt) for _ in subunits]
    )
    binding_history = []

    for step in range(steps):
        binding_events = sample_binding_sites(binding_probabilities)
        binding_history.append(binding_events)

    return binding_history
