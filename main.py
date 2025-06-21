from pathlib import Path

import numpy as np

from actin.filaments import generate_actin_filaments, update_actin_filaments
from config.config import InputModel
from geometry.membrane import (
    generate_circle_vertices,
    gradient_descent_with_history,
    update_membrane_by_actin,
)
from geometry.utils import clip_vertices, generate_cell_mask
from helper_funtions.binding_prob import (
    calculate_binding_probability,
    generate_spatial_distribution,
    simulate_F_actin_dynamics,
)
from helper_funtions.creating_initial_state import initialize_grid_and_mask
from helper_funtions.utilize import laplacian
from reaction_diffusion.Arp import compute_Arp_reac, update_Arp
from reaction_diffusion.cofilin import compute_cofilin_reac, update_cofilin
from reaction_diffusion.cp import compute_CP_reac, update_CP
from reaction_diffusion.f_actin import (
    compute_Factin_reac,
    get_Factin_positions,
    update_Factin,
    update_Factin_concentration,
)
from reaction_diffusion.g_actin import (
    apply_reflective_boundary,
    compute_G_actin_reaction_with_filaments,
    update_G_actin,
)
from reaction_diffusion.redistribution import (
    initialize_concentrations,
    redistribute_concentration,
)

input_model = InputModel()

# 格子点の座標と細胞内マスクを初期化
x, y, X, Y, cell_mask, NA, Acom, cell_center, cell_radius = initialize_grid_and_mask(
    input_model
)


# 日時でユニークな保存フォルダを作成
save_dir = Path("data")
save_dir.mkdir(parents=True, exist_ok=True)

# 空のリストを用意
C_Gactin_data = []
C_Factin_data = []
C_Arp_data = []
C_CP_data = []
C_cofilin_data = []
num_filaments_data = []
# 頂点座標の更新をアニメーション化するための保存リスト
vertices_history = []
actin_filaments_history = []
actin_filaments = []

# シミュレーションパラメータの設定
params = (
    input_model.V0,
    input_model.S0,
    input_model.A_volume,
    input_model.A_surface,
    input_model.A_bending,
    input_model.lambda_,
    input_model.KBT,
    input_model.dx,
    input_model.dy,
    input_model.Nx,
    input_model.Ny,
)

# C_sbpの生成
num_subunits = int(input_model.F_actin_length / input_model.G_actin_length)
C_sbp_distribution = generate_spatial_distribution(num_subunits, mean_concentration=1.0)
# サブユニットごとの結合確率
binding_probabilities = np.array(
    [
        calculate_binding_probability(
            input_model.K_bindarp + input_model.K_bindcof,
            C_sbp_distribution[i],
            input_model.Dt,
        )
        for i in range(num_subunits)
    ]
)
# 結合確率
binding_history = simulate_F_actin_dynamics(
    input_model.F_actin_length,
    input_model.G_actin_length,
    input_model.K_bindarp + input_model.K_bindcof,
    input_model.C_Arp_init + input_model.C_cofilin_init,
    input_model.Dt,
    input_model.steps,
)
# 以下アルゴリズム通り
# .初期膜の形状
vertices = generate_circle_vertices(cell_center, cell_radius, input_model.num_points)
vertices_history = [vertices.copy()]
# .細胞内外マスクを設定CP_init)C_Factin,
cell_mask = generate_cell_mask(
    vertices, input_model.Nx, input_model.Ny, input_model.num_points
)


# .初期濃度を設定
C_Factin, C_Gactin, C_CP, C_cofilin, C_Arp = initialize_concentrations(
    vertices,
    cell_mask,
    input_model.C_Factin_init,
    input_model.C_Gactin_init,
    input_model.C_CP_init,
    input_model.C_cofilin_init,
    input_model.C_Arp_init,
)
C_Factin_data = [C_Factin.copy()]
C_Gactin_data = [C_Gactin.copy()]
C_CP_data = [C_CP.copy()]
C_cofilin_data = [C_cofilin.copy()]
C_Arp_data = [C_Arp.copy()]
cofilin_state_grid = np.zeros_like(C_cofilin, dtype=int)
arp_state_grid = np.zeros_like(C_Arp, dtype=int)
arp_state_bunki = np.zeros_like(C_Arp, dtype=int)
# アクチンフィラメントの初期設定
prev_actin = generate_actin_filaments(
    input_model.num_filaments,
    input_model.G_actin_length,
    C_Gactin,
    cell_mask,
    vertices,
)
actin_filaments_history.append([filament.copy() for filament in prev_actin])
actin_filaments = prev_actin.copy()
vertices = update_membrane_by_actin(
    vertices, actin_filaments, cell_mask, input_model.dx, input_model.dy
)
vertices_history = [vertices.copy()]
num_filaments = input_model.num_filaments  # 初期のアクチンフィラメント数
# 反応拡散と膜ダイナミクスの更新
for t in range(input_model.steps):
    print(f"steps = {t}")

    # 膜がアクチンフィラメントに押し出される
    vertices = update_membrane_by_actin(
        vertices, actin_filaments, cell_mask, input_model.dx, input_model.dy
    )

    # 膜ダイナミクスの更新
    vertices = gradient_descent_with_history(
        vertices,
        actin_filaments,
        params,
        eta=3e-6,
        r=0.9,
        epsilon=1e-4,
        max_iter=500,
        max_resets=10,
        H=input_model.H,
    )

    # クリッピング（範囲外の座標の修正）
    vertices = clip_vertices(
        vertices, input_model.dx, input_model.dy, input_model.Nx, input_model.Ny
    )
    # 膜形状を記録

    # print(f"ステップ {t}: vertices_history の長さ = {len(vertices_history)}")
    # print(f"vertices[{t}] = {vertices}")
    # 細胞内外のマスクを再生成
    cell_mask = generate_cell_mask(
        vertices, input_model.Nx, input_model.Ny, input_model.num_points
    )

    # **濃度の再分布を行う**
    C_Gactin = redistribute_concentration(C_Gactin, cell_mask)
    C_Factin = redistribute_concentration(C_Factin, cell_mask)
    C_CP = redistribute_concentration(C_CP, cell_mask)
    C_cofilin = redistribute_concentration(C_cofilin, cell_mask)
    C_Arp = redistribute_concentration(C_Arp, cell_mask)

    # Gアクチンの反応拡散方程式
    laplacian_Gactin = laplacian(C_Gactin, input_model.dx, input_model.dy)
    diffusion_Gactin = input_model.D_Gactin * laplacian_Gactin
    reaction_Gactin = compute_G_actin_reaction_with_filaments(
        actin_filaments, C_Gactin, Acom, input_model.H, NA
    )
    C_Gactin = update_G_actin(
        C_Gactin, reaction_Gactin, diffusion_Gactin, input_model.Dt
    )
    C_Gactin = apply_reflective_boundary(C_Gactin)
    C_Gactin[~cell_mask] = 0

    # Fアクチンの反応拡散
    Factin_positions = get_Factin_positions(
        actin_filaments, input_model.Nx, input_model.Ny
    )  # 前回のアクチンフィラメントの位置
    C_Factin = update_Factin_concentration(
        C_Factin,
        actin_filaments,
        input_model.Nx,
        input_model.Ny,
        Acom,
        input_model.H,
        NA,
    )  # Fアクチンの存在する座標にアクチン１個分の濃度をプラス
    laplacian_Factin = laplacian(C_Factin, input_model.dx, input_model.dy)
    diffusion_Factin = 0
    reac_Factin = compute_Factin_reac(
        actin_filaments, C_Factin, Acom, input_model.H, NA
    )
    C_Factin = update_Factin(C_Factin, reac_Factin, diffusion_Factin, input_model.Dt)
    C_Factin = apply_reflective_boundary(C_Factin)
    C_Factin[~cell_mask] = 0

    # C_CPの反応拡散方程式
    laplacian_CP = laplacian(C_CP, input_model.dx, input_model.dy)
    diffusion_CP = input_model.D_CP * laplacian_CP
    reac_CP = compute_CP_reac(actin_filaments, C_CP, Acom, input_model.H, NA)
    C_CP = update_CP(C_CP, reac_CP, diffusion_CP, input_model.Dt)
    C_CP = apply_reflective_boundary(C_CP)
    C_CP[~cell_mask] = 0

    # コフィリンの反応拡散方程式
    laplacian_cofilin = laplacian(C_cofilin, input_model.dx, input_model.dy)
    diffusion_cofilin = input_model.D_cofilin * laplacian_cofilin
    reac_cofilin = compute_cofilin_reac(
        C_cofilin, cell_mask, cofilin_state_grid, Acom, input_model.H, NA
    )
    C_cofilin = update_cofilin(
        C_cofilin, reac_cofilin, diffusion_cofilin, input_model.Dt
    )
    C_cofilin = apply_reflective_boundary(C_cofilin)
    C_cofilin[~cell_mask] = 0

    # Arp2/3の反応拡散方程式
    laplacian_Arp = laplacian(C_Arp, input_model.dx, input_model.dy)
    diffusion_Arp = input_model.D_Arp * laplacian_Arp
    reac_Arp = compute_Arp_reac(
        C_Arp, cell_mask, arp_state_grid, Acom, input_model.H, NA
    )
    C_Arp = update_Arp(C_Arp, reac_Arp, diffusion_Arp, input_model.Dt)
    C_Arp = apply_reflective_boundary(C_Arp)
    C_Arp[~cell_mask] = 0

    # C_Arp_data.append(C_Arp.copy())
    # アクチンフィラメントの長さを更新
    (
        current_actin,
        cofilin_state_grid,
        num_filaments,
        arp_state_grid,
        arp_state_bunki,
    ) = update_actin_filaments(
        actin_filaments,
        num_filaments,
        vertices,
        C_Gactin,
        C_Factin,
        C_CP,
        C_cofilin,
        cofilin_state_grid,
        arp_state_grid,
        arp_state_bunki,
        C_Arp,
        input_model.F_actin_length,
        input_model.G_actin_length,
        input_model.dx,
        input_model.dy,
        input_model.Dt,
        cell_mask,
        input_model.K_bindarp,
        input_model.K_actarp,
        input_model.K_inactarp,
        input_model.K_bindcof,
        input_model.K_unbindcof,
        input_model.K_sev,
        input_model.K_unbindcp,
        input_model.K_actcp,
        input_model.K_inactcp,
        input_model.K_bindcp,
        input_model.K_polB,
        input_model.K_polP,
        input_model.K_depolB,
        input_model.K_depolP,
    )
    # 例：ループ末尾
    filename = save_dir / f"step_{t:04d}.npz"
    np.savez_compressed(
        filename,
        vertices=vertices,
        C_Gactin=C_Gactin,
        C_Factin=C_Factin,
        C_CP=C_CP,
        C_cofilin=C_cofilin,
        C_Arp=C_Arp,
        actin_filaments=current_actin,
        num_filaments=num_filaments,
    )
    print(f"📝 saved: {filename}")

    """
    # データの型を確認
    actin_filaments_history.append([filament.copy() for filament in current_actin])
    if input_model.steps % 10 == 0:
        vertices_history.append(vertices.copy())
        C_Gactin_data.append(C_Gactin.copy())
        C_Factin_data.append(C_Factin.copy())
        C_CP_data.append(C_CP.copy())
        C_cofilin_data.append(C_cofilin.copy())
        C_Arp_data.append(C_Arp.copy())
    """

    actin_filaments = current_actin.copy()

    print(f"{num_filaments}")
