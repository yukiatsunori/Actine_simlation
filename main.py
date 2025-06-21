import matplotlib
import numpy as np
from config.config import InputModel

from helper_funtions.binding_prob import (
    simulate_F_actin_dynamics,
    calculate_binding_probability,
    generate_spatial_distribution,
)


from helper_funtions.utilize import laplacian

from actin.filaments import (
    generate_actin_filaments,
    update_actin_filaments,
)

from geometry.membrane import generate_circle_vertices, update_membrane_by_actin, gradient_descent_with_history
from geometry.utils import generate_cell_mask, clip_vertices

from reaction_diffusion.redistribution import (
    initialize_concentrations,
    redistribute_concentration,
)

from reaction_diffusion.g_actin import (
    compute_G_actin_reaction_with_filaments,
    update_G_actin,
    apply_reflective_boundary,
)

from reaction_diffusion.f_actin import (
    update_Factin_concentration,
    get_Factin_positions,
    compute_Factin_reac,
    update_Factin,
)

from reaction_diffusion.Arp import (
    compute_Arp_reac,
    update_Arp,
)

from reaction_diffusion.cp import (
    compute_CP_reac,
    update_CP,
)
from reaction_diffusion.cofilin import (
    compute_cofilin_reac,
    update_cofilin,
)

from helper_funtions.creating_initial_state import initialize_grid_and_mask



input_model = InputModel()

# 格子点の座標と細胞内マスクを初期化
x, y, X, Y, cell_mask, NA, Acom, cell_center, cell_radius = initialize_grid_and_mask(input_model)

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
params = (input_model.V0, 
          input_model.S0, 
          input_model.A_volume,
          input_model.A_surface,
          input_model.A_bending, 
          input_model.lambda_, 
          input_model.KBT, 
          input_model.dx, 
          input_model.dy, 
          input_model.Nx, 
          input_model.Ny
)

# C_sbpの生成
num_subunits = int(input_model.F_actin_length / input_model.G_actin_length)
C_sbp_distribution = generate_spatial_distribution(num_subunits, mean_concentration=1.0)
# サブユニットごとの結合確率
binding_probabilities = np.array(
    [
        calculate_binding_probability(input_model.K_bindarp + input_model.K_bindcof, C_sbp_distribution[i], input_model.Dt)
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
cell_mask = generate_cell_mask(vertices, input_model.Nx, input_model.Ny, input_model.num_points)


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
    input_model.num_filaments, input_model.F_actin_length, input_model.G_actin_length, C_Gactin, cell_mask, vertices
)
actin_filaments_history.append([filament.copy() for filament in prev_actin])
actin_filaments = prev_actin.copy()
vertices = update_membrane_by_actin(vertices, actin_filaments, cell_mask, input_model.dx, input_model.dy)
vertices_history = [vertices.copy()]

# 反応拡散と膜ダイナミクスの更新
for t in range(input_model.steps):
    print(f"steps = {t}")

    # 膜がアクチンフィラメントに押し出される
    vertices = update_membrane_by_actin(vertices, actin_filaments, cell_mask, input_model.dx, input_model.dy)

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
        H=input_model.H
    )

    # クリッピング（範囲外の座標の修正）
    vertices = clip_vertices(vertices, dx, dy, Nx, Ny)
    # 膜形状を記録

    # print(f"ステップ {t}: vertices_history の長さ = {len(vertices_history)}")
    # print(f"vertices[{t}] = {vertices}")
    # 細胞内外のマスクを再生成
    cell_mask = generate_cell_mask(vertices, Nx, Ny)

    # **濃度の再分布を行う**
    C_Gactin = redistribute_concentration(C_Gactin, cell_mask)
    C_Factin = redistribute_concentration(C_Factin, cell_mask)
    C_CP = redistribute_concentration(C_CP, cell_mask)
    C_cofilin = redistribute_concentration(C_cofilin, cell_mask)
    C_Arp = redistribute_concentration(C_Arp, cell_mask)

    # Gアクチンの反応拡散方程式
    laplacian_Gactin = laplacian(C_Gactin, dx, dy)
    diffusion_Gactin = D_Gactin * laplacian_Gactin
    reaction_Gactin = compute_G_actin_reaction_with_filaments(actin_filaments, C_Gactin, Acom, H)
    C_Gactin = update_G_actin(C_Gactin, reaction_Gactin, diffusion_Gactin, Dt)
    C_Gactin = apply_reflective_boundary(C_Gactin)
    C_Gactin[~cell_mask] = 0

    # Fアクチンの反応拡散
    Factin_positions = get_Factin_positions(
        actin_filaments, Nx, Ny
    )  # 前回のアクチンフィラメントの位置
    C_Factin = update_Factin_concentration(
        C_Factin, actin_filaments, Nx, Ny
    )  # Fアクチンの存在する座標にアクチン１個分の濃度をプラス
    laplacian_Factin = laplacian(C_Factin, dx, dy)
    diffusion_Factin = 0
    reac_Factin = compute_Factin_reac(actin_filaments, C_Factin)
    C_Factin = update_Factin(C_Factin, reac_Factin, diffusion_Factin, Dt)
    C_Factin = apply_reflective_boundary(C_Factin)
    C_Factin[~cell_mask] = 0

    # C_CPの反応拡散方程式
    laplacian_CP = laplacian(C_CP, dx, dy)
    diffusion_CP = D_CP * laplacian_CP
    reac_CP = compute_CP_reac(actin_filaments, C_CP)
    C_CP = update_CP(C_CP, reac_CP, diffusion_CP, Dt)
    C_CP = apply_reflective_boundary(C_CP)
    C_CP[~cell_mask] = 0

    # コフィリンの反応拡散方程式
    laplacian_cofilin = laplacian(C_cofilin, dx, dy)
    diffusion_cofilin = D_cofilin * laplacian_cofilin
    reac_cofilin = compute_cofilin_reac(C_cofilin, cell_mask, cofilin_state_grid)
    C_cofilin = update_cofilin(C_cofilin, reac_cofilin, diffusion_cofilin, Dt)
    C_cofilin = apply_reflective_boundary(C_cofilin)
    C_cofilin[~cell_mask] = 0

    # Arp2/3の反応拡散方程式
    laplacian_Arp = laplacian(C_Arp, dx, dy)
    diffusion_Arp = D_Arp * laplacian_Arp
    reac_Arp = compute_Arp_reac(C_Arp, cell_mask, arp_state_grid)
    C_Arp = update_Arp(C_Arp, reac_Arp, diffusion_Arp, Dt)
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
        G_actin_length,
        dx,
        dy,
        Dt,
    )
    # データの型を確認
    actin_filaments_history.append([filament.copy() for filament in current_actin])
    if steps % 10 == 0:
        vertices_history.append(vertices.copy())
        C_Gactin_data.append(C_Gactin.copy())
        C_Gactin_data.append(C_Gactin.copy())
        C_Factin_data.append(C_Factin.copy())
        C_CP_data.append(C_CP.copy())
        C_cofilin_data.append(C_cofilin.copy())
        C_cofilin_data.append(C_cofilin.copy())
        C_Arp_data.append(C_Arp.copy())

    actin_filaments = current_actin.copy()

    print(f"{num_filaments}")


# .膜の変形をアニメーションとする
def animate_vertices(vertices_history, Nx, Ny):
    fig, ax = plt.subplots()
    ax.set_xlim(0, Nx)
    ax.set_ylim(0, Ny)
    ax.set_aspect("equal")
    # 初期のプロットを作成
    (line,) = ax.plot([], [], "o-", lw=2)
    step_text = ax.text(
        0.05 * Nx,
        0.95 * Ny,
        "",
        fontsize=12,
        color="black",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    def init():
        line.set_data([], [])
        step_text.set_text("")
        return line, step_text

    def update(frame):
        vertices = vertices_history[frame]
        x = np.append(vertices[:, 0], vertices[0, 0])  # 最初の点を最後に追加
        y = np.append(vertices[:, 1], vertices[0, 1])  # 最初の点を最後に追加
        line.set_data(x, y)
        step_text.set_text(f"Step: {frame + 1}")

        return line, step_text

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(vertices_history),
        init_func=init,
        blit=False,
        interval=20,
    )
    plt.show()


# 膜のアニメーション表示
animate_vertices(vertices_history, Nx, Ny)


# 分子濃度の可視化
def animate_Gactin(
    C_Gactin_data, C_CP_data, C_Factin_data, C_cofilin_data, C_Arp_data, Nx, Ny
):
    fig, axes = plt.subplots(1, 5, figsize=(10, 5))
    fig.subplots_adjust(wspace=1.0)
    ax1, ax2, ax3, ax4, ax5 = axes  # それぞれの軸を取得
    step_text = fig.text(
        0.1,
        0.9,
        "",
        fontsize=12,
        color="black",
        bbox=dict(facecolor="white", alpha=0.7),
    )

    cax3 = ax3.imshow(
        C_Gactin_data[0],
        cmap="viridis",
        vmin=0,
        vmax=np.max(C_Gactin_data),
        origin="lower",
        extent=[0, Nx, 0, Ny],
    )
    fig.colorbar(cax3, ax=ax3)
    ax3.set_title("G-Actin(μM) ..")

    cax2 = ax2.imshow(
        C_CP_data[0],
        cmap="viridis",
        vmin=0,
        vmax=np.max(C_CP_data),
        origin="lower",
        extent=[0, Nx, 0, Ny],
    )
    fig.colorbar(cax2, ax=ax2)
    ax2.set_title("C_CP(μM) ")

    cax1 = ax1.imshow(
        C_Factin_data[0],
        cmap="viridis",
        vmin=0,
        vmax=np.max(C_Factin_data),
        origin="lower",
        extent=[0, Nx, 0, Ny],
    )
    fig.colorbar(cax1, ax=ax1)
    ax1.set_title("F-Actin(μM) ..")

    cax4 = ax4.imshow(
        C_cofilin_data[0],
        cmap="viridis",
        vmin=0,
        vmax=np.max(C_cofilin_data),
        origin="lower",
        extent=[0, Nx, 0, Ny],
    )
    fig.colorbar(cax4, ax=ax4)
    ax4.set_title("cofilin(μM) ..")

    cax5 = ax5.imshow(
        C_Arp_data[0],
        cmap="viridis",
        vmin=0,
        vmax=np.max(C_Arp_data),
        origin="lower",
        extent=[0, Nx, 0, Ny],
    )
    fig.colorbar(cax5, ax=ax5)
    ax5.set_title("Arp2/3(μM) ..")

    def update(frame):
        cax3.set_array(C_Gactin_data[frame])
        cax2.set_array(C_CP_data[frame])
        cax1.set_array(C_Factin_data[frame])
        cax4.set_array(C_cofilin_data[frame])
        cax5.set_array(C_Arp_data[frame])
        step_text.set_text(f"Step: {frame + 1}")

        return [cax1, cax2, cax3, cax4, cax5, step_text]

    anim = FuncAnimation(
        fig, update, frames=len(C_Factin_data), interval=50, blit=False
    )
    plt.show()


animate_Gactin(
    C_Gactin_data, C_CP_data, C_Factin_data, C_cofilin_data, C_Arp_data, Nx, Ny
)


def plot_final_step(actin_filaments_history, vertices_history, Nx, Ny):
    # 図の描画
    fig, ax = plt.subplots()
    ax.set_xlim(0, Nx)
    ax.set_ylim(0, Ny)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_aspect("equal")
    ax.set_title("Final Step of Actin Filaments")

    # 格子線を表示
    for x in range(Nx + 1):
        ax.plot([x, x], [0, Ny], color="gray", linewidth=0.5, alpha=0.5)  # 縦線
    for y in range(Ny + 1):
        ax.plot([0, Nx], [y, y], color="gray", linewidth=0.5, alpha=0.5)  # 横線

    # 最終ステップのデータを取得
    final_frame = len(actin_filaments_history) - 1
    actin_filaments = actin_filaments_history[final_frame]
    vertices = vertices_history[final_frame]

    # 細胞膜を描画
    x = np.append(vertices[:, 0], vertices[0, 0])  # 閉じた形にする
    y = np.append(vertices[:, 1], vertices[0, 1])
    ax.plot(x, y, "o-", color="blue", lw=2, label="Cell Boundary")

    # 各フィラメントを描画
    for filament in actin_filaments:
        x_start = float(filament["x"])
        y_start = float(filament["y"])
        x_end = float(filament["x_out"])
        y_end = float(filament["y_out"])

        ax.plot([x_start, x_end], [y_start, y_end], "r-")  # フィラメントを赤色で描画
        # ax.plot([x_start], [y_start], 'go', markersize=5)  # 開始点（緑色）
        # ax.plot([x_end], [y_end], 'ro', markersize=5)  # 先端（赤色）

    ax.legend()
    plt.show()


plot_final_step(actin_filaments_history, vertices_history, Nx, Ny)


# アクチンフィラメントの可視化
def animate_actinfilaments(actin_filaments_history, vertices_history, Nx, Ny):
    #     図の描画
    fig, ax = plt.subplots()
    ax.set_xlim(0, Nx)
    ax.set_ylim(0, Ny)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_aspect("equal")
    ax.set_title("Actin Filaments")

    # 細胞膜の境界線をプロット（初期状態）
    (vertices_line,) = ax.plot([], [], "o-", color="blue", lw=2)  # 細胞膜の線を赤にする

    # **格子線を表示**
    for x in range(Nx + 1):
        ax.plot([x, x], [0, Ny], color="gray", linewidth=0.5, alpha=0.5)  # 縦線
    for y in range(Ny + 1):
        ax.plot([0, Nx], [y, y], color="gray", linewidth=0.5, alpha=0.5)  # 横線

    # print("actin_filaments_history:", actin_filaments_history)
    # 各フィラメントを描画
    max_filaments = max(
        (len(frame) for frame in actin_filaments_history), default=1
    )  # 全フレームの最大フィラメント数
    lines = [ax.plot([], [], "r-")[0] for _ in range(max_filaments)]
    end_points = [ax.plot([], [], "ro", markersize=5)[0] for _ in range(max_filaments)]
    start_points = [
        ax.plot([], [], "ro", markersize=5)[0] for _ in range(max_filaments)
    ]  # 開始点

    def init():
        vertices_line.set_data([], [])

        for line in lines:
            line.set_data([], [])
        for end_point in end_points:
            end_point.set_data([], [])
        for start_point in start_points:
            start_point.set_data([], [])  # 開始点も初期化

        return lines + start_points + end_points + [vertices_line]

    def update(frame):
        actin_filaments = actin_filaments_history[frame]
        # 必要なフィラメント数をチェック
        num_filaments = len(actin_filaments)
        # **細胞膜の更新**
        vertices = vertices_history[frame]
        x = np.append(vertices[:, 0], vertices[0, 0])  # 閉じた形にする
        y = np.append(vertices[:, 1], vertices[0, 1])
        vertices_line.set_data(x, y)

        for i, filament in enumerate(actin_filaments):

            x_start = float(filament["x"])  # np.int64 → int に変換
            y_start = float(filament["y"])
            x_end = float(filament["x_out"])  # np.float64 → float に変換
            y_end = float(filament["y_out"])

            # スタート地点から先端までプロット
            lines[i].set_data([x_start, x_end], [y_start, y_end])
            # start_points[i].set_data([x_start], [y_start])  # 開始点（緑）
            # end_points[i].set_data([x_end], [y_end])  # 先端（赤）

        return lines + start_points + end_points + [vertices_line]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(actin_filaments_history),
        init_func=init,
        blit=True,
        interval=50,
    )

    plt.show()


animate_actinfilaments(actin_filaments_history, vertices_history, Nx, Ny)


# verticesと濃度
def animate_combined(vertices_history, C_Gactin_data, Nx, Ny):
    fig, ax = plt.subplots()
    ax.set_xlim(0, Nx)
    ax.set_ylim(0, Ny)
    ax.set_aspect("equal")

    # 分子濃度のヒートマップ（最初のフレームを設定）
    cax = ax.imshow(
        C_Gactin_data[0],
        cmap="viridis",
        vmin=0,
        vmax=np.max(C_Gactin_data),
        origin="lower",
        extent=[0, Nx, 0, Ny],
    )
    fig.colorbar(cax)

    # 細胞膜の境界線をプロット（初期状態）
    (line,) = ax.plot([], [], "o-", color="red", lw=2)  # 細胞膜の線を赤にする
    # **格子線を表示**
    for x in range(Nx + 1):
        ax.plot([x, x], [0, Ny], color="gray", linewidth=0.5, alpha=0.5)  # 縦線
    for y in range(Ny + 1):
        ax.plot([0, Nx], [y, y], color="gray", linewidth=0.5, alpha=0.5)  # 横線

    # 初期化関数
    def init():
        line.set_data([], [])
        return cax, line

    # フレームごとの更新関数
    def update(frame):
        # ヒートマップの更新
        cax.set_array(C_Gactin_data[frame])

        # 細胞膜の境界の更新
        vertices = vertices_history[frame]
        x = np.append(
            vertices[:, 0], vertices[0, 0]
        )  # 最初の点を最後に追加（閉じた形にする）
        y = np.append(vertices[:, 1], vertices[0, 1])
        line.set_data(x, y)

        return cax, line

    # アニメーションの作成
    ani = FuncAnimation(
        fig,
        update,
        frames=len(vertices_history),
        init_func=init,
        blit=False,
        interval=50,
    )

    # アニメーションを表示
    plt.show()


# アニメーションを実行
animate_combined(vertices_history, C_Gactin_data, Nx, Ny)


# 細胞内外
def plot_cell_with_grid_and_mask(vertices_history, cell_mask, Nx, Ny):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, Nx)
    ax.set_ylim(0, Ny)
    ax.set_aspect("equal")

    # **細胞内を青色で塗りつぶし**
    cell_inside = cell_mask.astype(float)  # 細胞内を1、それ以外をNaN
    img = ax.imshow(
        cell_inside, cmap="Blues", origin="lower", alpha=0.5, extent=[0, Nx, 0, Ny]
    )
    # 細胞膜の境界線をプロット（初期状態）
    (line,) = ax.plot([], [], "o-", color="red", lw=2)  # 細胞膜の線を赤にする
    # **格子線を表示**
    for x in range(Nx + 1):
        ax.plot([x, x], [0, Ny], color="gray", linewidth=0.5, alpha=0.5)  # 縦線
    for y in range(Ny + 1):
        ax.plot([0, Nx], [y, y], color="gray", linewidth=0.5, alpha=0.5)  # 横線

    # 初期化関数
    def init():
        line.set_data([], [])
        return line, img

    # フレームごとの更新関数
    def update(frame):
        # 細胞膜の境界の更新
        vertices = vertices_history[frame]
        x = np.append(
            vertices[:, 0], vertices[0, 0]
        )  # 最初の点を最後に追加（閉じた形にする）
        y = np.append(vertices[:, 1], vertices[0, 1])
        line.set_data(x, y)

        # 細胞内領域を更新（もし細胞形状が時間とともに変化するなら）
        img.set_data(cell_mask.astype(float))  # 細胞内を1.0、それ以外を0.0 にする

        return line, img

    # アニメーションの作成
    ani = FuncAnimation(
        fig,
        update,
        frames=len(vertices_history),
        init_func=init,
        blit=False,
        interval=50,
    )
    # アニメーションを表示
    plt.show()


plot_cell_with_grid_and_mask(vertices_history, cell_mask, Nx, Ny)


# **アクチンフィラメントの本数**
def num_filaments_history(actin_filaments_history):
    # Time steps based on simulation lengt
    time_steps = np.arange(len(actin_filaments_history))
    num_filaments_over_time = [len(filaments) for filaments in actin_filaments_history]
    print(f"{num_filaments_over_time}")
    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(
        time_steps,
        num_filaments_over_time,
        marker="o",
        linestyle="-",
        color="b",
        alpha=0.7,
    )
    plt.xlabel("Time Step")
    plt.ylabel("Number of Actin Filaments")
    plt.title("Change in Actin Filament Count Over Time")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


num_filaments_history(actin_filaments_history)


def plot_average_filament_length(actin_filaments_history):
    # 各ステップの平均フィラメント長を計算
    average_lengths = []
    time_steps = np.arange(len(actin_filaments_history))

    for filaments in actin_filaments_history:
        if len(filaments) > 0:
            avg_length = np.mean([filament["length"] for filament in filaments])
        else:
            avg_length = 0  # フィラメントが存在しない場合は0
        average_lengths.append(avg_length)

    # 折れ線グラフをプロット
    plt.figure(figsize=(10, 5))
    plt.plot(
        time_steps, average_lengths, marker="o", linestyle="-", color="b", alpha=0.7
    )
    plt.xlabel("Time Step")
    plt.ylabel("Average Filament Length")
    plt.title("Average Actin Filament Length Over Time")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


# 関数の呼び出し
plot_average_filament_length(actin_filaments_history)
