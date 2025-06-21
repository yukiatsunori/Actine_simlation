import os
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# -----------------------------
# 可視化関数群
# -----------------------------


data_dir = "data"  # 適宜パスを修正
data_file = os.path.join(data_dir, "step_0000.npz")

if not os.path.exists(data_file):
    raise FileNotFoundError(f"{data_file} が見つかりません")

data = np.load(data_file, allow_pickle=True)
print("含まれているキー:", data.files)

# 各データのサイズ確認
for key in data.files:
    print(f"{key}: {type(data[key])}, shape={np.shape(data[key])}")


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


# -----------------------------
# ローディング関数
# -----------------------------


def load_simulation_data(data_dir):
    files = sorted(Path(data_dir).glob("step_*.npz"))
    vertices_history = []
    C_Gactin_data = []
    C_CP_data = []
    C_Factin_data = []
    C_cofilin_data = []
    C_Arp_data = []
    actin_filaments_history = []

    for file in files:
        data = np.load(file, allow_pickle=True)
        vertices_history.append(data["vertices"])
        C_Gactin_data.append(data["C_Gactin"])
        C_CP_data.append(data["C_CP"])
        C_Factin_data.append(data["C_Factin"])
        C_cofilin_data.append(data["C_cofilin"])
        C_Arp_data.append(data["C_Arp"])
        actin_filaments_history.append(data["actin_filaments"])

    Nx = int(np.max(vertices_history[0][:, 0]) + 5)
    Ny = int(np.max(vertices_history[0][:, 1]) + 5)
    return (
        vertices_history,
        C_Gactin_data,
        C_CP_data,
        C_Factin_data,
        C_cofilin_data,
        C_Arp_data,
        actin_filaments_history,
        Nx,
        Ny,
    )


# -----------------------------
# 実行部分
# -----------------------------

if __name__ == "__main__":

    data_dir = "data"  # 保存された npz ファイルが入っているディレクトリ
    (
        vertices_history,
        C_Gactin_data,
        C_CP_data,
        C_Factin_data,
        C_cofilin_data,
        C_Arp_data,
        actin_filaments_history,
        Nx,
        Ny,
    ) = load_simulation_data(data_dir)

    # それぞれコメントアウトを外すことで可視化
    animate_vertices(vertices_history, Nx, Ny)
    animate_Gactin(
        C_Gactin_data, C_CP_data, C_Factin_data, C_cofilin_data, C_Arp_data, Nx, Ny
    )
    plot_final_step(actin_filaments_history, vertices_history, Nx, Ny)
    animate_actinfilaments(actin_filaments_history, vertices_history, Nx, Ny)
    animate_combined(vertices_history, C_Gactin_data, Nx, Ny)
    # plot_cell_with_grid_and_mask(vertices_history, cell_mask, Nx, Ny)
    num_filaments_history(actin_filaments_history)
    plot_average_filament_length(actin_filaments_history)
