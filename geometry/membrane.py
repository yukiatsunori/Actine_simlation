import numpy as np
from geometry.utils import is_outside_cell, is_same_grid
from geometry.geometry_process import (
    calculate_area,
    calculate_curvatures,
)

from utils import (
    total_energy,
    grad_x,
    grad_y,
    clip_vertices,
)

# .頂点の初期状態（均等な円となるように頂点を配置）
def generate_circle_vertices(center, radius, num_points):

    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)  # 均等な角度

    vertices = np.array(
        [
            (center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle))
            for angle in angles
        ]
    )
    # print(f"vertices={vertices}")
    return vertices


def update_membrane_by_actin(vertices, actin_filaments, cell_mask, dx, dy):
    new_vertices = vertices.copy()

    # 細胞の中心を計算
    cx, cy = np.mean(vertices, axis=0)

    for actin_list in actin_filaments:
        if isinstance(actin_list, list):
            actin = actin_list[0]  # 最初の要素を取得（辞書）
        else:
            actin = actin_list  # 既に辞書ならそのまま使用

        x_out, y_out = float(actin["x_out"]), float(actin["y_out"])
        x_start, y_start = float(actin["x"]), float(actin["y"])

        mem_x_out, mem_y_out = int(x_out), int(y_out)
        mem_x_start, mem_y_start = int(x_start), int(y_start)

        out_outside = is_outside_cell(mem_x_out, mem_y_out, cell_mask)
        start_outside = is_outside_cell(mem_x_start, mem_y_start)

        # 最も近い膜の頂点を検索
        distances_out = np.sqrt(
            (vertices[:, 0] - x_out) ** 2 + (vertices[:, 1] - y_out) ** 2
        )
        distances_start = np.sqrt(
            (vertices[:, 0] - x_start) ** 2 + (vertices[:, 1] - y_start) ** 2
        )

        closest_idx_out = np.argmin(distances_out)
        closest_idx_start = np.argmin(distances_start)

        # もし out または start が細胞膜外なら頂点を移動
        if out_outside or start_outside:
            dist_out = (
                np.sqrt((x_out - cx) ** 2 + (y_out - cy) ** 2) if out_outside else -1
            )
            dist_start = (
                np.sqrt((x_start - cx) ** 2 + (y_start - cy) ** 2)
                if start_outside
                else -1
            )
            vertex_dist_out = np.sqrt(
                (vertices[closest_idx_out, 0] - cx) ** 2
                + (vertices[closest_idx_out, 1] - cy) ** 2
            )
            vertex_dist_start = np.sqrt(
                (vertices[closest_idx_start, 0] - cx) ** 2
                + (vertices[closest_idx_start, 1] - cy) ** 2
            )

            if dist_out > vertex_dist_out or dist_start > vertex_dist_start:
                if dist_out > dist_start:
                    target_x, target_y = x_out, y_out
                    closest_idx = closest_idx_out
                else:
                    target_x, target_y = x_start, y_start
                    closest_idx = closest_idx_start

                new_vertices[closest_idx, 0] = target_x
                new_vertices[closest_idx, 1] = target_y

        # もし膜の頂点と同じグリッド内にアクチンがいたら、頂点を更新
        elif is_same_grid(
            x_out,
            y_out,
            vertices[closest_idx_out, 0],
            vertices[closest_idx_out, 1],
            dx,
            dy,
        ):
            new_vertices[closest_idx_out, 0] = x_out
            new_vertices[closest_idx_out, 1] = y_out

        elif is_same_grid(
            x_start,
            y_start,
            vertices[closest_idx_start, 0],
            vertices[closest_idx_start, 1],
            dx,
            dy,
        ):
            new_vertices[closest_idx_start, 0] = x_start
            new_vertices[closest_idx_start, 1] = y_start

    return new_vertices

# .勾配降下法による位置の更新（式９）
def gradient_descent_with_history(
    vertices,
    actin_filaments,
    params,
    eta,
    r,
    epsilon,
    max_iter,
    max_resets,
    H
):
    original_vertices = vertices.copy()  # 元の座標を保存（リセット時に使用）
    original_cell_area = calculate_area(vertices)  # 元の面積を保存
    curvatures = calculate_curvatures(vertices)  # 元の曲率を保存
    original_curvatures = curvatures.copy()
    eta_reset = eta  # リセット時に η も元の値に戻す
    delta = 1e-5  # 勾配計算のための微小値
    for reset in range(max_resets):  # 収束しなかったらリセットしてやり直す
        vertices = original_vertices.copy()
        cell_area = original_cell_area
        curvatures = original_curvatures.copy()
        eta = eta_reset  # リセット時に η も元の値に戻す
        for iteration in range(max_iter):
            new_vertices = np.copy(vertices)
            current_energy = total_energy(
                cell_area, vertices, curvatures, actin_filaments, params, H
            )
            # x, y方向の勾配を計算
            grad_x_vals = grad_x(
                vertices, cell_area, curvatures, actin_filaments, params, delta, H
            )
            grad_y_vals = grad_y(
                vertices, cell_area, curvatures, actin_filaments, params, delta, H
            )
            # ✅ **デバッグ: 勾配を確認**
            # print(f"Iteration {iteration}: grad_x_vals = {grad_x_vals}")
            # print(f"Iteration {iteration}: grad_y_vals = {grad_y_vals}")
            # print(f"Iteration {iteration}: max |grad_x| = {np.max(np.abs(grad_x_vals))}, max |grad_y| = {np.max(np.abs(grad_y_vals))}")

            # print(f"Iteration {iteration}:  max Δy = {np.max(np.abs(eta * grad_y_vals))}")
            # 各頂点の位置を更新
            # print(f"Iteration {iteration}: max |update_x| = {np.max(np.abs(eta * grad_x_vals))}, max |update_y| = {np.max(np.abs(eta * grad_y_vals))}")

            new_vertices[:, 0] = vertices[:, 0] - (eta * grad_x_vals)
            new_vertices[:, 1] = vertices[:, 1] - (eta * grad_y_vals)

            new_vertices = clip_vertices(
                new_vertices, params[7], params[8], params[9], params[10]
            )  # 修正: クリッピングを適用
            # print(f"Iteration {iteration}: new_vertices = {new_vertices}")
            # 新しい面積,曲率を再計算
            new_cell_area = calculate_area(new_vertices)
            # print(f"Iteration {iteration}: new_cell_area = {new_cell_area}")
            new_curvatures = calculate_curvatures(new_vertices)
            # print(f"Iteration {iteration}: new_curvatures = {new_curvatures}")

            # 新しいエネルギーを計算
            new_energy = total_energy(
                new_cell_area, new_vertices, new_curvatures, actin_filaments, params, H
            )
            # print(f"current_energy = {current_energy}, new_energy= {new_energy}")
            print(
                f"Iteration {iteration}: energy difference = {abs(new_energy - current_energy)}"
            )

            # 頂点座標を更新（エネルギーが減少した場合のみ適用）
            if float(new_energy) < float(current_energy):
                vertices = new_vertices.copy()
                cell_area = new_cell_area
                curvatures = new_curvatures.copy()
            else:
                eta *= r

            # 収束判定（エネルギーの変化が十分小さい場合）
            if abs(float(new_energy) - float(current_energy)) < epsilon:
                print(f"Converged at iteration {iteration}")
                return vertices
        print(f"Did not converge in {max_iter} iterations, resetting...")

    print("Max resets reached, returning last computed vertices.")

    return vertices

# .膜の曲率（E_bending）(式11)
def bending_energy(vertices, curvatures, A_bending):
    vertices = np.array(vertices)
    curvatures = np.array(calculate_curvatures(vertices))
    n = vertices.shape[0]
    energy = 0.0
    for i in range(n):
        # 前後の頂点インデックスを取得
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n
        # 現在の頂点、前後の頂点の座標
        xi = vertices[i]
        xi_prev = vertices[prev_idx]
        xi_next = vertices[next_idx]
        # ベクトルの長さを計算
        length_next = np.linalg.norm(xi_next - xi)
        length_prev = np.linalg.norm(xi_prev - xi)
        weight = (length_next + length_prev) / 2.0  # 長さの平均
        # 曲率エネルギーの計算
        Ri = curvatures[i]
        Ri_safe = np.maximum(Ri, 1e-10)  # Riが1e-10未満のときは1e-10に置き換える
        # print(f"{Ri}")
        energy += (1 / Ri_safe) ** 2 * weight
    return A_bending * energy

