import numpy as np
from matplotlib.path import Path
from helper_funtions.utilize import bresenham_line, interpolate_line, volume_energy, surface_energy
from geometry_process import calculate_area, calculate_curvatures
from membrane import  bending_energy
from reaction_diffusion.f_actin import factin_energy


# .細胞内外の判定
def generate_cell_mask(vertices, Nx, Ny, num_points):
    mask = np.zeros((Ny, Nx), dtype=bool)  # 初期は全て細胞外（False）
    # ポリゴン内部（細胞内）判定
    poly_path = Path(vertices)
    for y in range(Ny):
        for x in range(Nx):
            # **四隅のどれかがポリゴン内に含まれているかをチェック**
            corners = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
            inside_corners = sum(poly_path.contains_point(corner) for corner in corners)
            if inside_corners >= 2:  # **4点中3点以上が細胞内なら細胞内と判定**
                mask[y, x] = True

    # 頂点間の線分を細胞外に設定
    for i in range(len(vertices)):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % len(vertices)]
        bresenham_points = bresenham_line(x1, y1, x2, y2, step_factor=20)  # ここで定義
        interpolated_points = []
        for j in range(len(bresenham_points) - 1):
            x_a, y_a = bresenham_points[j]
            x_b, y_b = bresenham_points[j + 1]
            interpolated_points.extend(
                interpolate_line(x_a, y_a, x_b, y_b, num_points)
            )  # 5分割補間

        for fx, fy in interpolated_points:
            cx = int(np.floor(fx))  # x座標の左下の整数値
            cy = int(np.floor(fy))  # y座標の左下の整数値

            if 0 <= cx < Nx and 0 <= cy < Ny:
                mask[cy, cx] = True
    return mask



# 細胞膜の範囲外チェック
def is_outside_cell(x, y, cell_mask):
    return (
        0 <= x < cell_mask.shape[1]
        and 0 <= y < cell_mask.shape[0]
        and not cell_mask[y, x]
    )


# 頂点と同じグリッド内にいるかチェック
def is_same_grid(x, y, vertex_x, vertex_y, dx, dy):
    return (int(x // dx) == int(vertex_x // dx)) and (
        int(y // dy) == int(vertex_y // dy)
    )


# .エネルギー関数E(E_volume+E_surface+E_bending+E_Factin)
def total_energy(cell_area, vertices, curvatures, actin_filaments, params, H):
    V0, S0, A_volume, A_surface, A_bending, lambda_, kBT = params[:7]

    # 各エネルギー項の計算

    e_volume = volume_energy(cell_area, V0, A_volume, H)
    e_surface = surface_energy(cell_area, S0, A_surface)
    e_bending = bending_energy(vertices, curvatures, A_bending)
    #e_factin = factin_energy(actin_filaments, vertices, cell_mask, lambda_, kBT)
    total = e_bending  # e_volume + e_surface + e_bending +e_factin
    # print(f"e_volume={e_volume}, e_surface={e_surface}, E_bending={e_bending}, e_factin={e_factin}")
    # print(f"Total energy: {total}")
    return float(np.sum(total))


# .δE/δxiの計算（式10）
def grad_x(vertices, cell_area, curvatures, actin_filaments, params, delta, H):
    grad_x_list = np.zeros(len(vertices))
    energy = float(
        np.sum(total_energy(cell_area, vertices, curvatures, actin_filaments, params,H))
    )
    for i, (x, y) in enumerate(vertices):
        x_plus = np.array(vertices, dtype=float).copy()  # 明示的に浮動小数点型にする
        x_minus = np.array(vertices, dtype=float).copy()
        x_plus[i, 0] += delta  # x座標のみ変更
        x_minus[i, 0] -= delta
        new_cell_area_plus = calculate_area(x_plus)  # 面積更新
        new_cell_area_minus = calculate_area(x_minus)
        new_curvatures_plus = calculate_curvatures(x_plus)  # 曲率更新
        new_curvatures_minus = calculate_curvatures(x_minus)
        # x調整後のエネルギー計算
        energy_plus = float(
            np.sum(
                total_energy(
                    new_cell_area_plus,
                    x_plus,
                    new_curvatures_plus,
                    actin_filaments,
                    params,
                    H,
                )
            )
        )
        energy_minus = float(
            np.sum(
                total_energy(
                    new_cell_area_minus,
                    x_minus,
                    new_curvatures_minus,
                    actin_filaments,
                    params,
                    H,
                )
            )
        )
        # 計算結果の保存
        grad_x_list[i] = (energy_plus - energy_minus) / (2 * delta)
    return grad_x_list


# .δE/δyiの計算
def grad_y(vertices, cell_area, curvatures, actin_filaments, params, delta, H):
    grad_y_list = np.zeros(len(vertices))
    energy = float(
        np.sum(total_energy(cell_area, vertices, curvatures, actin_filaments, params, H))
    )
    for i, (x, y) in enumerate(vertices):
        y_plus = np.array(vertices, dtype=float)
        y_minus = np.array(vertices, dtype=float)
        y_plus[i, 1] += delta  # y座標のみ変更
        y_minus[i, 1] -= delta
        new_cell_area_plus = calculate_area(y_plus)  # 面積更新
        new_cell_area_minus = calculate_area(y_minus)
        new_curvatures_plus = calculate_curvatures(y_plus)  # 曲率更新
        new_curvatures_minus = calculate_curvatures(y_minus)
        # y調整後のエネルギー計算
        energy_plus = float(
            np.sum(
                total_energy(
                    new_cell_area_plus,
                    y_plus,
                    new_curvatures_plus,
                    actin_filaments,
                    params,
                    H,
                )
            )
        )
        energy_minus = float(
            np.sum(
                total_energy(
                    new_cell_area_minus,
                    y_minus,
                    new_curvatures_minus,
                    actin_filaments,
                    params,
                    H,
                )
            )
        )
        # 計算結果の保存
        grad_y_list[i] = (energy_plus - energy_minus) / (2 * delta)
    return grad_y_list


# .頂点のクリッピング関数
def clip_vertices(vertices, dx, dy, Nx, Ny):
    max_x = Nx * dx
    max_y = Ny * dy
    clipped_vertices = []
    for x, y in vertices:
        clipped_x = max(0, min(x, max_x))
        clipped_y = max(0, min(y, max_y))
        # if x != clipped_x or y != clipped_y:
        # print(f"Clipping applied: ({x}, {y}) -> ({clipped_x}, {clipped_y})")
        clipped_vertices.append((clipped_x, clipped_y))
    return np.array(clipped_vertices)