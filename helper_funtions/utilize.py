import numpy as np
def bresenham_line(x1, y1, x2, y2, step_factor):
    # **step_factor 倍の細かさで補間**
    interp_x = np.linspace(x1, x2, step_factor + 1)
    interp_y = np.linspace(y1, y2, step_factor + 1)

    return list(zip(interp_x, interp_y))


def interpolate_line(x1, y1, x2, y2, num_points=10):
    """2点間を線形補間して小数点を含む座標を生成"""
    points = []
    for t in np.linspace(0, 1, num_points):
        x_interp = x1 + t * (x2 - x1)
        y_interp = y1 + t * (y2 - y1)
        points.append((x_interp, y_interp))
    return points



# .perimeter(周囲長の計算)
def calculate_perimeter(vertices):
    n = vertices.shape[0]
    perimeter = 0.0
    for i in range(n):
        j = (i + 1) % n
        dx = vertices[j, 0] - vertices[i, 0]
        dy = vertices[j, 1] - vertices[i, 1]
        perimeter += np.sqrt(dx**2 + dy**2)
    return perimeter


# .細胞の体積(E_volume)(式３)
def volume_energy(cell_area, V0, A_volume, H):
    cell_volume = cell_area * H  # 定義されてた高さをかけて体積計算
    return A_volume * (cell_volume - V0) ** 2


# .細胞の表面積（E_surface）(式４)
def surface_energy(cell_area, S0, A_surface):
    return A_surface * (cell_area - S0) ** 2

# .ラプラシアンの計算
def laplacian(C, dx, dy):
    term_x = (np.roll(C, -1, axis=0) - 2 * C + np.roll(C, 1, axis=0)) / dx**2
    term_y = (np.roll(C, -1, axis=1) - 2 * C + np.roll(C, 1, axis=1)) / dy**2
    return term_x + term_y