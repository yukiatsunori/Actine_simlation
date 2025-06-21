import numpy as np


# .cell_areaの計算
def calculate_area(vertices):
    n = vertices.shape[0]
    area = 0
    for i in range(n):
        j = (i + 1) % n  # 次の頂点（最後の頂点と最初の頂点を閉じる）
        term = vertices[i, 0] * vertices[j, 1] - vertices[j, 0] * vertices[i, 1]
        area += term
    return abs(area) / 2.0


# 曲率の計算
def calculate_curvatures(vertices):
    vertices = np.array(vertices)  # NumPy配列に変換
    n = vertices.shape[0]  # 頂点数
    curvatures = np.zeros(n)  # 曲率を格納する配列

    for i in range(n):
        # 前後の頂点を取得
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n

        # ベクトルを計算
        vec1 = vertices[i] - vertices[prev_idx]
        vec2 = vertices[next_idx] - vertices[i]

        # ベクトルの長さを計算
        len1 = np.linalg.norm(vec1)
        len2 = np.linalg.norm(vec2)

        # 長さがゼロなら曲率もゼロ
        if len1 < 1e-10 or len2 < 1e-10:
            curvatures[i] = 1e10
            continue

        # 内積から角度を計算
        cos_theta = np.dot(vec1, vec2) / (len1 * len2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 数値誤差を防ぐ
        theta = np.arccos(cos_theta)  # ラジアンでの角度

        # 曲率半径 R を計算
        if np.abs(theta) < 1e-5:  # ほぼ直線なら曲率をゼロ
            curvatures[i] = 1e10
        else:
            R = (len1 + len2) / (2 * np.sin(theta / 2))
            if R > 1e10:  # R が異常に大きい場合はゼロとする
                curvatures[i] = 1e10
            else:
                curvatures[i] = R  # 曲率 κ = 1/R

    return curvatures
