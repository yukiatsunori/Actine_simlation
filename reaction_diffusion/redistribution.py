import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from geometry.geometry_process import calculate_area



# .細胞外をゼロに設定し、細胞内を正規化
def initialize_concentrations(
    vertices,
    cell_mask,
    C_Factin_init,
    C_Gactin_init,
    C_CP_init,
    C_cofilin_init,
    C_Arp_init,
):
    # 全体をゼロで初期化C_CP
    C_Factin = np.zeros_like(cell_mask, dtype=float)
    C_Gactin = np.zeros_like(cell_mask, dtype=float)
    C_CP = np.zeros_like(cell_mask, dtype=float)
    C_cofilin = np.zeros_like(cell_mask, dtype=float)
    C_Arp = np.zeros_like(cell_mask, dtype=float)
    # 細胞内の格子点数
    num_inside_cells = np.sum(cell_mask)
    if num_inside_cells > 0:
        cell_positions = np.argwhere(cell_mask == 1)
        # 細胞の面積を計算
        cell_area = calculate_area(vertices)
        # 細胞内全体の濃度
        F_actin_total = C_Factin_init * cell_area
        G_actin_total = C_Gactin_init * cell_area
        C_CP_total = C_CP_init * cell_area
        C_cofilin_total = C_cofilin_init * cell_area
        C_Arp_total = C_Arp_init * cell_area
        # 細胞の外側からの距離マップを作成F_actin_total = F_actin_init * cell_area
        distance_map = distance_transform_edt(cell_mask)
        decay_factor = 0.1  # 減衰の強さ（調整可能）
        # 距離が一定以下の部分を膜とする（例えば、距離が1以下なら膜）
        membrane_mask = (distance_map <= 2) & cell_mask
        decay_factor_map = np.exp(-decay_factor * distance_map)
        sum_factor = np.sum(decay_factor_map[cell_mask == 1])

        for y, x in cell_positions:
            factor = decay_factor_map[y, x]
            # 正規化して全体のG-アクチン濃度を維持
            C_Factin[y, x] = F_actin_total / sum_factor
            C_Gactin[y, x] = (G_actin_total / sum_factor) * factor
            C_CP[y, x] = (C_CP_total / sum_factor) * factor
            C_cofilin[y, x] = (C_cofilin_total / sum_factor) * factor
            C_Arp[y, x] = (C_Arp_total / sum_factor) * factor

    return C_Factin, C_Gactin, C_CP, C_cofilin, C_Arp


def redistribute_concentration(C, cell_mask):

    # 元の濃度をコピー
    new_C = np.zeros_like(C)

    # もともと細胞内だった座標を取得
    old_mask = C > 0

    # 新しい細胞領域で、以前も細胞内だった部分は元の値を保持
    new_C[cell_mask & old_mask] = C[cell_mask & old_mask]

    # 細胞内の新たな部分（以前は細胞外だった）
    new_inside = cell_mask & ~old_mask

    if np.any(new_inside):

        smoothed_C = gaussian_filter(C, sigma=1)  # 平滑化
        new_C[new_inside] = smoothed_C[new_inside]  # 平滑化した値を適用

    return new_C