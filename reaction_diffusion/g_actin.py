import numpy as np


def compute_G_actin_reaction_with_filaments(actin_filaments, C_Gactin, Acom, H, NA):
    # G-アクチンの反応項を初期化
    reaction_Gactin = np.zeros_like(C_Gactin)
    Vcom = Acom * H

    for filament in actin_filaments:
        # **フィラメントの開始点・終了点を整数に変換して取得**
        y_start, x_start = int(filament["y"]), int(filament["x"])
        y_end, x_end = int(filament["y_out"]), int(filament["x_out"])
        y_start_prev, x_start_prev = int(filament["prev_y"]), int(filament["prev_x"])
        y_end_prev, x_end_prev = int(filament["prev_y_out"]), int(
            filament["prev_x_out"]
        )
        barbed_growth = filament["barbed_growth"]  # barbedendのGアクチンの変化量
        pointed_growth = filament["pointed_growth"]  # pointedendのGアクチンの変化量

        if barbed_growth > 0:  # 脱重合によりGアクチン増加
            reaction_Gactin[y_end_prev, x_end_prev] += barbed_growth * NA**-1 * Vcom**-1
        elif barbed_growth < 0:
            reaction_Gactin[y_end, x_end] -= abs(barbed_growth) * NA**-1 * Vcom**-1

        if pointed_growth > 0:
            reaction_Gactin[y_start_prev, x_start_prev] += (
                pointed_growth * NA**-1 * Vcom**-1
            )
        elif pointed_growth < 0:
            reaction_Gactin[y_start, x_start] -= abs(pointed_growth) * NA**-1 * Vcom**-1

    return reaction_Gactin


def update_G_actin(C_Gactin, reaction_Gactin, diffusion_Gactin, Dt):

    # 各格子セルごとにG-アクチン濃度を更新
    C_Gactin_new = C_Gactin + Dt * (reaction_Gactin + diffusion_Gactin)

    # G-アクチン濃度が負にならないようにする
    C_Gactin_new = np.maximum(C_Gactin_new, 0)

    return C_Gactin_new


# 反応拡散方程式
# .反射境界条件を適用する関数
def apply_reflective_boundary(C):
    # 左右の境界
    C[0, :] = C[1, :]  # 左側
    C[-1, :] = C[-2, :]  # 右側
    # 上下の境界
    C[:, 0] = C[:, 1]  # 上側
    C[:, -1] = C[:, -2]  # 下側
    return C
