import numpy as np


def update_CP(C_CP, reac_CP, diffusion_CP, Dt):

    # 各格子セルごとにG-アクチン濃度を更新
    C_CP_new = C_CP + Dt * (reac_CP + diffusion_CP)

    # G-アクチン濃度が負にならないようにする
    C_CP_new = np.maximum(C_CP_new, 0)

    return C_CP_new


def compute_CP_reac(actin_filaments, C_CP, Acom, H, NA):
    # G-アクチンの反応項を初期化
    reaction_CP = np.zeros_like(C_CP)
    Vcom = Acom * H

    for filament in actin_filaments:
        # キャッピングが結合する終了点を整数に変換して取得**
        y_end, x_end = int(filament["y_out"]), int(filament["x_out"])
        cap_growth = filament["cap_growth"]

        if cap_growth == 1:  # 解離したとき
            reaction_CP[y_end, x_end] -= 1 * NA**-1 * Vcom**-1  # 解離によりCP減少
        elif cap_growth == -1:  # 結合したとき
            reaction_CP[y_end, x_end] += 1 * NA**-1 * Vcom**-1  # 結合によりCP増加

    return reaction_CP
