import numpy as np


# .F-actin(E_Factin)
def factin_energy(actin_filaments, lambda_, kBT):
    energy = 0
    E_factin = 0

    for actin_list in actin_filaments:
        if isinstance(actin_list, list):
            actin = actin_list[0]  # 最初の要素を取得（辞書）
        else:
            actin = actin_list  # 既に辞書ならそのまま使用
        length = actin["length"]  # ℓ_i
        x, y = actin["x"], actin["y"]
        x_out, y_out = actin["x_out"], actin["y_out"]
        theta = np.radians(actin["theta"])
        delta_x = actin["delta_x"]

        # ゼロ除算を防ぐためのチェック
        if delta_x == 0:
            continue
        theta_safe = np.maximum(np.abs(theta), 1e-10)  # 0 に近い値を防ぐ
        # エネルギー計算（修正: `length` ではなく `new_length` を使用）
        energy = np.where(
            np.abs(np.tan(theta_safe)) < 1e6,
            ((4 * lambda_ * kBT) / (length**3 * np.tan(theta_safe) ** 2)) * delta_x**2,
            0,
        )

        E_factin += energy

    return E_factin


def get_Factin_positions(actin_filaments, Nx, Ny):
    """
    アクチンフィラメントの座標を取得し、Fアクチンが存在するセルのインデックスをリスト化
    """
    Factin_positions = set()  # Fアクチンの座標を保存する集合（重複を防ぐ）

    for filament in actin_filaments:
        # フィラメントの開始・終了座標を取得
        x_start, y_start = filament["prev_x"], filament["prev_y"]
        x_end, y_end = filament["prev_x_out"], filament["prev_y_out"]

        # フィラメント全体を補間するための点数
        num_points = int(np.ceil(filament["prev_length"]))

        # 開始点から終了点までの座標を補間（小数点を考慮）
        x_values = np.linspace(x_start, x_end, num_points)
        y_values = np.linspace(y_start, y_end, num_points)

        for x, y in zip(x_values, y_values):
            xi, yi = int(round(x)), int(round(y))  # 近い整数座標にマッピング
            if 0 <= xi < Nx and 0 <= yi < Ny:
                Factin_positions.add((yi, xi))  # Fアクチンの座標として保存

    return Factin_positions  # 座標リストを返す


def update_Factin_concentration(C_Factin, actin_filaments, Nx, Ny, Acom, H, NA):
    """
    F-アクチンの濃度を更新（アクチンフィラメントの座標に基づく）
    """
    Vcom = Acom * H

    # Fアクチンの座標を取得
    Factin_positions = get_Factin_positions(actin_filaments, Nx, Ny)

    # 各座標に対してFアクチンのカウントを増やす
    for yi, xi in Factin_positions:
        if 0 <= yi < C_Factin.shape[0] and 0 <= xi < C_Factin.shape[1]:
            C_Factin[yi, xi] += NA**-1 * Vcom**-1  # **特定の座標にのみ適用**

    return C_Factin  # 更新された F-アクチン濃度を返す


def compute_Factin_reac(actin_filaments, C_Factin, Acom, H, NA):
    # F-アクチンの反応項を初期化Factin[y_end, x_end] += F_barbed_growth *NA**-1 * Vcom**-1
    reaction_Factin = np.zeros_like(C_Factin)
    Vcom = Acom * H
    for filament in actin_filaments:
        # **フィラメントの開始点・終了点を整数に変換して取得**
        y_start, x_start = int(filament["y"]), int(filament["x"])
        y_end, x_end = int(filament["y_out"]), int(filament["x_out"])
        y_start_prev, x_start_prev = int(filament["prev_y"]), int(filament["prev_x"])
        y_end_prev, x_end_prev = int(filament["prev_y_out"]), int(
            filament["prev_x_out"]
        )
        F_barbed_growth = filament["F_barbed_growth"]
        F_pointed_growth = filament["F_pointed_growth"]

        if F_barbed_growth > 0:  # **重合**
            reaction_Factin[y_end, x_end] += F_barbed_growth * NA**-1 * Vcom**-1
        elif F_barbed_growth < 0:  # **脱重合**
            reaction_Factin[y_end_prev, x_end_prev] -= (
                abs(F_barbed_growth) * NA**-1 * Vcom**-1
            )

        # **Pointed End の処理**
        if F_pointed_growth > 0:  # **重合**
            reaction_Factin[y_start, x_start] += F_pointed_growth * NA**-1 * Vcom**-1
        elif F_pointed_growth < 0:  # **脱重合**
            reaction_Factin[y_start_prev, x_start_prev] -= (
                abs(F_pointed_growth) * NA**-1 * Vcom**-1
            )
    return reaction_Factin


def update_Factin(C_Factin, reac_Factin, diffusion_Factin, Dt):
    C_Factin_new = C_Factin.copy()
    C_Factin_new = C_Factin + Dt * (reac_Factin + diffusion_Factin)
    C_Factin_new = np.maximum(C_Factin_new, 0)
    return C_Factin_new
