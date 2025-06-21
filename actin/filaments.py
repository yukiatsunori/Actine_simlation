import numpy as np


# アクチンフィラメントの初期設定
def generate_actin_filaments(
    num_filaments, G_actin_length, C_Gactin, cell_mask, vertices
):
    initial_actin = []
    cell_positions = np.argwhere(cell_mask == 1)

    # **G-アクチン濃度に基づく重みを作成**
    weights = C_Gactin[cell_positions[:, 0], cell_positions[:, 1]]
    weights /= np.sum(weights)

    for i in range(num_filaments):
        # **G-アクチン濃度に比例した確率で座標を選択**
        y_int, x_int = cell_positions[np.random.choice(len(cell_positions), p=weights)]
        y = y_int + np.random.uniform(-0.5, 0.5)  # -0.5 〜 +0.5 のランダムシフト
        x = x_int + np.random.uniform(-0.5, 0.5)
        # **アクチンフィラメントの方向を決定（ランダムな角度）**
        theta = np.random.uniform(0, 2 * np.pi)

        # フィラメントの長さを F-アクチン濃度に応じて決定
        # F_actin_length = G_actin_length*2  # 最小Gアクチン長
        F_actin_length = G_actin_length * 2

        # 法線方向に成長（膜の内側の長さ + 膜の外の長さ）
        x_out = x + (F_actin_length * np.cos(theta))
        y_out = y + (F_actin_length * np.sin(theta))

        # **膜の内外を判定**
        i_start, j_start = int(round(y)), int(round(x))
        i_end, j_end = int(round(y_out)), int(round(x_out))

        is_start_inside = (
            cell_mask[i_start, j_start]
            if (0 <= i_start < cell_mask.shape[0] and 0 <= j_start < cell_mask.shape[1])
            else False
        )
        is_end_inside = (
            cell_mask[i_end, j_end]
            if (0 <= i_end < cell_mask.shape[0] and 0 <= j_end < cell_mask.shape[1])
            else False
        )

        # **もし両方とも膜内なら delta_x = 0**
        if is_start_inside and is_end_inside:
            delta_x = 0
        else:
            # 膜外の点を決定
            if not is_start_inside:
                outside_x, outside_y = x, y
                inside_x, inside_y = x_out, y_out
            else:
                outside_x, outside_y = x_out, y_out
                inside_x, inside_y = x, y

            # **最近傍の膜内の頂点を探す**
            vertices_array = np.array(vertices)  # 頂点のリスト
            distances = np.linalg.norm(
                vertices_array - np.array([outside_x, outside_y]), axis=1
            )
            nearest_vertex = vertices_array[np.argmin(distances)]  # 最も近い膜内頂点

            # **delta_x を計算**
            sa = np.linalg.norm(nearest_vertex - np.array([inside_x, inside_y]))
            delta_x = max(F_actin_length - sa, 0)  # 長さが負にならないように制限

        # フィラメントデータを辞書に保存
        initial_actin.append(
            {
                "prev_x": x,
                "prev_y": y,
                "x": x,
                "y": y,
                "prev_x_out": x_out,
                "prev_y_out": y_out,
                "x_out": x_out,
                "y_out": y_out,
                "prev_length": F_actin_length,
                "length": F_actin_length,
                "theta": theta,
                "delta_x": delta_x,
                "age": 0,
                "barbed_growth": 0,
                "pointed_growth": 0,
                "cap": 0,
                "cap_growth": 0,
                "F_barbed_growth": 0,
                "F_pointed_growth": 0,
                "bunki": 0,
            }
        )

    return initial_actin


def update_actin_filaments(
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
    F_actin_length,
    G_actin_length,
    dx,
    dy,
    Dt,
    cell_mask,
    K_bindarp,
    K_actarp,
    K_inactarp,
    K_bindcof,
    K_unbindcof,
    K_sev,
    K_unbindcp,
    K_actcp,
    K_inactcp,
    K_bindcp,
    K_polB,
    K_polP,
    K_depolB,
    K_depolP,
):
    updated_filaments = []
    filaments = actin_filaments.copy()
    C_cofilin = C_cofilin.copy()
    new_num_filaments = num_filaments
    deleted_filament_count = 0
    # 既存フィラメントの寿命を考慮
    for i in range(num_filaments):
        if isinstance(filaments, list):
            actin = filaments[i]  # 最初の要素を取得（辞書）
        else:
            actin = filaments  # 既に辞P_cap = 書ならそのまま使用

        new_age = actin["age"] + 1  # 時間経過
        actin_theta = actin["theta"]
        actin_x = actin["x"]
        actin_y = actin["y"]
        actin_x_out = actin["x_out"]
        actin_y_out = actin["y_out"]
        actin_length = actin["length"]
        actin_bunki = actin["bunki"]
        actin_cap = actin["cap"]

        # **F-アクチンの長さをGアクチンの単位で分割**
        num_segments = int(
            np.ceil(actin_length / G_actin_length)
        )  # Gアクチン単位で区切る
        # **分割された点を生成**
        x_values = np.linspace(actin_x, actin_x_out, num_segments)
        y_values = np.linspace(actin_y, actin_y_out, num_segments)

        """
        Arp2/3によるアクチンフィラメントの変化
        """
        for x, y in zip(x_values, y_values):
            i, j = int(y), int(x)  # グリッド上の整数座標にマッピング
            a = x
            b = y
            C_Factin_local = C_Factin[i, j]
            C_Arp_local = C_Arp[i, j]
            # コフィリンの値を更新
            if cofilin_state_grid[i, j] == 1:  # 前回離れてる
                cofilin_state_grid[i, j] = 0
            elif cofilin_state_grid[i, j] == -1:  # 前回結合した
                cofilin_state_grid[i, j] -= -1
            # arp2/3の値を更新
            if arp_state_grid[i, j] == 1:  # 前回離れてる
                arp_state_grid[i, j] = 0
            elif arp_state_grid[i, j] == -1:  # 前回結合した
                arp_state_grid[i, j] -= 1
            # **Arp2/3 のサイド結合確率**が
            P_bindArp = K_bindarp * C_Arp_local * Dt
            # **Arp2/3 の活性化確率**
            P_activateArp = K_actarp * Dt
            # **Arp2/3 の不活性化確率**
            P_inactArp = K_inactarp * Dt
            # print(f"{P_bindArp}, {P_activateArp}, {P_inactArp}")

            if (
                cofilin_state_grid[i, j] == 0 and arp_state_grid[i, j] == 0
            ):  # コフィリンとArp2/3が結合していないとき、
                if np.random.rand() < P_bindArp:  # Arpが結合する
                    # **フィラメントの途中に Arp2/3 を結合**
                    arp_state_grid[i, j] = -1  # 結合した
            if arp_state_grid[i, j] < 0:  # Arp2/3が結合しているとき
                if arp_state_bunki[i, j] == 1:  # 分岐しているとき
                    break
                elif arp_state_bunki[i, j] == 0:  # 分岐していないとき
                    if np.random.rand() < P_inactArp:  # Arp2/3が解離する
                        arp_state_grid[i, j] = 1  # 解離した
                        arp_state_bunki[i, j] = 0
                        continue
                    elif (
                        np.random.rand() < P_activateArp
                    ):  # Arp2/3が活性化したとき新しく分岐のフィラメント
                        arp_state_bunki[i, j] = 1
                        bunki_x = a
                        bunki_y = b
                        bunki_length = G_actin_length * 2
                        bunki_theta = actin_theta + np.radians(
                            np.random.normal(loc=70, scale=10)
                        )
                        bunki_x_end = bunki_x + (bunki_length * np.cos(bunki_theta))
                        bunki_y_end = bunki_y + (bunki_length * np.sin(bunki_theta))
                        i_start, j_start = int(bunki_y), int(bunki_x)
                        i_end, j_end = int(bunki_y_end), int(bunki_x_end)
                        is_start_inside = (
                            cell_mask[i_start, j_start]
                            if (
                                0 <= i_start < cell_mask.shape[0]
                                and 0 <= j_start < cell_mask.shape[1]
                            )
                            else False
                        )
                        is_end_inside = (
                            cell_mask[i_end, j_end]
                            if (
                                0 <= i_end < cell_mask.shape[0]
                                and 0 <= j_end < cell_mask.shape[1]
                            )
                            else False
                        )

                        if is_start_inside and is_end_inside:
                            bunki_delta_x = 0
                        else:
                            if not is_start_inside:
                                outside_x, outside_y = bunki_x, bunki_y
                                inside_x, inside_y = bunki_x_end, bunki_y_end
                            else:
                                outside_x, outside_y = bunki_x_end, bunki_y_end
                                inside_x, inside_y = bunki_x, bunki_y

                            vertices_array = np.array(vertices)  # 頂点のリスト
                            distances = np.linalg.norm(
                                vertices_array - np.array([outside_x, outside_y]),
                                axis=1,
                            )
                            nearest_vertex = vertices_array[
                                np.argmin(distances)
                            ]  # 最も近い膜内頂点

                            sa = np.linalg.norm(
                                nearest_vertex - np.array([inside_x, inside_y])
                            )
                            bunki_delta_x = max(
                                F_actin_length - sa, 0
                            )  # 長さが負にならないように制限

                        bunki = actin.copy()
                        bunki["prev_x"] = bunki_x
                        bunki["prev_y"] = bunki_y
                        bunki["x"] = bunki_x
                        bunki["y"] = bunki_y
                        bunki["prev_x_out"] = bunki_x_end
                        bunki["prev_y_out"] = bunki_y_end
                        bunki["x_out"] = bunki_x_end
                        bunki["y_out"] = bunki_y_end
                        bunki["length"] = bunki_length
                        bunki["theta"] = bunki_theta
                        bunki["delta_x"] = bunki_delta_x
                        bunki["age"] = 0
                        bunki["barbed_growth"] = 0
                        bunki["pointed_growth"] = 0
                        bunki["cap"] = 0
                        bunki["cap_growth"] = 0
                        bunki["F_barbed_growth"] = 0
                        bunki["F_pointed_growth"] = 0
                        bunki["bunki"] = 1
                        updated_filaments.append(bunki)
                        new_num_filaments += 1
                        print(f"bunki")

                        continue

        """
        コフィリンによるアクチンフィラメントの変化
        """
        for x, y in zip(x_values, y_values):
            i, j = int(y), int(x)  # グリッド上の整数座標にマッピング
            C_Factin_cof = C_Factin[i, j]
            C_cofilin_cof = C_cofilin[i, j]
            # **コフィリンの結合・解離確率を計算（各セルごとに）**
            # **コフィリンの結合・解離確率を計算（各セルごとに）**
            P_bind_cof = (
                K_bindcof * C_cofilin_cof * Dt
            )  # **すでに結合している場合は増加しない*
            P_unbind_cof = K_unbindcof * Dt  # **すでに結合している場合のみ適用**
            P_sever = K_sev * Dt
            # print(f"{P_bind_cof},{P_unbind_cof}, {P_sever}")#かくりつひくすぎ

            if (
                cofilin_state_grid[i, j] == 0 and arp_state_grid[i, j] == 0
            ):  # コフィリンとArp2/3が結合していないとき、
                if cofilin_state_grid[i, j] == 0:  # コフィリンが離れてるとき
                    if np.random.rand() < P_bind_cof:  # **確率的に結合**
                        cofilin_state_grid[i, j] = -1  # 結合した
                if cofilin_state_grid[i, j] < 0:  # コフィリンが結合しているとき
                    if np.random.rand() < P_unbind_cof:  # **確率的に解離**
                        cofilin_state_grid[i, j] = 1  # 解離した
                if (
                    cofilin_state_grid[i, j] < 0
                ):  # コフィリンが結合しているとき以下で確率的に切断される
                    if np.random.rand() < P_sever:
                        x_cofilin = x  # 既に格子点として処理済み
                        y_cofilin = y
                        # **切断位置を計算**
                        cut_length = np.sqrt(
                            (x_cofilin - actin["x"]) ** 2
                            + (y_cofilin - actin["y"]) ** 2
                        )  # フィラメント始点からコフィリン位置までの距離
                        new_theta = actin_theta + np.radians(np.random.uniform(-10, 10))
                        # **新しいフィラメントの終点を設定**
                        new_x_out = x_cofilin + (cut_length * np.cos(new_theta))
                        new_y_out = y_cofilin + (cut_length * np.sin(new_theta))

                        new_num_segments = int(np.ceil(cut_length / G_actin_length))
                        new_x_values = np.linspace(
                            x_cofilin, new_x_out, new_num_segments
                        )
                        new_y_values = np.linspace(
                            y_cofilin, new_y_out, new_num_segments
                        )

                        # 新しい位置ともとの位置の濃度とstateを交換
                        for new_x, new_y in zip(new_x_values, new_y_values):
                            new_i, new_j = int(new_y), int(
                                new_x
                            )  # グリッド上の整数座標にマッピング
                            C_cofilin_cof = C_cofilin[i, j]
                            cofilin_state_grid = cofilin_state_grid[i, j]
                            C_arp_cof = C_Arp[i, j]
                            arp_state_grid = arp_state_grid[i, j]
                            arp_state_bunki = arp_state_bunki[i, j]

                            new_C_cofilin_cof = C_cofilin[new_i, new_j]
                            new_cofilin_state_grid = cofilin_state_grid[new_i, new_j]
                            new_C_arp_cof = C_Arp[new_i, new_j]
                            new_are_state_grid = arp_state_grid[new_i, new_j]
                            new_arp_state_bunki = arp_state_bunki[new_i, new_j]

                            C_cofilin[new_i, new_j] = C_cofilin_cof
                            C_cofilin[i, j] = new_C_cofilin_cof
                            cofilin_state_grid[new_i, new_j] = cofilin_state_grid
                            cofilin_state_grid[i, j] = new_cofilin_state_grid
                            C_Arp[new_i, new_j] = C_arp_cof
                            C_Arp[i, j] = new_C_arp_cof
                            arp_state_grid[new_i, new_j] = arp_state_grid
                            arp_state_grid[i, j] = new_are_state_grid
                            arp_state_bunki[new_i, new_j] = arp_state_bunki
                            arp_state_bunki[i, j] = new_arp_state_bunki

                        if cut_length < G_actin_length * 2:
                            deleted_filament_count += 1  # フィラメントの総数を減少
                            new_num_filaments -= 1
                            continue  # 削除するので更新リストに追加しない

                        # **膜の内外を判定**
                        i_start, j_start = int(y_cofilin), int(x_cofilin)
                        i_end, j_end = int(new_y_out), int(new_x_out)

                        is_start_inside = (
                            cell_mask[i_start, j_start]
                            if (
                                0 <= i_start < cell_mask.shape[0]
                                and 0 <= j_start < cell_mask.shape[1]
                            )
                            else False
                        )
                        is_end_inside = (
                            cell_mask[i_end, j_end]
                            if (
                                0 <= i_end < cell_mask.shape[0]
                                and 0 <= j_end < cell_mask.shape[1]
                            )
                            else False
                        )

                        # **もし両方とも膜内なら delta_x = 0**
                        if is_start_inside and is_end_inside:
                            cofilin_delta_x = 0
                        else:
                            # 膜外の点を決定
                            if not is_start_inside:
                                outside_x, outside_y = x_cofilin, y_cofilin
                                inside_x, inside_y = new_x_out, new_y_out
                            else:
                                outside_x, outside_y = new_x_out, new_y_out
                                inside_x, inside_y = x_cofilin, y_cofilin

                        cofilin_cap = actin_cap
                        actin_cap = 0

                        # **最近傍の膜内の頂点を探す**
                        vertices_array = np.array(vertices)  # 頂点のリスト
                        distances = np.linalg.norm(
                            vertices_array - np.array([outside_x, outside_y]), axis=1
                        )
                        nearest_vertex = vertices_array[
                            np.argmin(distances)
                        ]  # 最も近い膜内頂点
                        # **delta_x を計算**
                        sa = np.linalg.norm(
                            nearest_vertex - np.array([inside_x, inside_y])
                        )
                        cofilin_delta_x = max(
                            cut_length - sa, 0
                        )  # 長さが負にならないように制限

                        if actin_x_out != x_cofilin or actin_y_out != y_cofilin:
                            before_cap = C_CP[int(actin_y_out), int(actin_x_out)]
                            after_cap = C_CP[int(y_cofilin), int(x_cofilin)]
                            C_CP[int(actin_y_out), int(actin_x_out)] = after_cap
                            C_CP[int(y_cofilin), int(x_cofilin)] = before_cap

                        new_filament = actin.copy()
                        new_filament["prev_x"] = x_cofilin
                        new_filament["prev_y"] = y_cofilin
                        new_filament["x"] = x_cofilin
                        new_filament["y"] = y_cofilin
                        new_filament["prev_x_out"] = new_x_out
                        new_filament["prev_y_out"] = new_y_out
                        new_filament["x_out"] = new_x_out
                        new_filament["y_out"] = new_y_out
                        new_filament["length"] = cut_length
                        new_filament["theta"] = new_theta
                        new_filament["delta_x"] = cofilin_delta_x
                        new_filament["age"] = 0
                        new_filament["barbed_growth"] = 0
                        new_filament["pointed_growth"] = 0
                        new_filament["cap"] = cofilin_cap
                        new_filament["cap_growth"] = 0
                        new_filament["F_barbed_growth"] = 0
                        new_filament["F_pointed_growth"] = 0
                        new_filament["bunki"] = actin_bunki
                        updated_filaments.append(new_filament)

                        actin_length -= cut_length
                        actin_x_out = x_cofilin
                        actin_y_out = y_cofilin
                        new_num_filaments += 1

                        print(f"cofilin")

                    break

        """
        キャッピングタンパク質による変化、ポリ、デポリ含む
        """
        P_unbindCP = K_unbindcp * Dt
        C_CP_active = (
            (K_inactcp - K_actcp) * C_CP[int(actin_y_out), int(actin_x_out)]
        ) * Dt
        P_CP_active = (K_inactcp - K_actcp) * Dt
        P_capB = K_bindcp * C_CP_active * Dt

        cap_growth = 0

        # print(f"{P_CP_active}, {P_capB}")
        if actin_cap == 1:  # キャッピング中
            if np.random.rand() < P_unbindCP:
                actin_cap = 0
                cap_growth += 1  # キャッピング解離したら増える
        elif actin_cap == 0:  # キャッピングしてない
            if np.random.rand() < P_capB:
                actin_cap = 1
                cap_growth -= 1  # キャッピング結合したら減る

        # **ポリメライゼーションとデポリメライゼーションの確率を計算**
        if actin_cap == 0:
            P_polyB = (
                K_polB * C_Gactin[int(actin_y_out), int(actin_x_out)] * Dt
            )  # 結合確率 #1を超える
            P_depolyB = K_depolB * Dt  # 解離確率
            # print(f"{P_polyB},{P_depolyB}")
        elif actin_cap == 1:
            P_polyB = 0
            P_depolyB = 0

        if actin_bunki == 0:
            P_polyP = K_polP * (C_Gactin[int(actin_x), int(actin_y)]) * Dt  # 結合確率
            P_depolyP = K_depolP * Dt  # 解離確率
            # print(f"{P_polyP}, {P_depolyP}")
        elif actin_bunki == 1:
            P_polyP = 0
            P_depolyP = 0

        # **伸長 or 縮小を G-actin の単位で決定**
        delta_lengthB = 0
        delta_lengthP = 0
        barbed_growth = 0
        pointed_growth = 0
        F_barbed_growth = 0
        F_pointed_growth = 0
        if 0 <= i < C_Factin.shape[0] and 0 <= j < C_Factin.shape[1]:
            # if np.random.rand() < P_polyB:  # 確率的にポリメライゼーションが起こる
            if True:
                delta_lengthB += G_actin_length
                barbed_growth -= 1  # 重合によりG→FでG→１つ減る
                F_barbed_growth += 1  # 重合によりFが増える

            if np.random.rand() < P_depolyB:  # 確率的にデポリメライゼーションが起こる
                delta_lengthB -= G_actin_length
                barbed_growth += 1
                F_barbed_growth -= 1
            if np.random.rand() < P_polyP:  # 確率的にポリメライゼーションが起こる
                delta_lengthP += G_actin_length
                pointed_growth -= 1
                F_pointed_growth += 1
            # if np.random.rand() < P_depolyP:  # 確率的にデポリメライゼーションが起こる
            # delta_lengthP -= G_actin_length
            # pointed_growth += 1
            # F_pointed_growth -= 1

        # **最小長以下にはならないようにする**
        new_length = actin_length + delta_lengthB + delta_lengthP
        # print(f"{new_length},{actin_length }, {delta_lengthB }, {delta_lengthP}")

        if new_length < G_actin_length * 2:
            deleted_filament_count += 1  # フィラメントの総数を減少
            new_num_filaments -= 1
            continue  # 削除するので更新リストに追加しな

        # **開始位置の更新**
        x_new = actin_x + (-delta_lengthP * np.cos(actin_theta))
        y_new = actin_y + (-delta_lengthP * np.sin(actin_theta))

        # 伸長後の新しい座標を計算
        actin_x_out_new = x_new + (new_length * np.cos(actin_theta))
        actin_y_out_new = y_new + (new_length * np.sin(actin_theta))

        # **膜の内外を判定**
        i_start, j_start = int(y_new), int(x_new)
        i_end, j_end = int(actin_y_out_new), int(actin_x_out_new)

        is_start_inside = (
            cell_mask[i_start, j_start]
            if (0 <= i_start < cell_mask.shape[0] and 0 <= j_start < cell_mask.shape[1])
            else False
        )
        is_end_inside = (
            cell_mask[i_end, j_end]
            if (0 <= i_end < cell_mask.shape[0] and 0 <= j_end < cell_mask.shape[1])
            else False
        )

        # **もし両方とも膜内なら delta_x = 0**
        if is_start_inside and is_end_inside:
            new_delta_x = 0
        else:
            # 膜外の点を決定
            if not is_start_inside:
                outside_x, outside_y = x_new, y_new
                inside_x, inside_y = actin_x_out_new, actin_y_out_new
            else:
                outside_x, outside_y = actin_x_out_new, actin_y_out_new
                inside_x, inside_y = x_new, y_new
            # **最近傍の膜内の頂点を探す**
            vertices_array = np.array(vertices)  # 頂点のリスト
            distances = np.linalg.norm(
                vertices_array - np.array([outside_x, outside_y]), axis=1
            )
            nearest_vertex = vertices_array[np.argmin(distances)]  # 最も近い膜内頂点

            # **delta_x を計算**
            sa = np.linalg.norm(nearest_vertex - np.array([inside_x, inside_y]))
            new_delta_x = max(F_actin_length - sa, 0)  # 長さが負にならないように制限

        if actin_x_out != actin_x_out_new or actin_y_out != actin_y_out_new:
            before_cap = C_CP[int(actin_y_out), int(actin_x_out)]
            after_cap = C_CP[int(actin_y_out_new), int(actin_x_out_new)]
            C_CP[int(actin_y_out), int(actin_x_out)] = after_cap
            C_CP[int(actin_y_out_new), int(actin_x_out_new)] = before_cap

        # if actin["age"] < actin["max_age"]:  # 最大寿命を超えていなければ保持
        updated_filaments.append(
            {
                "prev_x": actin_x,
                "prev_y": actin_y,
                "x": x_new,  # **開始位置を保存**
                "y": y_new,
                "prev_x_out": actin_x_out,
                "prev_y_out": actin_y_out,
                "x_out": actin_x_out_new,
                "y_out": actin_y_out_new,
                "prev_length": actin_length,
                "length": new_length,
                "theta": actin_theta,  # **そのまま継承**
                "delta_x": new_delta_x,
                "age": new_age,
                "barbed_growth": barbed_growth,
                "pointed_growth": pointed_growth,
                "cap": actin_cap,
                "cap_growth": cap_growth,
                "F_barbed_growth": F_barbed_growth,
                "F_pointed_growth": F_pointed_growth,
                "bunki": actin_bunki,
            }
        )

    if deleted_filament_count > 0:
        new_filaments = generate_actin_filaments(
            deleted_filament_count, G_actin_length, C_Gactin, cell_mask
        )
        updated_filaments.extend(new_filaments)  # 新しいフィラメントをリストに追加
        new_num_filaments += deleted_filament_count

    return (
        updated_filaments,
        cofilin_state_grid,
        new_num_filaments,
        arp_state_grid,
        arp_state_bunki,
    )
