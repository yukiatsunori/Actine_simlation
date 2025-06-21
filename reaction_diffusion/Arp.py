


def compute_Arp_reac(C_Arp, cell_mask, arp_state_grid):
    reaction_Arp = np.zeros_like(C_Arp)
    Vcom = Acom * H
    num_inside_cells = np.sum(cell_mask)
    if num_inside_cells > 0:
        cell_positions = np.argwhere(cell_mask == 1)
        for y, x in cell_positions:
            if 0 <= y < arp_state_grid.shape[0] and 0 <= x < arp_state_grid.shape[1]:
                if arp_state_grid[y, x] == 1:  # 解離したとき
                    reaction_Arp[y, x] -= 1 * NA**-1 * Vcom**-1  # 減る
                elif arp_state_grid[y, x] == -1:  # 結合したとき
                    reaction_Arp[y, x] += 1 * NA**-1 * Vcom**-1  # 増える

    return reaction_Arp



def update_Arp(C_Arp, reac_Arp, diffusion_Arp, Dt):
    # 各格子セルごとにG-アクチン濃度を更新
    C_Arp_new = C_Arp + Dt * (reac_Arp + diffusion_Arp)

    # G-アクチン濃度が負にならないようにする
    C_Arp_new = np.maximum(C_Arp_new, 0)

    return C_Arp_new