import numpy as np

def update_cofilin(C_cofilin, reac_cofilin, diffusion_cofilin, Dt):
    # 各格子セルごとにG-アクチン濃度を更新
    C_cofilin_new = C_cofilin + Dt * (reac_cofilin + diffusion_cofilin)

    # G-アクチン濃度が負にならないようにする
    C_cofilin_new = np.maximum(C_cofilin_new, 0)

    return C_cofilin_new



def compute_cofilin_reac(C_cofilin, cell_mask, cofilin_state_grid):
    reaction_cofilin = np.zeros_like(C_cofilin)
    Vcom = Acom * H
    num_inside_cells = np.sum(cell_mask)
    if num_inside_cells > 0:
        cell_positions = np.argwhere(cell_mask == 1)
        for y, x in cell_positions:
            if (
                0 <= y < cofilin_state_grid.shape[0]
                and 0 <= x < cofilin_state_grid.shape[1]
            ):
                if cofilin_state_grid[y, x] == 1:  # 解離したとき
                    reaction_cofilin[y, x] -= 1 * NA**-1 * Vcom**-1  # 減る
                elif cofilin_state_grid[y, x] == -1:  # 結合したとき
                    reaction_cofilin[y, x] += 1 * NA**-1 * Vcom**-1  # 増える

    return reaction_cofilin