import numpy as np

from helper_funtions.utilize import bending_energy, surface_energy, volume_energy


# .エネルギー関数E(E_volume+E_surface+E_bending+E_Factin)
def total_energy(cell_area, vertices, curvatures, actin_filaments, params, H):
    V0, S0, A_volume, A_surface, A_bending, lambda_, kBT = params[:7]

    # 各エネルギー項の計算

    e_volume = volume_energy(cell_area, V0, A_volume, H)
    e_surface = surface_energy(cell_area, S0, A_surface)
    e_bending = bending_energy(vertices, curvatures, A_bending)
    # e_factin = factin_energy(actin_filaments, vertices, cell_mask, lambda_, kBT)
    total = e_bending  # e_volume + e_surface + e_bending +e_factin
    # print(f"e_volume={e_volume}, e_surface={e_surface}, E_bending={e_bending}, e_factin={e_factin}")
    # print(f"Total energy: {total}")
    return float(np.sum(total))
