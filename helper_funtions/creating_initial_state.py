# functions/grid_initializer.py

import numpy as np


def initialize_grid_and_mask(model):
    """
    格子点の座標と細胞内マスクを初期化

    Parameters
    ----------
    model : InputModel
        格子サイズ・空間解像度などを含むモデル

    Returns
    -------
    x, y : np.ndarray
        格子点の1D座標（X軸、Y軸）
    X, Y : np.ndarray
        格子点の2D座標グリッド（meshgrid）
    cell_mask : np.ndarray[bool]
        細胞内部と外部をTrue/Falseで分けたマスク
    NA : float
        アボガドロ数（定数）
    Acom : float
        体積あたりの定数（μm）
    """
    # 格子座標
    x = np.linspace(0, model.Nx, model.Ny)
    y = np.linspace(0, model.Ny, model.Nx)

    # 細胞中心とマスク
    cell_center = (model.Nx // 2, model.Ny // 2)
    cell_radius = 10.0  # μm
    X, Y = np.meshgrid(np.arange(model.Nx), np.arange(model.Ny))
    cell_mask = (X - cell_center[0])**2 + (Y - cell_center[1])**2 <= cell_radius**2

    # 定数
    NA = 6.022e23
    Acom = 0.4

    return x, y, X, Y, cell_mask, NA, Acom, cell_center, cell_radius
