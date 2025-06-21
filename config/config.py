from pydantic import BaseModel


class InputModel(BaseModel):
    # ── シミュレーション領域と格子 ─────────────────────────
    Nx: int = 70  # 格子サイズ
    Ny: int = 70
    dx: float = 1.0  # 格子間隔 (µm)
    dy: float = 1.0
    Dt: float = 9e-4  # 時間ステップ (s)
    D_Factin: float = 0.01  # F-actin 拡散係数 (µm²/s)
    steps: int = 10  # シミュレーション総ステップ
    num_points: int = 50  # 膜頂点数
    total_concentration: float = 10000  # F+G-actin 総濃度
    F_actin_length: float = 1.0  # ⟵ 要確認 (µm?)
    num_filaments: int = 300

    # ── 初期濃度 (µM) ─────────────────────────────────────
    C_Factin_init: float = 0.0
    C_Gactin_init: float = 18.0
    C_Arp_init: float = 0.3
    C_CP_init: float = 2.0
    C_cofilin_init: float = 0.5

    # ── 拡散係数 (µm²/s) ──────────────────────────────────
    D_Gactin: float = 30.0
    D_Arp: float = 30.0
    D_CP: float = 0.1
    D_cofilin: float = 0.1

    # ── 反応速度定数 ─────────────────────────────────────
    K_polB: float = 120.0  # Barbed end ポリメライズ
    K_polP: float = 3.0  # Pointed end ポリメライズ
    K_depolB: float = 1.4
    K_depolP: float = 21.0
    K_bindarp: float = 3.4
    K_bindcp: float = 4.0
    K_unbindcp: float = 0.04
    K_bindcof: float = 0.0085
    K_unbindcof: float = 0.005
    K_sev: float = 0.012
    K_actarp: float = 4.0
    K_inactarp: float = 1.0
    K_actcp: float = 0.04
    K_inactcp: float = 4.0**5  # = 1024.0

    # ── 形状・物性パラメータ ─────────────────────────────
    H: float = 0.2  # 小胞体の高さ (µm)
    delta_Gactin: float = 2.7  # アクチン単量体長 (?)
    shita_Mean: float = 70.0  # 分岐角平均 (°)
    shita_SD: float = 10.0  # 分岐角SD (°)
    lambda_: float = 10.0  # F-actin 持続長 (µm)
    G_actin_length: float = 0.0027
    theta_mean: float = 70.0
    theta_sd: float = 10.0
    KBT: float = 4.41e-3  # 熱エネルギー (pJ·µm)
    A_volume: float = 1e-16  # 体積エネルギー係数 (pJ/µm⁵)
    A_surface: float = 1e-13  # 表面エネルギー係数 (pJ/µm⁴)
    A_bending: float = 3.2e-10  # 曲率エネルギー係数 (pJ/µm²)
    V0: float = 15.6  # 定常体積 (µm³)
    S0: float = 49.6  # 定常表面積 (µm²)

    class Config:
        # 変更不可にしたい場合は uncomment
        # frozen = True
        arbitrary_types_allowed = False
