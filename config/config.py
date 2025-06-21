from pydantic import BaseModel



class InputModel(BaseModel):
    # シミュレーション領域と格子の設定
    Nx, Ny = 70, 70  # 格子サイズ
    dx = dy = 1.0  # 格子間隔 (μm)
    Dt = 0.0009  # 時間ステップ（0.01だとGアクチンおかしい）
    D_Factin = 0.01  # 拡散係数 (μm^2/s)
    steps = 10000  # シミュレーションステップ数
    num_points = 50  # 頂点の数
    total_concentration = 10000  # 細胞内のF-actinとG-actinの濃度総和
    F_actin_length = 1.0  # 仮設定
    num_filaments = 300  # 設定

    # 論文のパラメータM
    # 初期濃度の設定。
    C_Factin_init = 0
    C_Gactin_init = 18.0  # Gアクチンの濃度(μM)
    C_Arp_init = 0.3  # Arp2/3の初期濃度(μM)
    C_CP_init = 2.0  # cpの初期濃度（μM）
    C_cofilin_init = 0.5  # コフィリンの初期濃度(μM)


    # 拡散係数
    D_Gactin = 30  # アクチン単量体の拡散係数m2/s
    D_Arp = 30  # Arp2/3の拡散係数m2/s
    D_CP = 0.1  # キャッピングタンパク質の拡散係数m2/s
    D_cofilin = 0.1  # コフィリンの拡散係数m2/s

    # 反応速度定数
    K_polB = 120  # Barbed endでのポリメライゼーション速度（μM⁻¹ s⁻¹）(120)
    K_polP = 3  # Pointed endでのポリメライゼーション速度（μM⁻¹ s⁻¹）
    K_depolB = 1.4  # barded end のデポリメライゼーション速度（s⁻¹）
    K_depolP = 21  # pointed end のデポリメライゼーション速度（s⁻¹）(21)
    K_bindarp = 3.4  # Arp2/3のside-binding rate(3.4)
    K_bindcp = 4  # キャッピングタンパク質の結合速度(3)
    K_unbindcp = 0.04  # キャッピングタンパク質の解離速度
    K_bindcof = 0.0085  # コフィリンの結合速度(0.0085)
    K_unbindcof = 0.005  # コフィリンの解離速度
    K_sev = 0.012  # コフィリンによる切断速度(0.012)
    K_actarp = 4  # Arp2/3の活性化速度(4)
    K_inactarp = 1  # Arp2/3の不活性化速度
    K_actcp = 0.04  # キャッピングタンパク質の活性率(0.04)
    K_inactcp = 4.0**5  # キャッピングタンパク質の不活性率

    H = 0.2  # 小胞体の高さ(μm)
    delta_Gactin = 2.7  # アクチン単量体の長さ
    shita_Mean = 70  # 分岐角の平均
    shita_SD = 10  # 分岐角の標準偏差
    lambda_ = 10  # Fアクチンの持続長(μm)
    G_actin_length = 0.0027  # アクチンモノマーの長さ(μm)
    theta_mean = 70  # 分岐角度の平均（°）
    theta_sd = 10  # 分岐角度の標準偏差（°）
    KBT = 4.41 * 10**-3  # 熱エネルギーの単位(pJ μｍ)
    A_volume = 10**-16  # 体積エネルギー係数(pJ/μm^5)
    A_surface = 10**-13  # 表面エネルギー係数(pJ/μm^4)
    A_bending = 3.2 * 10**-10  # 膜の曲率エネルギー係数(pJ/μm^2)
    V0 = 15.6  # 定常状態の体積μm3
    S0 = 49.6  # 定常状態の表面積μm2
