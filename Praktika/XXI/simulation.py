import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
})

# =====================================================================
# PARSOVÁNÍ DAT
# =====================================================================
def parse_data(filename):
    """
    Načte INI-like strukturu s tabulkami. 
    Umí zpracovat i tabulky bez textové hlavičky (jen pole hodnot).
    """
    data = {}
    current_section = None
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                data[current_section] = {}
                continue
            if current_section is None:
                continue
            
            if "Tabulka" in current_section:
                parts = [p.strip() for p in line.split(',')]
                if any(c.isalpha() for c in parts[0]):
                    for p in parts:
                        data[current_section][p] = []
                    data[current_section]['_keys'] = parts
                else:
                    if '_keys' in data[current_section]:
                        keys = data[current_section]['_keys']
                        for i, val in enumerate(parts):
                            data[current_section][keys[i]].append(float(val))
                    else:
                        if 'values' not in data[current_section]:
                            data[current_section]['values'] = []
                        data[current_section]['values'].extend([float(x) for x in parts])
            else:
                if '=' in line:
                    k, v = [x.strip() for x in line.split('=', 1)]
                    try:
                        data[current_section][k] = float(v)
                    except ValueError:
                        data[current_section][k] = v
    return data


# =====================================================================
# FYZIKÁLNÍ MODELY PERIODY
# =====================================================================
def T_ideal(L, g):
    """
    Model 1 — matematické kyvadlo, malé výchylky.
    T = 2π√(L/g)
    Zanedbává: hmotnost závěsu, rozměr kuličky, amplitudu.
    """
    return 2 * np.pi * np.sqrt(L / g)


def T_fyzicke_mala_vychylka(I, M, d, g):
    """
    Model 2 — fyzické kyvadlo, stále malé výchylky (sin α ≈ α).
    T = 2π√(I / (M·g·d))
    Zahrnuje: reálný moment setrvačnosti, polohu těžiště.
    Zanedbává: konečnou amplitudu.
    """
    return 2 * np.pi * np.sqrt(I / (M * g * d))


def T_fyzicke_konecna_amplituda(I, M, d, g, theta0_rad):
    """
    Model 3 — fyzické kyvadlo, konečná amplituda, analytická korekce.
    T = 2π√(I/Mgd) · (1 + (1/4)·sin²(α/2))
    Zahrnuje: reálný I, d, konečnou amplitudu.
    Zanedbává: útlum.
    """
    T0 = T_fyzicke_mala_vychylka(I, M, d, g)
    korekce = 1.0 + 0.25 * np.sin(theta0_rad / 2)**2
    return T0 * korekce


def T_numericky(I, M, d, g, theta0_rad):
    """
    Model 4 — numerické řešení plné pohybové rovnice fyzického kyvadla
    bez útlumu (konzervativní systém):
        I·θ'' + M·g·d·sin(θ) = 0
    """
    def eq(t, y):
        theta, omega = y
        return [omega, -(M * g * d / I) * np.sin(theta)]

    def zero_crossing(t, y):
        return y[0]
    zero_crossing.terminal  = True   
    zero_crossing.direction = -1     

    T_approx = T_fyzicke_konecna_amplituda(I, M, d, g, theta0_rad)
    t_max    = T_approx * 0.6        

    sol = solve_ivp(
        eq, (0, t_max), [theta0_rad, 0.0],
        events=zero_crossing,
        rtol=1e-10, atol=1e-12,
        max_step=t_max / 500
    )

    if sol.t_events[0].size > 0:
        return 4.0 * sol.t_events[0][0]
    else:
        return T_approx


# =====================================================================
# VÝPOČET FYZIKÁLNÍCH PARAMETRŮ PRO JEDNU MC REALIZACI
# =====================================================================
def fyzikalni_parametry(m_k, m_t, D_k, h_h, l_z):
    L_M   = l_z + h_h + D_k / 2
    I_koule = (2/5) * m_k * (D_k / 2)**2
    I_tyc   = (1/12) * m_t * l_z**2
    I_celk  = I_tyc + m_t * (l_z / 2)**2 + I_koule + m_k * L_M**2
    M_celk = m_t + m_k
    d_tez  = (m_t * (l_z / 2) + m_k * L_M) / M_celk

    return L_M, I_celk, M_celk, d_tez


# =====================================================================
# HLAVNÍ MONTE CARLO
# =====================================================================
def main():
    INPUT_FILE = "data.txt"
    print(f"Načítám data z {INPUT_FILE}...")
    try:
        data = parse_data(INPUT_FILE)
    except FileNotFoundError:
        print(f"CHYBA: Soubor {INPUT_FILE} nenalezen.")
        return

    # --- Nejistoty ---
    err_t   = data['Nejistoty_pristroju']['err_t_s']
    err_m   = data['Nejistoty_pristroju']['err_m_g'] / 1000
    err_l_m = data['Nejistoty_pristroju']['err_l_metr_mm'] / 1000
    err_l_p = data['Nejistoty_pristroju']['err_l_posuvka_mm'] / 1000

    # --- Střední hodnoty (SI) a odvození průměru D_k ---
    if 'Tabulka_1_Prumery_koule' in data:
        D_raw_mm = np.array(data['Tabulka_1_Prumery_koule']['values'])
        mu_Dk = np.mean(D_raw_mm) / 1000
        # Kombinovaná nejistota D_k (statistická typu A + přístrojová posuvky typu B)
        D_stat_err_mm = np.std(D_raw_mm, ddof=1) / np.sqrt(len(D_raw_mm))
        err_Dk = np.sqrt(D_stat_err_mm**2 + (err_l_p * 1000)**2) / 1000
    else:
        mu_Dk = data['Jednorazova_mereni']['D_k_mm'] / 1000
        err_Dk = err_l_p

    mu = {
        'm_k': data['Jednorazova_mereni']['m_k_g'] / 1000,
        'm_t': data['Jednorazova_mereni']['m_t_g'] / 1000,
        'h_h': data['Jednorazova_mereni']['h_h_mm'] / 1000,
        'l_z': data['Jednorazova_mereni']['l_z_mm'] / 1000,
    }

    # --- Naměřená perioda (Nyní z Tabulky 2) ---
    t10M_raw       = np.array(data['Tabulka_2_Matematicke_kyvadlo']['t_10M'])
    t10M_mean      = np.mean(t10M_raw)
    t10M_stat_err  = np.std(t10M_raw, ddof=1) / np.sqrt(len(t10M_raw))
    t10M_total_err = np.sqrt(t10M_stat_err**2 + err_t**2)

    T_exp     = t10M_mean      / 10   
    T_exp_err = t10M_total_err / 10   

    g_tab      = 9.811
    theta0_rad = np.radians(5.0)   

    # --- Monte Carlo ---
    N_iter = 5000
    np.random.seed(42)

    mc = {
        'm_k': np.random.normal(mu['m_k'], err_m, N_iter),
        'm_t': np.random.normal(mu['m_t'], err_m, N_iter),
        'D_k': np.random.normal(mu_Dk, err_Dk, N_iter), # Zde použito složené err_Dk
        'h_h': np.random.normal(mu['h_h'], err_l_p, N_iter),
        'l_z': np.random.normal(mu['l_z'], err_l_m, N_iter)
    }

    T1 = np.zeros(N_iter)   
    T2 = np.zeros(N_iter)   
    T3 = np.zeros(N_iter)   
    T4 = np.zeros(N_iter)   

    print(f"Spouštím MC simulaci ({N_iter} iterací, 4 modely)...")
    for i in range(N_iter):
        if i % (N_iter // 5) == 0 and i > 0:
            print(f"  {i/N_iter*100:.0f} % ...")

        L_M, I, M, d = fyzikalni_parametry(
            mc['m_k'][i], mc['m_t'][i], mc['D_k'][i],
            mc['h_h'][i], mc['l_z'][i]
        )

        T1[i] = T_ideal(L_M, g_tab)
        T2[i] = T_fyzicke_mala_vychylka(I, M, d, g_tab)
        T3[i] = T_fyzicke_konecna_amplituda(I, M, d, g_tab, theta0_rad)
        T4[i] = T_numericky(I, M, d, g_tab, theta0_rad)

    print("Simulace dokončena.\n")

    modely = [
        (T1, 'Model 1: Ideální ($T=2\\pi\\sqrt{L/g}$)',              'lightgray',   'dimgray'),
        (T2, 'Model 2: Fyzické kyvadlo, malé výchylky',              'steelblue',   'navy'),
        (T3, 'Model 3: + korekce amplitudy (analytická)',            'mediumseagreen','darkgreen'),
        (T4, 'Model 4: Numerické ODE (referenční)',                  'darkorange',  'saddlebrown'),
    ]

    print(f"{'Model':<45} {'střed [s]':>12} {'σ [ms]':>10} {'Δ vs. exp [ms]':>16}")
    print("-" * 87)
    for T_arr, nazev, _, _ in modely:
        delta = (np.mean(T_arr) - T_exp) * 1000
        print(f"{nazev:<45} {np.mean(T_arr):>12.6f} {np.std(T_arr)*1000:>10.4f} {delta:>+16.4f}")
    print(f"{'Experiment':.<45} {T_exp:>12.6f} {T_exp_err*1000:>10.4f}")


    # =====================================================================
    # GRAF — Expected vs. Observed (Vylepšená vizualizace)
    # =====================================================================
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2.5, 1]})
    ax_main, ax_res = axes

    # Definice barev pro lepší kontrast
    barvy = ['#BDC3C7', '#3498DB', '#2ECC71', '#C0392B']
    
    # Biny se počítají jen pro oblast simulovaných dat
    hist_min = min(T1.min(), T2.min(), T3.min(), T4.min())
    hist_max = max(T1.max(), T2.max(), T3.max(), T4.max())
    bins = np.linspace(hist_min, hist_max, 45)

    # Vykreslení M1, M2 a M3 jako poloprůhledné plochy
    for i in range(3):
        T_arr, nazev, _, _ = modely[i]
        ax_main.hist(T_arr, bins=bins, alpha=0.4, color=barvy[i], 
                     edgecolor=barvy[i], linewidth=2.5, histtype='stepfilled',
                     label=f'{nazev}\n  $\\mu={np.mean(T_arr):.4f}$ s, $\\sigma={np.std(T_arr)*1e3:.2f}$ ms')

    # Vykreslení M4 pouze jako tlustou přerušovanou čáru
    T_arr4, nazev4, _, _ = modely[3]
    ax_main.hist(T_arr4, bins=bins, alpha=1.0, color=barvy[3], 
                 linewidth=2.5, linestyle='--', histtype='step', zorder=4,
                 label=f'{nazev4}\n  $\\mu={np.mean(T_arr4):.4f}$ s, $\\sigma={np.std(T_arr4)*1e3:.2f}$ ms')

    # Naměřená hodnota a experimentální chyba
    ax_main.axvline(T_exp, color='black', linewidth=3, zorder=6,
                    label=f'Experiment (naměřená data):\n  $T = {T_exp:.5f}$ s')
    ax_main.axvspan(T_exp - T_exp_err, T_exp + T_exp_err,
                    color='black', alpha=0.15, zorder=5,
                    label=f'Nejistota měření ($\\pm{T_exp_err*1e3:.2f}$ ms)')

    # Anotace se šipkou na systematickou chybu
    rozdil_M4_exp = np.mean(T_arr4) - T_exp
    ax_main.annotate(f'Neznámá systematická chyba\nModely jsou posunuty o $\\approx {rozdil_M4_exp*1000:.1f}$ ms',
                     xy=(T_exp, N_iter * 0.05),
                     xytext=(T_exp + 0.003, N_iter * 0.08),
                     arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
                     fontsize=11, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.9), zorder=10)

    ax_main.set_xlabel(r'Perioda $T$ / s', fontsize=13)
    ax_main.set_ylabel('Počet MC realizací', fontsize=13)
    ax_main.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax_main.grid(axis='y', linestyle='--', alpha=0.6)
    ax_main.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.4f}".replace('.', ',')))
    
    # Rozšíření osy X doleva
    ax_main.set_xlim(left=T_exp - 0.002, right=hist_max + 0.001)

    # --- Spodní panel: Sloupcový graf odchylek ---
    stredni   = [np.mean(m[0]) for m in modely]
    nazvy_kr  = ['M1\nIdeální', 'M2\nFyzické', 'M3\nAnalytické', 'M4\nNumerické']
    odchylky_ms = [(s - T_exp) * 1000 for s in stredni]

    bars = ax_res.bar(nazvy_kr, odchylky_ms, color=barvy,
                      edgecolor='black', linewidth=1.2, alpha=0.85)
    
    ax_res.axhline(0, color='black', linewidth=1.5, linestyle='-')
    ax_res.axhspan(-T_exp_err*1000, T_exp_err*1000,
                   color='black', alpha=0.15, label='Nejistota exp. měření')
    
    ax_res.set_ylabel(r'$\bar{T}_\mathrm{model} - T_\mathrm{exp}$ / ms', fontsize=13)
    ax_res.legend(loc='lower right', fontsize=10)
    ax_res.grid(axis='y', linestyle=':', alpha=0.6)

    max_odchylka = max(odchylky_ms)
    ax_res.set_ylim(bottom=-2, top=max_odchylka + 3)
    for bar, val in zip(bars, odchylky_ms):
        ax_res.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                    f'{val:+.2f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    PLOT_FILE = 'porovnani_modelu_experiment_vylepseno.pdf'
    fig.savefig(PLOT_FILE, dpi=300)
    print(f"Graf uložen jako '{PLOT_FILE}'.")


if __name__ == "__main__":
    main()