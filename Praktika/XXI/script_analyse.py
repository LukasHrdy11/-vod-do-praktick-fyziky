import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from uncertainties import ufloat
from uncertainties.umath import *

# =====================================================================
# KONFIGURACE SKRIPTU
# =====================================================================
INPUT_FILE = "data.txt"
OUTPUT_FILE = "vysledky_analyzy.txt"
PLOT_FILE = "graf_reverzni_kyvadlo.pdf"

pi = np.pi

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
})

# =====================================================================
# POMOCNÉ FUNKCE
# =====================================================================
def format_result(u_val, unit=""):
    val = u_val.n
    err = u_val.s

    if err == 0 or np.isnan(err) or not np.isfinite(err):
        if val != 0:
            order = int(np.floor(np.log10(abs(val))))
            decimals = max(0, 4 - 1 - order)
            res = f"{val:.{decimals}f} [nejistota neurčena]"
        else:
            res = "0 [nejistota neurčena]"
        res = res.replace('.', ',')
        return f"({res}) {unit}".strip() if unit else res

    order = int(np.floor(np.log10(err)))
    rounded_err = round(err, -order)

    if rounded_err >= 10**(order + 1):
        order += 1
        rounded_err = round(err, -order)

    rounded_val = round(val, -order)

    if order < 0:
        decimals = -order
        res = f"{rounded_val:.{decimals}f} ± {rounded_err:.{decimals}f}"
    else:
        res = f"{int(rounded_val)} ± {int(rounded_err)}"

    res = res.replace('.', ',')
    return f"({res}) {unit}" if unit else res


def parse_data(filename):
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
                    keys = data[current_section]['_keys']
                    for i, val in enumerate(parts):
                        data[current_section][keys[i]].append(float(val))
            else:
                if '=' in line:
                    k, v = [x.strip() for x in line.split('=', 1)]
                    try:
                        data[current_section][k] = float(v)
                    except ValueError:
                        data[current_section][k] = v
    return data


# =====================================================================
# FITOVÁNÍ — effective variance (metoda z kurzu)
# =====================================================================
def fit_line_effective_variance(x, y, ex, ey):
    """
    Lineární fit y = q + k*x metodou effective variance.
    Minimalizuje chi^2 = sum[(y - q - k*x)^2 / (ey^2 + (k*ex)^2)]
    pomocí scipy.optimize.leastsq.

    Vrací:
        beta  : [q, k]  (stejné pořadí jako scipy.odr: intercept, slope)
        cov   : 2x2 kovariační matice parametrů [q, k]
    """
    def residuals(theta, x, y, ex, ey):
        q, k = theta
        return (y - q - k * x) / np.sqrt(ey**2 + (k * ex)**2)

    # Počáteční odhad — obyčejná přímka přes první a poslední bod
    k0 = (y[-1] - y[0]) / (x[-1] - x[0])
    q0 = y[0] - k0 * x[0]

    res = leastsq(residuals, [q0, k0], args=(x, y, ex, ey), full_output=True)

    beta = res[0]           # [q, k]
    cov  = res[1]           # kovariační matice 2x2

    if cov is None:
        raise RuntimeError("leastsq nevrátil kovariační matici — fit nekonvergoval.")

    return beta, cov


# =====================================================================
# PRŮSEČÍK S ANALYTICKOU PROPAGACÍ CHYB
# =====================================================================
def intersect_with_uncertainty(beta1, cov1, beta2, cov2):
    """
    Průsečík dvou přímek y = q + k*x.
    beta = [q, k], cov = 2x2 kovariační matice.
    Předpokládá nezávislost obou fitů.
    """
    q1, k1 = beta1
    q2, k2 = beta2
    dk = k1 - k2
    dq = q2 - q1

    if abs(dk) < 1e-15:
        raise ValueError("Přímky jsou rovnoběžné, průsečík neexistuje.")

    x0 = dq / dk
    y0 = k1 * x0 + q1

    # Jacobiány x0 = (q2-q1)/(k1-k2) vůči [q1,k1] a [q2,k2]
    dx0_dq1 = -1.0 / dk
    dx0_dk1 = -dq  / dk**2
    dx0_dq2 =  1.0 / dk
    dx0_dk2 =  dq  / dk**2

    # Jacobiány y0 = k1*x0 + q1
    dy0_dq1 = k1 * dx0_dq1 + 1.0
    dy0_dk1 = x0 + k1 * dx0_dk1
    dy0_dq2 = k1 * dx0_dq2
    dy0_dk2 = k1 * dx0_dk2

    Jx1 = np.array([dx0_dq1, dx0_dk1])
    Jx2 = np.array([dx0_dq2, dx0_dk2])
    var_x0 = float(Jx1 @ cov1 @ Jx1 + Jx2 @ cov2 @ Jx2)

    Jy1 = np.array([dy0_dq1, dy0_dk1])
    Jy2 = np.array([dy0_dq2, dy0_dk2])
    var_y0 = float(Jy1 @ cov1 @ Jy1 + Jy2 @ cov2 @ Jy2)

    std_x0 = np.sqrt(var_x0) if var_x0 > 0 else float('nan')
    std_y0 = np.sqrt(var_y0) if var_y0 > 0 else float('nan')

    return ufloat(x0, std_x0), ufloat(y0, std_y0)


def check_intersection_quality(x0, x_data, label="průsečík"):
    x_min, x_max = np.min(x_data), np.max(x_data)
    x_range = x_max - x_min

    if x0 < x_min or x0 > x_max:
        vzdalenost = min(abs(x0 - x_min), abs(x0 - x_max))
        nasobek = vzdalenost / x_range
        print(f"\n{'='*60}")
        print(f"  VAROVÁNÍ: {label} leží MIMO rozsah dat!")
        print(f"  Rozsah dat:    [{x_min:.2f}, {x_max:.2f}] mm")
        print(f"  Průsečík:       {x0:.2f} mm")
        print(f"  Extrapolace o:  {vzdalenost:.1f} mm ({nasobek:.1f}× šířka rozsahu dat)")
        print(f"  → Přidejte body blíže k {x0:.1f} mm.")
        print(f"{'='*60}\n")
    else:
        print(f"  OK: {label} leží uvnitř rozsahu dat ({x0:.2f} mm).")


# =====================================================================
# HLAVNÍ VÝPOČETNÍ BLOK
# =====================================================================
def analyze():
    print(f"Načítám data ze souboru {INPUT_FILE}...")
    try:
        data = parse_data(INPUT_FILE)
    except FileNotFoundError:
        print(f"CHYBA: Soubor {INPUT_FILE} nebyl nalezen.")
        return

    err_t   = data['Nejistoty_pristroju']['err_t_s']
    err_m   = data['Nejistoty_pristroju']['err_m_g'] / 1000
    err_l_m = data['Nejistoty_pristroju']['err_l_metr_mm'] / 1000
    err_l_p = data['Nejistoty_pristroju']['err_l_posuvka_mm'] / 1000

    m_k = ufloat(data['Jednorazova_mereni']['m_k_g'], err_m * 1000) / 1000
    m_t = ufloat(data['Jednorazova_mereni']['m_t_g'], err_m * 1000) / 1000
    D_k = ufloat(data['Jednorazova_mereni']['D_k_mm'], err_l_p * 1000) / 1000
    h_h = ufloat(data['Jednorazova_mereni']['h_h_mm'], err_l_p * 1000) / 1000
    l_z = ufloat(data['Jednorazova_mereni']['l_z_mm'], err_l_m * 1000) / 1000
    L_r = ufloat(data['Jednorazova_mereni']['L_r_mm'], err_l_m * 1000) / 1000

    # =================================================================
    # 1. MATEMATICKÉ KYVADLO
    # =================================================================
    t10M_raw = np.array(data['Tabulka_1_Matematicke_kyvadlo']['t_10M'])

    t10M_mean      = np.mean(t10M_raw)
    t10M_stat_err  = np.std(t10M_raw, ddof=1) / np.sqrt(len(t10M_raw))
    t10M_total_err = np.sqrt(t10M_stat_err**2 + err_t**2)

    T_M = ufloat(t10M_mean, t10M_total_err) / 10
    L_M = l_z + h_h + (D_k / 2)
    g_M = (4 * pi**2 * L_M) / (T_M**2)

    # =================================================================
    # 2. CHYBA IDEALIZACE
    # =================================================================
    I_ideal = m_k * L_M**2
    I_koule = (2/5) * m_k * (D_k / 2)**2
    I_tyc   = (1/12) * m_t * l_z**2
    I_celk  = I_tyc + m_t * (l_z / 2)**2 + I_koule + m_k * L_M**2

    diff_I     = I_celk - I_ideal
    rel_diff_I = (diff_I / I_ideal) * 100

    d_tez      = (m_t * (l_z / 2) + m_k * L_M) / (m_t + m_k)
    diff_L     = L_M - d_tez
    rel_diff_L = (diff_L / L_M) * 100

    # =================================================================
    # 3. REVERZNÍ KYVADLO — effective variance fit
    # =================================================================
    lp_raw       = np.array(data['Tabulka_2_Reverzni_kyvadlo']['l_p'])
    t10_up_raw   = np.array(data['Tabulka_2_Reverzni_kyvadlo']['t_10_nahore'])
    t10_down_raw = np.array(data['Tabulka_2_Reverzni_kyvadlo']['t_10_dole'])

    T_up   = t10_up_raw   / 10
    T_down = t10_down_raw / 10

    err_lp_mm = err_l_p * 1000
    err_T_s   = err_t   / 10

    sx = np.full_like(lp_raw, err_lp_mm)
    sy = np.full_like(T_up,   err_T_s)

    beta_up,   cov_up   = fit_line_effective_variance(lp_raw, T_up,   sx, sy)
    beta_down, cov_down = fit_line_effective_variance(lp_raw, T_down, sx, sy)

    lp_intersect, T_r = intersect_with_uncertainty(
        beta_up, cov_up, beta_down, cov_down
    )

    check_intersection_quality(lp_intersect.n, lp_raw, label="Průsečík l_p0")

    g_r = (4 * pi**2 * L_r) / (T_r**2)

    # =================================================================
    # 4. POROVNÁNÍ
    # =================================================================
    g_tab = ufloat(9.811, 0.001)

    diff_g_M  = g_M - g_tab
    rel_err_M = abs(diff_g_M.n) / g_tab.n * 100
    z_score_M = abs(diff_g_M.n) / diff_g_M.s
    rel_unc_M = (g_M.s / g_M.n) * 100

    diff_g_r  = g_r - g_tab
    rel_err_r = abs(diff_g_r.n) / g_tab.n * 100
    z_score_r = abs(diff_g_r.n) / diff_g_r.s if g_r.s > 0 else float('nan')
    rel_unc_r = (g_r.s / g_r.n) * 100

    # =================================================================
    # GRAF
    # =================================================================
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.errorbar(lp_raw, T_up,   xerr=sx, yerr=sy, fmt='ko',
                markerfacecolor='none', markersize=8, capsize=3,
                label='čočka nahoře', zorder=3)
    ax.errorbar(lp_raw, T_down, xerr=sx, yerr=sy, fmt='ks',
                markersize=8, capsize=3, label='čočka dole', zorder=3)

    x_min_plot = min(np.min(lp_raw) - 2, lp_intersect.n - 2)
    x_max_plot = max(np.max(lp_raw) + 2, lp_intersect.n + 2)
    x_fit = np.linspace(x_min_plot, x_max_plot, 300)

    q_up,  k_up  = beta_up
    q_down, k_down = beta_down

    ax.plot(x_fit, q_up   + k_up   * x_fit, 'k--', alpha=0.5)
    ax.plot(x_fit, q_down + k_down * x_fit, 'k--', alpha=0.5)

    ax.axvline(x=lp_intersect.n, color='gray', linestyle=':', linewidth=1)
    ax.axhline(y=T_r.n,          color='gray', linestyle=':', linewidth=1)

    ax.plot(lp_intersect.n, T_r.n, 'k*', markersize=12,
            label=f'průsečík ({lp_intersect.n:.2f} mm, {T_r.n:.4f} s)', zorder=5)

    x_data_min, x_data_max = np.min(lp_raw), np.max(lp_raw)
    if lp_intersect.n > x_data_max:
        ax.axvspan(x_data_max, x_max_plot, alpha=0.08, color='red', label='extrapolace')
    elif lp_intersect.n < x_data_min:
        ax.axvspan(x_min_plot, x_data_min, alpha=0.08, color='red', label='extrapolace')

    ax.set_xlabel(r'$l_p$ (mm)')
    ax.set_ylabel(r'$T$ (s)')
    ax.legend(loc='best')
    ax.grid(True, linestyle=':', alpha=0.7)

    fig.savefig(PLOT_FILE, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Graf uložen jako {PLOT_FILE}")

    # =================================================================
    # VÝSTUP
    # =================================================================
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(" VÝSLEDKY ANALÝZY EXPERIMENTU: MĚŘENÍ TÍHOVÉHO ZRYCHLENÍ\n")
        f.write("="*60 + "\n\n")

        f.write("--- 1. MATEMATICKÉ KYVADLO ---\n")
        f.write(f"Zprůměrovaná perioda:       T_M = {format_result(T_M, 's')}\n")
        f.write(f"Délka idealiz. kyvadla:     L_M = {format_result(L_M, 'm')}\n")
        f.write(f"Vypočtené tíhové zrychlení: g_M = {format_result(g_M, 'm/s^2')}\n\n")

        f.write("--- 2. CHYBOVÁ ANALÝZA IDEALIZACE ---\n")
        f.write("a) Srovnání momentů setrvačnosti:\n")
        f.write(f"   Idealizovaný model: I_ideal = {format_result(I_ideal, 'kg*m^2')}\n")
        f.write(f"   Reálné kyvadlo:     I_celk  = {format_result(I_celk, 'kg*m^2')}\n")
        f.write(f"   Rozdíl idealizace:  delta_I = {format_result(diff_I, 'kg*m^2')} ({format_result(rel_diff_I, '%')})\n\n")
        f.write("b) Srovnání délek:\n")
        f.write(f"   Délka matematického k.: L_M     = {format_result(L_M, 'm')}\n")
        f.write(f"   Skutečná poloha těžiště: d_tez  = {format_result(d_tez, 'm')}\n")
        f.write(f"   Posun těžiště:          delta_L = {format_result(diff_L, 'm')} ({format_result(rel_diff_L, '%')})\n\n")

        f.write("--- 3. REVERZNÍ KYVADLO ---\n")

        x_min_d, x_max_d = np.min(lp_raw), np.max(lp_raw)
        if not (x_min_d <= lp_intersect.n <= x_max_d):
            f.write("  !! VAROVÁNÍ: Průsečík leží MIMO rozsah naměřených dat.\n")
            f.write(f"  !! Rozsah dat: [{x_min_d:.2f}, {x_max_d:.2f}] mm, průsečík: {lp_intersect.n:.2f} mm\n")
            f.write("  !! Nejistota průsečíku závisí na předpokladu linearity mimo data.\n")
            f.write("  !! Přidejte body blíže k průsečíku pro spolehlivý výsledek.\n\n")

        f.write(f"Společná poloha čočky:  l_p0 = {format_result(lp_intersect, 'mm')}\n")
        f.write(f"Interpolovaná perioda:  T_r  = {format_result(T_r, 's')}\n")
        f.write(f"Redukovaná délka:       L_r  = {format_result(L_r, 'm')}\n")
        f.write(f"Vypočtené tíhové zrychlení: g_r  = {format_result(g_r, 'm/s^2')}\n\n")

        f.write("--- 4. POROVNÁNÍ S TABULKOVOU HODNOTOU A DISKUZE ---\n")
        f.write(f"Tabulková hodnota: g_tab = {format_result(g_tab, 'm/s^2')}\n\n")

        f.write("a) MATEMATICKÉ KYVADLO:\n")
        f.write(f"   Absolutní odchylka: Delta g_M = {format_result(diff_g_M, 'm/s^2')}\n")
        f.write(f"   Relativní odchylka:           = {rel_err_M:.3f} %\n")
        f.write(f"   Shoda v rámci nejistot:   z_M = {z_score_M:.1f} sigma\n")
        f.write(f"   Relativní přesnost měření:    = {rel_unc_M:.2f} %\n\n")

        f.write("b) REVERZNÍ KYVADLO:\n")
        f.write(f"   Absolutní odchylka: Delta g_r = {format_result(diff_g_r, 'm/s^2')}\n")
        f.write(f"   Relativní odchylka:           = {rel_err_r:.3f} %\n")
        if np.isfinite(z_score_r):
            f.write(f"   Shoda v rámci nejistot:   z_r = {z_score_r:.1f} sigma\n")
        else:
            f.write(f"   Shoda v rámci nejistot:   z_r = [nelze určit]\n")
        f.write(f"   Relativní přesnost měření:    = {rel_unc_r:.2f} %\n")

    print(f"Výsledky byly úspěšně zapsány do {OUTPUT_FILE}")


if __name__ == "__main__":
    analyze()