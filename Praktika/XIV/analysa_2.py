import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit
import uncertainties
from uncertainties import unumpy as unp
from uncertainties import ufloat
import uncertainties.umath as umath

# ==============================================================================
# KONFIGURACE SKRIPTU
# ==============================================================================
INPUT_FILE = 'data.txt'
OUTPUT_FILE = 'vysledky_analyzy.txt'
GRAPH_FILE = 'graf_zavislosti.pdf'

# Tabulkové hodnoty
TAB_T = np.array([10, 15, 20, 25, 30, 40, 50, 60, 70, 80])
TAB_SIGMA = np.array([74.2, 73.5, 72.75, 72.0, 71.2, 69.6, 67.9, 66.2, 64.4, 62.6])

# ==============================================================================
# POMOCNÉ FUNKCE
# ==============================================================================
def hustota_vody_smow(t):
    a0 = 999.842594
    a1 = 6.793952e-2
    a2 = -9.095290e-3
    a3 = 1.001685e-4
    a4 = -1.120083e-6
    a5 = 6.536332e-9
    return a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5

def format_cz_pm(u_val):
    if isinstance(u_val, (float, np.float64)):
        return f"{u_val:.2f}".replace('.', ',')
    text = f"{u_val:.1u}"
    return text.replace('+/-', ' ± ').replace('.', ',')

def czech_comma_formatter(x, pos):
    return f"{x}".replace('.', ',')

def model_kvadraticky(t, A, B, C):
    return A*t**2 + B*t + C

def model_linearni(t, B, C):
    return B*t + C

# ==============================================================================
# HLAVNÍ ČÁST SKRIPTU
# ==============================================================================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Chyba: Soubor {INPUT_FILE} nebyl nalezen.")
        # Pro demonstraci vygenerujeme dummy data, pokud soubor chybí
        with open(INPUT_FILE, 'w') as f:
            f.write("[Laboratorni_podminky]\nt_lab = 22.0\n[Konstanty]\nr_0 = 0.26\nalpha = 30.0\ng = 9.811\n[Nejistoty_pristroju]\nu_t = 0.5\nu_dmax = 0.5\nu_r0 = 0.01\nu_alpha = 0.5\n[Mereni_povrchoveho_napeti]\n20.0, 15.2\n25.0, 14.9\n30.0, 14.5\n")

    konstanty = {}
    nejistoty = {}
    t_data, d_max_data = [], []
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        sekce = None
        for line in f:
            line = line.split('#')[0].strip()
            if not line: continue
            if line.startswith('[') and line.endswith(']'):
                sekce = line[1:-1]
                continue
                
            if sekce in ['Laboratorni_podminky', 'Konstanty']:
                if '=' in line:
                    k, v = line.split('=')
                    konstanty[k.strip()] = float(v.strip())
            elif sekce == 'Nejistoty_pristroju':
                if '=' in line:
                    k, v = line.split('=')
                    nejistoty[k.strip()] = float(v.strip())
            elif sekce == 'Mereni_povrchoveho_napeti':
                if ',' in line:
                    try:
                        vals = line.split(',')
                        t_data.append(float(vals[0].strip()))
                        d_max_data.append(float(vals[1].strip()))
                    except ValueError:
                        pass 

    u_t = nejistoty.get('u_t', 0.5)
    u_dmax = nejistoty.get('u_dmax', 0.5)
    r0_mm = ufloat(konstanty.get('r_0', 0.26), nejistoty.get('u_r0', 0.01))
    r0_m = r0_mm / 1000.0  
    alpha_deg = ufloat(konstanty.get('alpha', 30.0), nejistoty.get('u_alpha', 0.5))
    alpha_rad = alpha_deg * (math.pi / 180.0)
    g = konstanty.get('g', 9.811)

    t_u = unp.uarray(t_data, np.full(len(t_data), u_t))
    d_max_m_u = unp.uarray(d_max_data, np.full(len(d_max_data), u_dmax)) / 1000.0 

    t_lab_aktualni = konstanty.get('t_lab', 22.0)
    rho_vody = hustota_vody_smow(t_lab_aktualni)

    sin_alpha = umath.sin(alpha_rad)
    dp_max_u = d_max_m_u * rho_vody * g * sin_alpha
    sigma_u = (r0_m * dp_max_u) / 2.0
    sigma_scaled_u = sigma_u * 1000.0

    # Výpočet vlivu Cantorovy korekce (pro diskuzi)
    cantor_diff = (r0_m.n**2) * rho_vody * g / 3.0 * 1000.0

    x_val = unp.nominal_values(t_u)
    x_err = unp.std_devs(t_u)
    y_val = unp.nominal_values(sigma_scaled_u)
    y_err = unp.std_devs(sigma_scaled_u)

    # 4. Regrese
    # ==========================
    # A) EXPERIMENTÁLNÍ DATA
    # ==========================
    # Kvadratická exp
    popt_k, pcov_k = curve_fit(model_kvadraticky, x_val, y_val, sigma=y_err, absolute_sigma=True)
    perr_k = np.sqrt(np.diag(pcov_k))
    A_u = ufloat(popt_k[0], perr_k[0])
    B_k_u = ufloat(popt_k[1], perr_k[1])
    C_k_u = ufloat(popt_k[2], perr_k[2])

    # Lineární exp
    popt_l, pcov_l = curve_fit(model_linearni, x_val, y_val, sigma=y_err, absolute_sigma=True)
    perr_l = np.sqrt(np.diag(pcov_l))
    B_l_u = ufloat(popt_l[0], perr_l[0])
    C_l_u = ufloat(popt_l[1], perr_l[1])

    # ==========================
    # B) TABULKOVÁ DATA
    # ==========================
    # Kvadratická tab
    popt_tab_k, pcov_tab_k = curve_fit(model_kvadraticky, TAB_T, TAB_SIGMA)
    perr_tab_k = np.sqrt(np.diag(pcov_tab_k))
    A_tab_u = ufloat(popt_tab_k[0], perr_tab_k[0])
    B_tab_k_u = ufloat(popt_tab_k[1], perr_tab_k[1])
    C_tab_k_u = ufloat(popt_tab_k[2], perr_tab_k[2])

    # Lineární tab
    popt_tab_l, pcov_tab_l = curve_fit(model_linearni, TAB_T, TAB_SIGMA)
    perr_tab_l = np.sqrt(np.diag(pcov_tab_l))
    B_tab_l_u = ufloat(popt_tab_l[0], perr_tab_l[0])
    C_tab_l_u = ufloat(popt_tab_l[1], perr_tab_l[1])

    # ==========================
    # SROVNÁNÍ: EXP vs TAB
    # ==========================
    # Tabulkové hodnoty vyhodnocené při teplotách experimentu (pomocí lineárního fitu tabulek)
    sigma_tab_fit_at_x = model_linearni(x_val, *popt_tab_l)
    
    # Rozdíl: Δσ = exp - tab
    rozdil_sigma = y_val - sigma_tab_fit_at_x
    # Počet sigma: Z-score = |Δσ| / u(σ_exp)
    z_score = np.abs(rozdil_sigma) / y_err

    # 5. Generování grafu s rezidui
    fig, (ax_main, ax_res) = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Hlavní graf
    ax_main.plot(TAB_T, TAB_SIGMA, 's', color='orange', markersize=6, label='Tabulkové hodnoty', zorder=4)
    ax_main.errorbar(x_val, y_val, xerr=x_err, yerr=y_err, fmt='bo', capsize=3, label='Naměřená data', zorder=5)

    x_smooth = np.linspace(min(min(x_val), min(TAB_T))-2, max(max(x_val), max(TAB_T))+2, 200)
    
    # Fit experimentálních dat
    ax_main.plot(x_smooth, model_linearni(x_smooth, *popt_l), 'r-', label='Lineární fit (exp)')
    ax_main.plot(x_smooth, model_kvadraticky(x_smooth, *popt_k), 'g--', alpha=0.7, label='Kvadratický fit (exp)')
    
    # Fit tabulkových dat
    ax_main.plot(x_smooth, model_linearni(x_smooth, *popt_tab_l), color='darkorange', linestyle=':', linewidth=2, label='Lineární fit (tab)')

    ax_main.set_ylabel(r'$\sigma$ / $10^{-3}$ N$\cdot$m$^{-1}$', fontsize=12)
    ax_main.yaxis.set_major_formatter(FuncFormatter(czech_comma_formatter))
    ax_main.grid(True, linestyle='--', alpha=0.6)
    ax_main.legend(loc='best', fontsize=10)

    # Graf reziduí (pro lineární model experimentálních dat)
    rezidua = y_val - model_linearni(x_val, *popt_l)
    ax_res.axhline(0, color='black', linewidth=1)
    ax_res.errorbar(x_val, rezidua, yerr=y_err, fmt='ro', capsize=3)
    
    ax_res.set_xlabel('$t$ / °C', fontsize=12)
    ax_res.set_ylabel('Rezidua (exp)', fontsize=12)
    ax_res.xaxis.set_major_formatter(FuncFormatter(czech_comma_formatter))
    ax_res.yaxis.set_major_formatter(FuncFormatter(czech_comma_formatter))
    ax_res.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(GRAPH_FILE, dpi=300)

    # 6. Export výsledků
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("======================================================================\n")
        f.write(" VÝSLEDKY ANALÝZY MĚŘENÍ POVRCHOVÉHO NAPĚTÍ\n")
        f.write("======================================================================\n\n")
        f.write(f"Hustota vody (t = {t_lab_aktualni} °C): {rho_vody:.3f} kg/m^3\n")
        f.write(f"Poloměr r0: {format_cz_pm(r0_mm)} mm\n")
        f.write(f"Cantorova korekce (odhad vlivu): ~ {cantor_diff:.4f} * 10^-3 N/m\n\n")
        
        f.write("--- REGRESE EXPERIMENTÁLNÍCH DAT ---\n")
        f.write("KVADRATICKÁ REGRESE (sigma = A*t^2 + B*t + C):\n")
        f.write(f"A = ({format_cz_pm(A_u)}) 10^-3 N/(m·°C^2)\n")
        f.write(f"B = ({format_cz_pm(B_k_u)}) 10^-3 N/(m·°C)\n")
        f.write(f"C = ({format_cz_pm(C_k_u)}) 10^-3 N/m\n\n")

        f.write("LINEÁRNÍ REGRESE (sigma = B*t + C):\n")
        f.write(f"B = ({format_cz_pm(B_l_u)}) 10^-3 N/(m·°C)\n")
        f.write(f"C = ({format_cz_pm(C_l_u)}) 10^-3 N/m\n\n")

        f.write("--- REGRESE TABULKOVÝCH DAT ---\n")
        f.write("KVADRATICKÁ REGRESE (sigma_tab = A*t^2 + B*t + C):\n")
        f.write(f"A = ({format_cz_pm(A_tab_u)}) 10^-3 N/(m·°C^2)\n")
        f.write(f"B = ({format_cz_pm(B_tab_k_u)}) 10^-3 N/(m·°C)\n")
        f.write(f"C = ({format_cz_pm(C_tab_k_u)}) 10^-3 N/m\n\n")

        f.write("LINEÁRNÍ REGRESE (sigma_tab = B*t + C):\n")
        f.write(f"B = ({format_cz_pm(B_tab_l_u)}) 10^-3 N/(m·°C)\n")
        f.write(f"C = ({format_cz_pm(C_tab_l_u)}) 10^-3 N/m\n\n")
        
        f.write("======================================================================\n")
        f.write(" TABULKA NAMĚŘENÝCH HODNOT\n")
        f.write("======================================================================\n")
        f.write(f"{'t / °C':>14} | {'d_max / mm':>14} | {'Δp_max / Pa':>15} | {'σ / 10^-3 N/m':>16}\n")
        f.write("-" * 69 + "\n")
        for i in range(len(t_data)):
            f.write(f"{format_cz_pm(t_u[i]):>14} | {format_cz_pm(d_max_m_u[i]*1000.0):>14} | {format_cz_pm(dp_max_u[i]):>15} | {format_cz_pm(sigma_scaled_u[i]):>16}\n")
        f.write("-" * 69 + "\n\n")

        f.write("======================================================================\n")
        f.write(" SROVNÁNÍ EXPERIMENTU A TABULEK (Lineární model tabulek)\n")
        f.write("======================================================================\n")
        f.write(" Z-score určuje odchylku v násobcích standardní nejistoty měření u(σ).\n")
        f.write(" Hodnota > 2-3 obvykle značí statisticky významnou systematickou chybu.\n")
        f.write("-" * 88 + "\n")
        f.write(f"{'t / °C':>10} | {'σ_exp / 10^-3 N/m':>20} | {'σ_tab / 10^-3 N/m':>20} | {'Δσ (exp-tab)':>15} | {'Z-score':>10}\n")
        f.write("-" * 88 + "\n")
        for i in range(len(x_val)):
            exp_val = format_cz_pm(sigma_scaled_u[i])
            tab_val = format_cz_pm(sigma_tab_fit_at_x[i])
            delta = format_cz_pm(rozdil_sigma[i])
            z = format_cz_pm(z_score[i])
            f.write(f"{x_val[i]:>10.1f} | {exp_val:>20} | {tab_val:>20} | {delta:>15} | {z:>10}\n")
        f.write("-" * 88 + "\n")

if __name__ == '__main__':
    main()