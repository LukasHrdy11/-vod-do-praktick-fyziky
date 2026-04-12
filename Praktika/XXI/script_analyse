import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from uncertainties import ufloat
from uncertainties.umath import *

# =====================================================================
# KONFIGURACE SKRIPTU
# =====================================================================
INPUT_FILE = "data.txt"
OUTPUT_FILE = "vysledky_analyzy.txt"
PLOT_FILE = "graf_reverzni_kyvadlo.png"

# Nastavení globálního vzhledu grafů podle zvyklostí
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
    """
    Zaokrouhlí hodnotu a nejistotu podle pravidel fyzikálního praktika.
    Nejistota na 1 platnou cifru, hodnota na stejný počet desetinných míst.
    """
    val = u_val.n
    err = u_val.s
    
    if err == 0 or np.isnan(err):
        res = str(val).replace('.', ',')
        return f"{res} {unit}".strip()
        
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
    """Robustní načtení hybridního INI/CSV formátu."""
    data = {}
    current_section = None
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # Odstranění komentářů a prázdných znaků
            line = line.split('#')[0].strip()
            if not line:
                continue
                
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                data[current_section] = {}
                continue
                
            if current_section is None:
                continue
                
            # Dynamická detekce tabulek
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
# HLAVNÍ VÝPOČETNÍ BLOK
# =====================================================================
def analyze():
    print(f"Načítám data ze souboru {INPUT_FILE}...")
    try:
        data = parse_data(INPUT_FILE)
    except FileNotFoundError:
        print(f"CHYBA: Soubor {INPUT_FILE} nebyl nalezen.")
        return

    # --- NAČTENÍ NEJISTOT ---
    err_t = data['Nejistoty_pristroju']['err_t_s']
    err_m = data['Nejistoty_pristroju']['err_m_g'] / 1000
    err_l_m = data['Nejistoty_pristroju']['err_l_metr_mm'] / 1000
    err_l_p = data['Nejistoty_pristroju']['err_l_posuvka_mm'] / 1000

    # --- JEDNORÁZOVÁ MĚŘENÍ (převod na základní jednotky SI: kg, m) ---
    m_k = ufloat(data['Jednorazova_mereni']['m_k_g'], err_m * 1000) / 1000
    m_t = ufloat(data['Jednorazova_mereni']['m_t_g'], err_m * 1000) / 1000
    D_k = ufloat(data['Jednorazova_mereni']['D_k_mm'], err_l_p * 1000) / 1000
    h_h = ufloat(data['Jednorazova_mereni']['h_h_mm'], err_l_p * 1000) / 1000
    l_z = ufloat(data['Jednorazova_mereni']['l_z_mm'], err_l_m * 1000) / 1000
    L_r = ufloat(data['Jednorazova_mereni']['L_r_mm'], err_l_m * 1000) / 1000

    # =================================================================
    # 1. MATEMATICKÉ KYVADLO (Úkol 1)
    # =================================================================
    t10M_raw = np.array(data['Tabulka_1_Matematicke_kyvadlo']['t_10M'])
    
    t10M_mean = np.mean(t10M_raw)
    t10M_stat_err = np.std(t10M_raw, ddof=1) / np.sqrt(len(t10M_raw))
    t10M_total_err = np.sqrt(t10M_stat_err**2 + err_t**2)
    
    T_M = ufloat(t10M_mean, t10M_total_err) / 10
    
    L_M = l_z + h_h + (D_k / 2)
    g_M = (4 * np.pi**2 * L_M) / (T_M**2)

    # =================================================================
    # 2. CHYBA IDEALIZACE (Úkol 4 a 5)
    # =================================================================
    # Moment setrvačnosti idealizovaného mat. kyvadla (bodová masa)
    I_ideal = m_k * L_M**2
    
    # Skutečný moment setrvačnosti (započítání rozměru kuličky a hmotnosti provázku)
    I_koule = (2/5) * m_k * (D_k / 2)**2
    I_tyc = (1/12) * m_t * l_z**2
    I_celk = I_tyc + m_t * (l_z / 2)**2 + I_koule + m_k * L_M**2
    
    # Rozdíly v momentu setrvačnosti
    diff_I = I_celk - I_ideal
    rel_diff_I = (diff_I / I_ideal) * 100 # v procentech

    # Poloha těžiště
    d_tez = (m_t * (l_z / 2) + m_k * L_M) / (m_t + m_k)
    
    # Rozdíl v poloze těžiště vůči L_M
    diff_L = L_M - d_tez
    rel_diff_L = (diff_L / L_M) * 100 # v procentech

    # =================================================================
    # 3. REVERZNÍ KYVADLO (Úkol 2 a 3)
    # =================================================================
    lp_raw = np.array(data['Tabulka_2_Reverzni_kyvadlo']['l_p'])
    t10_up_raw = np.array(data['Tabulka_2_Reverzni_kyvadlo']['t_10_nahore'])
    t10_down_raw = np.array(data['Tabulka_2_Reverzni_kyvadlo']['t_10_dole'])

    T_up = t10_up_raw / 10
    T_down = t10_down_raw / 10

    reg_up = linregress(lp_raw, T_up)
    reg_down = linregress(lp_raw, T_down)

    k1 = ufloat(reg_up.slope, reg_up.stderr)
    q1 = ufloat(reg_up.intercept, reg_up.intercept_stderr)

    k2 = ufloat(reg_down.slope, reg_down.stderr)
    q2 = ufloat(reg_down.intercept, reg_down.intercept_stderr)

    lp_intersect = (q2 - q1) / (k1 - k2)
    T_r = k1 * lp_intersect + q1

    g_r = (4 * np.pi**2 * L_r) / (T_r**2)

    # --- VYKRESLENÍ GRAFU ---
    plt.figure(figsize=(9, 6))
    
    plt.plot(lp_raw, T_up, 'ko', markerfacecolor='none', markersize=8, label='čočka nahoře')
    plt.plot(lp_raw, T_down, 'ks', markersize=8, label='čočka dole')
    
    x_fit = np.linspace(min(lp_raw)-2, max(lp_raw)+2, 100)
    plt.plot(x_fit, reg_up.intercept + reg_up.slope * x_fit, 'k--', alpha=0.5)
    plt.plot(x_fit, reg_down.intercept + reg_down.slope * x_fit, 'k--', alpha=0.5)

    # Vyznačení průsečíku šipkou jako v návodu
    plt.annotate('', xy=(lp_intersect.n, plt.ylim()[0]), xytext=(lp_intersect.n, T_r.n),
                 arrowprops=dict(arrowstyle="->", linestyle="--", color='black'))

    plt.xlabel(r'$l_p$ / mm')
    plt.ylabel(r'$T$ / s')
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.savefig(PLOT_FILE, dpi=300, bbox_inches='tight')
    print(f"Graf uložen jako {PLOT_FILE}")

    # =================================================================
    # VÝSTUPNÍ PROTOKOL
    # =================================================================
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(" VÝSLEDKY ANALÝZY EXPERIMENTU: MĚŘENÍ TÍHOVÉHO ZRYCHLENÍ\n")
        f.write("="*60 + "\n\n")

        f.write("--- 1. MATEMATICKÉ KYVADLO ---\n")
        f.write(f"Zprůměrovaná perioda:       T_M = {format_result(T_M, 's')}\n")
        f.write(f"Délka idealiz. kyvadla:     L_M = {format_result(L_M, 'm')}\n")
        f.write(f"Vypočtené tíhové zrychlení: g_M = {format_result(g_M, 'm/s^2')}\n\n")

        f.write("--- 2. CHYBOVÁ ANALÝZA IDEALIZACE (Úkoly 4 a 5) ---\n")
        f.write("a) Srovnání momentů setrvačnosti:\n")
        f.write(f"   Idealizovaný model: I_ideal = {format_result(I_ideal, 'kg*m^2')}\n")
        f.write(f"   Reálné kyvadlo:     I_celk  = {format_result(I_celk, 'kg*m^2')}\n")
        f.write(f"   Rozdíl idealizace:  delta_I = {format_result(diff_I, 'kg*m^2')} ({format_result(rel_diff_I, '%')})\n\n")
        
        f.write("b) Srovnání délek:\n")
        f.write(f"   Délka matematického k.: L_M   = {format_result(L_M, 'm')}\n")
        f.write(f"   Skutečná poloha těžiště: d_tez = {format_result(d_tez, 'm')}\n")
        f.write(f"   Posun těžiště:          delta_L = {format_result(diff_L, 'm')} ({format_result(rel_diff_L, '%')})\n\n")

        f.write("--- 3. REVERZNÍ KYVADLO ---\n")
        f.write(f"Společná poloha čočky:  l_p0 = {format_result(lp_intersect, 'mm')}\n")
        f.write(f"Interpolovaná perioda:  T_r  = {format_result(T_r, 's')}\n")
        f.write(f"Redukovaná délka:       L_r  = {format_result(L_r, 'm')}\n")
        f.write(f"Vypočtené tíhové zrychlení: g_r  = {format_result(g_r, 'm/s^2')}\n")

    print(f"Výsledky byly úspěšně zapsány do {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze()