import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
from collections import defaultdict

# ==============================================================================
# KONFIGURACE
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(SCRIPT_DIR, 'data.txt')
OUTPUT_TEXT = os.path.join(SCRIPT_DIR, 'vysledky_protokol.txt')
OUTPUT_PLOT = os.path.join(SCRIPT_DIR, 'graf_regrese.png')

R_KONST = 8.314  # J/(mol*K)
NA_TAB = 6.02214076e23  # mol^-1 (přesná tabulková hodnota)
# ==============================================================================

def parse_data(filename):
    data = {
        'T_val': None, 'T_err': None,
        'd_raw': [],
        'measurements': defaultdict(lambda: {'dt': [], 'dt_err':[], 's2': [], 's2_err':[]})
    }
    
    current_section = None
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                continue
                
            if current_section == 'PODMINKY':
                line_clean = line.split('#')[0].strip()
                if line_clean:
                    parts = [float(x.strip()) for x in line_clean.split(',')]
                    data['T_val'] = parts[0]
                    data['T_err'] = parts[1]
            elif current_section == 'PRUMER_CASTIC':
                data['d_raw'].append(float(line.replace(',', '.')))
            elif current_section == 'BROWN_POSUNUTI':
                parts =[float(x.strip()) for x in line.split(',')]
                m_id = int(parts[0])
                data['measurements'][m_id]['dt'].append(parts[1])
                data['measurements'][m_id]['dt_err'].append(parts[2])
                data['measurements'][m_id]['s2'].append(parts[3])
                data['measurements'][m_id]['s2_err'].append(parts[4])
                
    data['d_raw'] = np.array(data['d_raw'])
    for m_id in data['measurements']:
        for key in ['dt', 'dt_err', 's2', 's2_err']:
            data['measurements'][m_id][key] = np.array(data['measurements'][m_id][key])
            
    return data

def linear_model(x, a):
    return a * x

def calculate_variant_results(activities_list, r_u, eta_u, T_kelvin):
    """Pomocná funkce pro výpočet výsledků z daného seznamu aktivit."""
    # Aritmetický průměr a jeho standardní chyba
    A_mean_val = np.mean([a.n for a in activities_list])
    A_mean_err = np.std([a.n for a in activities_list], ddof=1) / np.sqrt(len(activities_list))
    A_mean_u = ufloat(A_mean_val, A_mean_err)
    
    # Převod do SI
    A_SI = A_mean_u * 1e-12
    r_SI = r_u * 1e-9
    eta_SI = eta_u * 1e-3
    
    # Výpočet NA
    N_a = (R_KONST * T_kelvin) / (3 * np.pi * eta_SI * r_SI * A_SI)
    
    # Signifikance a odchylky
    absolutni_odchylka = abs(N_a.n - NA_TAB)
    relativni_odchylka = (absolutni_odchylka / NA_TAB) * 100
    z_score = absolutni_odchylka / N_a.s
    
    return A_mean_u, N_a, absolutni_odchylka, relativni_odchylka, z_score

def main():
    data = parse_data(DATA_FILE)
    
    # 1. Teplota a dynamická viskozita
    T_celzius = ufloat(data['T_val'], data['T_err'])
    T_kelvin = T_celzius + 273.15
    eta_u = ufloat(0.89, 0.06) 
    
    # 2. Velikost částic s KOREKCÍ o nejistotu typu B
    d_mean = np.mean(data['d_raw'])
    d_std_A = np.std(data['d_raw'], ddof=1) # Výběrová nejistota typu A
    d_std_B = 4.0 # 2.0 nm pro poloměr znamená 4.0 nm pro průměr
    d_std_celkova = np.sqrt(d_std_A**2 + d_std_B**2)
    
    d_u = ufloat(d_mean, d_std_celkova)
    r_u = d_u / 2
    
    # 3. Zpracování všech měření Brownova pohybu
    plt.figure(figsize=(9, 6))
    colors =['#1f77b4', '#ff7f0e', '#2ca02c']
    markers =['o', 's', '^']
    
    activities_all = []
    activities_good = []
    report_lines = []
    
    for i, (m_id, m_data) in enumerate(sorted(data['measurements'].items())):
        x_data = m_data['dt']
        y_data = m_data['s2']
        y_err = m_data['s2_err']
        
        # Regrese y = a * x
        popt, pcov = curve_fit(linear_model, x_data, y_data, sigma=y_err, absolute_sigma=True)
        a_val = popt[0]
        a_err = np.sqrt(pcov[0, 0])
        
        a_u = ufloat(a_val, a_err)
        A_u = a_u / 2  # Aktivita A
        
        # Rozřazení do variant
        activities_all.append(A_u)
        if m_id != 2:  # Vyřazení anomálního měření 2
            activities_good.append(A_u)
            
        # Přidání do grafu
        plt.errorbar(x_data, y_data, xerr=m_data['dt_err'], yerr=y_err, 
                     fmt=markers[i%len(markers)], color=colors[i%len(colors)], 
                     capsize=4, label=f'Měření {m_id}')
        
        x_fit = np.linspace(0, max(x_data)*1.05, 100)
        y_fit = linear_model(x_fit, a_val)
        plt.plot(x_fit, y_fit, color=colors[i%len(colors)], linestyle='--', alpha=0.7)
        
        report_lines.append(f"  Měření {m_id}: směrnice = {a_u:.1u} μm^2/s  =>  A = {A_u:.1u} μm^2/s")

    # 4. Výpočty pro obě varianty
    res_A = calculate_variant_results(activities_all, r_u, eta_u, T_kelvin)
    res_B = calculate_variant_results(activities_good, r_u, eta_u, T_kelvin)

    # 5. Generování textového reportu
    report =[
        "===========================================================",
        " VÝSLEDKY ANALÝZY: Úloha 16 - Studium Brownova pohybu",
        "===========================================================\n",
        "[1] PARAMETRY A PODMÍNKY",
        f"  Teplota měření:   T = {T_celzius:.1u} °C  ({T_kelvin:.1u} K)",
        f"  Viskozita vody:   η = {eta_u:.1u} mPa·s\n",
        "[2] VELIKOST ČÁSTIC (vč. nejistoty typu B)",
        f"  Průměr částice:   d = {d_u:.1u} nm",
        f"  Poloměr částice:  r = {r_u:.1u} nm\n",
        "[3] AKTIVITY BROWNOVA POHYBU (Jednotlivá měření)",
        *report_lines,
        "\n===========================================================",
        " VARIANTA A: VŠECHNA MĚŘENÍ (včetně č. 2)",
        "===========================================================",
        f"  Průměrná aktivita: A_průměr = {res_A[0]:.1u} μm^2/s",
        f"  Avogadrova konst.: N_A      = {res_A[1]:.1u} mol^-1",
        f"  Odchylka od N_A_tab: δ = {res_A[3]:.1f} %, Z-score = {res_A[4]:.1f} σ",
    ]
    
    if res_A[4] <= 2.0:
        report.append("  -> Výsledek je V SHODĚ s tabulkovou hodnotou (v rámci 2σ).")
    else:
        report.append("  -> Výsledek se SIGNIFIKANTNĚ LIŠÍ od tabulkové hodnoty (>2σ).")

    report.extend([
        "\n===========================================================",
        " VARIANTA B: BEZ ANOMÁLNÍHO MĚŘENÍ 2",
        "===========================================================",
        f"  Průměrná aktivita: A_průměr = {res_B[0]:.1u} μm^2/s",
        f"  Avogadrova konst.: N_A      = {res_B[1]:.1u} mol^-1",
        f"  Odchylka od N_A_tab: δ = {res_B[3]:.1f} %, Z-score = {res_B[4]:.1f} σ",
    ])

    if res_B[4] <= 2.0:
        report.append("  -> Výsledek je V SHODĚ s tabulkovou hodnotou (v rámci 2σ).")
    else:
        report.append("  -> Výsledek se SIGNIFIKANTNĚ LIŠÍ od tabulkové hodnoty (>2σ).")

    report.append("===========================================================")

    with open(OUTPUT_TEXT, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
        
    # Dokončení grafu
    plt.xlabel(r'$\Delta t \, / \, \mathrm{s}$', fontsize=12)
    plt.ylabel(r'$\overline{s^2} \, / \, \mathrm{\mu m^2}$', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)

if __name__ == '__main__':
    main()