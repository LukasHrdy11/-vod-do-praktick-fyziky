import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import sqrt
import os

# ==========================================
# 1. KONFIGURACE
# ==========================================
DATA_FILE = 'data.txt'
OUTPUT_FILE = 'vysledky_analyzy.txt'
PLOT_FILE = 'graf_rezonator.pdf'

# ==========================================
# 2. PARSOVÁNÍ DAT
# ==========================================
def load_data(filename):
    data = {'scalars': {}, 'tables': {}}
    current_section = None
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            
            # Detekce sekcí
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                if current_section in ['REZONATOR_VZDUCH', 'REZONATOR_CO2']:
                    data['tables'][current_section] = {'k':[], 'f':[]}
                elif current_section == 'KUNDTOVA_TRUBICE_MERENI':
                    data['tables'][current_section] = {'l_dutina':[], 'd':[], 'k':[]}
                continue
            
            # Načítání skalárních hodnot (a textového seznamu délek)
            if current_section in ['PODMINKY', 'NEJISTOTY', 'KONSTANTY', 'KUNDTOVA_TRUBICE_PARAMETRY', 'REZONATOR_PROM_DELKA']:
                if '=' in line:
                    key, val = line.split('=', 1)
                    key = key.strip()
                    val = val.strip()
                    
                    if key == 'delky':
                        data['scalars'][key] = val 
                    else:
                        data['scalars'][key] = float(val)
                continue
            
            # Načítání tabulek pro rezonátor
            if current_section in ['REZONATOR_VZDUCH', 'REZONATOR_CO2']:
                parts = line.split(',')
                if len(parts) == 2:
                    data['tables'][current_section]['k'].append(float(parts[0].strip()))
                    data['tables'][current_section]['f'].append(float(parts[1].strip()))
                continue
                    
            # Načítání tabulky pro Kundtovu trubici (3 sloupce)
            if current_section == 'KUNDTOVA_TRUBICE_MERENI':
                parts = line.split(',')
                if len(parts) == 3:
                    data['tables'][current_section]['l_dutina'].append(float(parts[0].strip()))
                    data['tables'][current_section]['d'].append(float(parts[1].strip()))
                    data['tables'][current_section]['k'].append(int(parts[2].strip()))
                    
    return data

# ==========================================
# 3. VÝPOČTY A ŠÍŘENÍ NEJISTOT
# ==========================================
def analyze():
    print("Načítám data a provádím analýzu...")
    d = load_data(DATA_FILE)
    s = d['scalars']
    
    # --- Příprava proměnných s nejistotami (ufloat) ---
    t = ufloat(s['t'], s['err_t'])
    T = t + 273.15  
    
    l_tyc = ufloat(s['l_tyc'], s['err_l_tyc'])
    rho_mosaz = ufloat(s['rho_mosaz'], s['err_rho'])
    
    # --- A) KUNDTOVA TRUBICE ---
    lam_mosaz = 2 * l_tyc
    
    kundt_data = d['tables']['KUNDTOVA_TRUBICE_MERENI']
    lam_vzduch_list = []
    
    # Výpočet vlnové délky pro body, kde byl odečten počet půlvln (k > 0)
    for i in range(len(kundt_data['k'])):
        if kundt_data['k'][i] > 0:
            d_val = ufloat(kundt_data['d'][i], s['err_d'])
            k_val = kundt_data['k'][i]
            # Výpočet z konkrétního rezonančního obrazce
            lam_v = (2 * d_val) / k_val
            lam_vzduch_list.append(lam_v)
            
    if lam_vzduch_list:
        lam_vzduch = sum(lam_vzduch_list) / len(lam_vzduch_list)
    else:
        print("CHYBA: V datech pro Kundtovu trubici není zaznamenáno žádné platné měření (k > 0)!")
        lam_vzduch = ufloat(1, 0) # Dummy hodnota pro zabránění pádu skriptu
    
    c_vzduch_ref = 331.82 + 0.61 * t
    c_mosaz = c_vzduch_ref * (lam_mosaz / lam_vzduch)
    E_mosaz = (c_mosaz**2) * rho_mosaz
    
    # --- B) UZAVŘENÝ REZONÁTOR (REGRESE - PROMĚNNÁ FREKVENCE) ---
    l_rez = ufloat(s.get('l_rez', 0.5), s.get('err_l_rez', 0.002))
    
    def linear_fit(x, a, b):
        return a * x + b

    f_vzd = np.array(d['tables']['REZONATOR_VZDUCH']['f'])
    k_vzd = np.array(d['tables']['REZONATOR_VZDUCH']['k'])
    popt_vzd, pcov_vzd = curve_fit(linear_fit, f_vzd, k_vzd)
    a_vzd = ufloat(popt_vzd[0], np.sqrt(pcov_vzd[0][0])) 
    c_vzduch_rez = (2 * l_rez) / a_vzd
    
    f_co2 = np.array(d['tables']['REZONATOR_CO2']['f'])
    k_co2 = np.array(d['tables']['REZONATOR_CO2']['k'])
    popt_co2, pcov_co2 = curve_fit(linear_fit, f_co2, k_co2)
    a_co2 = ufloat(popt_co2[0], np.sqrt(pcov_co2[0][0]))
    c_co2_rez = (2 * l_rez) / a_co2

    # --- C) UZAVŘENÝ REZONÁTOR (PROMĚNNÁ DÉLKA - VZDUCH) ---
    f_konst = ufloat(s['f_konst'], s['err_f'])
    delky_str = s['delky'].split(',')
    delky_arr = np.array([float(x.strip()) for x in delky_str])
    rozdily = np.abs(np.diff(delky_arr))
    prum_delta_l = ufloat(np.mean(rozdily), s['err_l_rez'])
    c_vzduch_delka = 2 * prum_delta_l * f_konst
    
    # --- D) POISSONOVA KONSTANTA CO2 ---
    R_konst = s['R']
    mu = s['mu_co2']
    kappa = (c_co2_rez**2 * mu) / (R_konst * T)

    # ==========================================
    # 4. GENEROVÁNÍ GRAFU
    # ==========================================
    plt.figure(figsize=(8, 6))
    err_f_val = s.get('err_f', 0)
    err_k_val = s.get('err_k', 0)
    
    # Data a fit pro vzduch
    plt.errorbar(f_vzd, k_vzd, xerr=err_f_val, yerr=err_k_val, fmt='o', color='blue', 
                 label='Vzduch (data)', capsize=3, linestyle='None')
    f_fit_vzd = np.linspace(min(f_vzd)*0.9, max(f_vzd)*1.1, 100)
    plt.plot(f_fit_vzd, linear_fit(f_fit_vzd, *popt_vzd), color='blue', linestyle='--', label='Vzduch (fit)')
    
    # Data a fit pro CO2
    plt.errorbar(f_co2, k_co2, xerr=err_f_val, yerr=err_k_val, fmt='s', color='red', 
                 label='CO$_2$ (data)', capsize=3, linestyle='None')
    f_fit_co2 = np.linspace(min(f_co2)*0.9, max(f_co2)*1.1, 100)
    plt.plot(f_fit_co2, linear_fit(f_fit_co2, *popt_co2), color='red', linestyle='--', label='CO$_2$ (fit)')
    
    plt.xlabel('f / Hz', fontsize=12)
    plt.ylabel('k', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    print(f"Graf byl uložen jako: {PLOT_FILE}")

    # ==========================================
    # 5. VÝSTUPNÍ TEXTOVÝ SOUBOR
    # ==========================================
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write("VÝSLEDKY ANALÝZY: Rychlost šíření zvuku\n")
        f.write("==================================================\n\n")
        
        f.write("--- 1. KUNDTOVA TRUBICE ---\n")
        f.write(f"Zaznamenaných délek dutiny: {len(kundt_data['l_dutina'])}\n")
        f.write(f"Z toho proměřených rezonancí (k > 0): {len(lam_vzduch_list)}\n")
        f.write(f"Vlnová délka v mosazi:    lambda_1 = {lam_mosaz:.1u} m\n")
        f.write(f"Vlnová délka ve vzduchu (průměr): lambda_2 = {lam_vzduch:.1u} m\n")
        f.write(f"Teoretická ref. rychlost: c_vzd_ref = {c_vzduch_ref:.1u} m/s\n")
        f.write(f"Rychlost zvuku v mosazi:  c_mosaz = {c_mosaz:.1u} m/s\n")
        f.write(f"Youngův modul (mosaz):    E = {E_mosaz:.1u} Pa\n\n")
        
        f.write("--- 2. UZAVŘENÝ REZONÁTOR (Lineární regrese, proměnná f) ---\n")
        f.write("Model fitu: k = a * f + b\n")
        f.write(f"VZDUCH:\n")
        f.write(f"  Směrnice a = {a_vzd:.1u} s\n")
        f.write(f"  Rychlost zvuku: c_vzd = {c_vzduch_rez:.1u} m/s\n\n")
        f.write(f"OXID UHLIČITÝ (CO2):\n")
        f.write(f"  Směrnice a = {a_co2:.1u} s\n")
        f.write(f"  Rychlost zvuku: c_CO2 = {c_co2_rez:.1u} m/s\n\n")
        
        f.write("--- 3. UZAVŘENÝ REZONÁTOR (Proměnná délka, konst. f) ---\n")
        f.write(f"Průměrný rozdíl délek:    delta_l = {prum_delta_l:.1u} m\n")
        f.write(f"Rychlost zvuku (vzduch):  c_vzd_delka = {c_vzduch_delka:.1u} m/s\n\n")

        f.write("--- 4. POISSONOVA KONSTANTA ---\n")
        f.write(f"Termodynamická teplota:    T = {T:.1u} K\n")
        f.write(f"Poissonova konst. CO2:     kappa = {kappa:.1u}\n")
        
    print(f"Výsledky byly uloženy do souboru: {OUTPUT_FILE}")

if __name__ == '__main__':
    if not os.path.exists(DATA_FILE):
        print(f"CHYBA: Soubor '{DATA_FILE}' nebyl nalezen ve stejné složce!")
    else:
        analyze()