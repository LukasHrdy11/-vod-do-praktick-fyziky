import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

# =============================================================================
# KONFIGURACE SKRIPTU
# =============================================================================
DATA_FILE = "data.txt"                     
DATA_DIR = "data"                          
OUTPUT_RESULTS = "vysledky_analyzy.txt"    
GRAFY_DIR = "grafy"                        
GRAV_ZRYCHLENI = ufloat(9.811, 0.001)
DELIC_KMITU = 10                           

# SEZNAM MĚŘENÍ K VYŘAZENÍ Z REGRESE I* vs alpha
EXCLUDE_FROM_FIT = []

# Vytvoření složky pro grafy, pokud neexistuje
if not os.path.exists(GRAFY_DIR):
    os.makedirs(GRAFY_DIR)

# =============================================================================
# 1. FUNKCE PRO NAČÍTÁNÍ DATA.TXT
# =============================================================================
def load_data_txt(filepath):
    data = {'single': {}, 'tables': {}}
    current_section = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            match_section = re.match(r'\[(.*?)\]', line)
            if match_section:
                current_section = match_section.group(1)
                if current_section.startswith('Table_'):
                    data['tables'][current_section] = []
                continue
            
            if current_section in ['Header', 'Conditions', 'Instruments', 'Uncertainties', 'Measurements_Single']:
                if '=' in line:
                    key, val = line.split('=', 1)
                    key, val = key.strip(), val.strip()
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                    data['single'][key] = val
            elif current_section and current_section.startswith('Table_'):
                items = [x.strip() for x in line.split(',')]
                data['tables'][current_section].append(items)
                
    return data

# =============================================================================
# 2. POMOCNÉ MATEMATICKÉ FUNKCE
# =============================================================================
def linear_func(x, a, b):
    return a * x + b

def fit_line(x, y):
    popt, pcov = curve_fit(linear_func, x, y)
    a = ufloat(popt[0], np.sqrt(pcov[0,0]))
    b = ufloat(popt[1], np.sqrt(pcov[1,1]))
    return a, b

# =============================================================================
# 3. ZPRACOVÁNÍ METODY 1 - KYVADLO
# =============================================================================
def process_pendulum(data):
    single = data['single']
    tables = data['tables']
    
    err_mass = single['err_mass_g'] / 1000            
    err_caliper = single['err_caliper_cm'] / 100      
    err_weight_diam = single['err_weight_diameter_cm'] / 100 
    err_time = single['err_time_reaction_s']         
    err_gap_cm = single['err_gap_cm'] / 100 
    
    # Výpočet polohy těžiště l 
    d_hridel = ufloat(single['d_hridel'] / 100, err_caliper)
    dist_lambda = ufloat(single['lambda'] / 100, err_gap_cm) 
    D_zav = ufloat(single['diameter_weight_cm'] / 100, err_weight_diam)
    
    l_kyv = (d_hridel / 2) + dist_lambda + (D_zav / 2)
    m_kyv = ufloat(single['mass_main_weight_g'] / 1000, err_mass)
    
    # Výpočet periody T a její nejistoty
    t_vals = np.array([float(x[0]) for x in tables['Table_Periods']])
    t_mean = np.mean(t_vals)
    t_stat_err = np.std(t_vals, ddof=1) / np.sqrt(len(t_vals))
    T_10 = ufloat(t_mean, np.sqrt(t_stat_err**2 + err_time**2))
    T_kyv = T_10 / DELIC_KMITU
    
    g = GRAV_ZRYCHLENI
    I_kyv = m_kyv * l_kyv * ((g * T_kyv**2) / (4 * np.pi**2) - l_kyv)
    
    return I_kyv, m_kyv, l_kyv, T_kyv

# =============================================================================
# 4. ZPRACOVÁNÍ METODY 2 - OTÁČENÍ
# =============================================================================
def process_rotation(data):
    single = data['single']
    tables = data['tables']
    
    err_mass = single['err_mass_g'] / 1000
    err_shaft = single['err_shaft_diameter_cm'] / 100
    m_nit = single.get('m_nit', 0.0) / 1000  # Hmotnost nitě v kg
    
    # Namapování přesných poloměrů z tabulky
    real_shaft_diams = [float(x[0]) for x in tables['Table_Shaft_Diameters']]
    
    def get_real_diam(nom_diam):
        # Najde nejbližší reálný průměr k tomu nominálnímu z názvu souboru
        return min(real_shaft_diams, key=lambda x: abs(x - nom_diam))

    # Načtení hmotností a přičtení hmotnosti nitě
    mass_dict = {}
    for row in tables['Table_Weight_Masses']:
        mass_zavazi = ufloat(float(row[1]) / 1000, err_mass)
        mass_nit = ufloat(m_nit, err_mass) # Konzervativně použijeme stejnou chybu vah
        mass_dict[row[0]] = mass_zavazi + mass_nit
    
    rotation_results = []
    
    search_pattern = os.path.join(DATA_DIR, 'TO * mm & *.txt')
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"POZOR: Nebyly nalezeny žádné soubory ve složce '{DATA_DIR}'!")
        return []
        
    print(f"Nalezeno {len(files)} souborů pro metodu otáčení. Generuji ukázkový graf...")
    
    plot_data = [] # Pomocný seznam pro uložení dat ke kreslení

    for idx, file in enumerate(files):
        match = re.search(r'TO (\d+) mm & ([A-Z])\.txt', file)
        if not match:
            continue
        
        nom_diam_mm = int(match.group(1))
        weight_id = match.group(2)
        
        # Přiřazení skutečného průměru místo zaokrouhleného z názvu souboru
        real_diam_mm = get_real_diam(nom_diam_mm)
        r_val = (real_diam_mm / 1000) / 2
        r_shaft = ufloat(r_val, err_shaft / 2)
        
        m_weight = mass_dict[weight_id]
        
        # Načtení a regrese
        d = np.loadtxt(file, skiprows=1)
        t = d[:, 0]
        omega = d[:, 1]
        
        eps, om0 = fit_line(t, omega)
        alpha = 1 / eps
        g = GRAV_ZRYCHLENI
        I_star = m_weight * r_shaft**2 * ((g / (r_shaft * eps)) - 1)
        
        rotation_results.append({
            'file': file,
            'nom_mm': str(nom_diam_mm), # Pro účely filtrování
            'r_mm': real_diam_mm,       # Skutečný průměr
            'weight_id': weight_id,
            'eps': eps,
            'alpha': alpha,
            'I_star': I_star
        })
        
        # Uložení dat pro graf (potřebujeme je později vyfiltrovat na 3 vzorky)
        plot_data.append({
            't': t,
            'omega': omega,
            'eps_val': eps.n,
            'om0_val': om0.n,
            'label': f"Kl. {real_diam_mm:.1f}, Zav. {weight_id}: $\\omega = {eps.n:.4f}t + {om0.n:.4f}$"
        })
        
    # --- VYKRESLENÍ REPREZENTATIVNÍHO GRAFU (pouze 3 závislosti) ---
    if plot_data:
        # Seřadíme nasbíraná data podle zrychlení eps
        plot_data.sort(key=lambda x: x['eps_val'])
        
        # Vybereme indexy: nejmenší zrychlení, medián, největší zrychlení
        if len(plot_data) >= 3:
            selected_indices = [0, len(plot_data) // 2, len(plot_data) - 1]
            selected_plots = [plot_data[i] for i in selected_indices]
        else:
            selected_plots = plot_data # Pokud je dat náhodou méně než 3, vykreslí se všechna
            
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_plots)))
        plt.figure(figsize=(10, 6))
        
        for idx, p in enumerate(selected_plots):
            t_vals = p['t']
            plt.plot(t_vals, p['omega'], '.', color=colors[idx], markersize=5)
            
            t_fit = np.linspace(min(t_vals), max(t_vals), 100)
            omega_fit = p['eps_val'] * t_fit + p['om0_val']
            plt.plot(t_fit, omega_fit, '-', color=colors[idx], label=p['label'])
            
        plt.xlabel('t / s')
        plt.ylabel(r'$\omega$ / rad$\cdot$s$^{-1}$')
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        safename = "graf_ukazka_omega_vs_t.pdf"
        plt.savefig(os.path.join(GRAFY_DIR, safename), bbox_inches='tight', dpi=150)
        plt.close()

    return rotation_results

# =============================================================================
# 5. HLAVNÍ BLOK SKRIPTU
# =============================================================================
def main():
    print("Načítám data...")
    data = load_data_txt(DATA_FILE)
    
    I_kyv, m_kyv, l_kyv, T_kyv = process_pendulum(data)
    rot_res = process_rotation(data)
    
    M_t, I_k = None, None
    if len(rot_res) > 2:
        
        # Třídění dat na platná a vyřazená (podle EXCLUDE_FROM_FIT)
        alpha_valid, I_star_valid = [], []
        a_err_valid, i_err_valid = [], []
        
        alpha_excl, I_star_excl = [], []
        a_err_excl, i_err_excl = [], []
        
        for r in rot_res:
            is_excluded = any(r['nom_mm'] == ex[0] and r['weight_id'] == ex[1] for ex in EXCLUDE_FROM_FIT)
            
            if is_excluded:
                alpha_excl.append(r['alpha'].n)
                I_star_excl.append(r['I_star'].n)
                a_err_excl.append(r['alpha'].s)
                i_err_excl.append(r['I_star'].s)
            else:
                alpha_valid.append(r['alpha'].n)
                I_star_valid.append(r['I_star'].n)
                a_err_valid.append(r['alpha'].s)
                i_err_valid.append(r['I_star'].s)
                
        # Fitování POUZE platných bodů
        M_t, I_k = fit_line(alpha_valid, I_star_valid)
        
        # Generování grafu
        plt.figure(figsize=(8, 6))
        
        # Zahrnuté body
        plt.errorbar(alpha_valid, I_star_valid, xerr=a_err_valid, yerr=i_err_valid, 
                     fmt='bo', ecolor='gray', capsize=3, label='Hodnoty I* (zahrnuto)')
        
        # Vyřazené body (červené křížky)
        if alpha_excl:
            plt.errorbar(alpha_excl, I_star_excl, xerr=a_err_excl, yerr=i_err_excl, 
                         fmt='rx', ecolor='red', capsize=3, label='Vyřazená hodnota')
                     
        alpha_fit = np.linspace(min(alpha_valid + alpha_excl), max(alpha_valid + alpha_excl), 100)
        I_fit = M_t.n * alpha_fit + I_k.n
        
        eq_label = f"Regrese: $I^* = {M_t.n:.5f}\\alpha + {I_k.n:.5f}$"
        plt.plot(alpha_fit, I_fit, 'r-', label=eq_label)
        
        plt.xlabel(r'$\alpha$ / s$^{2}$')
        plt.ylabel(r'$I^*$ / kg$\cdot$m$^2$')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(GRAFY_DIR, "graf_I_star_vs_alpha.pdf"), bbox_inches='tight', dpi=150)
        plt.close()
        
    # --- ZÁPIS VÝSLEDKŮ DO SOUBORU ---
    with open(OUTPUT_RESULTS, 'w', encoding='utf-8') as f:
        f.write("=========================================================\n")
        f.write("      VÝSLEDKY ANALÝZY: STUDIUM OTÁČENÍ TUHÉHO TĚLESA    \n")
        f.write("=========================================================\n\n")
        
        f.write("--- METODA 1: METODA KYVŮ ---\n")
        f.write(f"Hmotnost závaží m:       {m_kyv:.1u} kg\n")
        f.write(f"Poloha těžiště l:        {l_kyv:.1u} m\n")
        f.write(f"Perioda T:               {T_kyv:.1u} s\n")
        f.write(f">> Moment setrvačnosti I:{I_kyv:.1u} kg*m^2\n\n")
        
        f.write("--- METODA 2: METODA OTÁČENÍ ---\n")
        if not rot_res:
             f.write("Zadna data pro tuto metodu nebyla nalezena.\n")
        else:
            f.write("Tabulka dílčích výsledků:\n")
            f.write(f"{'Válec(mm)':<12} {'Závaží':<8} {'epsilon(rad/s^2)':<20} {'alpha(s^2)':<20} {'I*(kg*m^2)':<20} {'Poznámka':<10}\n")
            f.write("-" * 90 + "\n")
            
            for r in sorted(rot_res, key=lambda x: (x['r_mm'], x['weight_id'])):
                is_excl = "<Vyřazeno z fitu>" if any(r['nom_mm'] == ex[0] and r['weight_id'] == ex[1] for ex in EXCLUDE_FROM_FIT) else ""
                
                f.write(f"{r['r_mm']:<12.1f} {r['weight_id']:<8} "
                        f"{r['eps']:.1u}    ".ljust(20) + 
                        f"{r['alpha']:.1u}    ".ljust(20) + 
                        f"{r['I_star']:.1u}    ".ljust(20) + 
                        f"{is_excl}\n")
                
            f.write("\n")
            if M_t and I_k:
                f.write("--- VÝSLEDEK REGRESE (Oddělení tření) ---\n")
                f.write(f">> Korigovaný moment setrv. I_k: {I_k:.1u} kg*m^2\n")
                f.write(f">> Moment třecích sil M_t:       {M_t:.1u} N*m\n")

    print(f"Analýza úspěšně dokončena.")
    print(f"Výsledky zapsány do: {OUTPUT_RESULTS}")
    print(f"Grafy uloženy do složky: {GRAFY_DIR}")

if __name__ == "__main__":
    main()