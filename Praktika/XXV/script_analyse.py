import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import sqrt as usqrt
import warnings

# Skrytí neškodného varování knihovny uncertainties při nulové nejistotě
warnings.filterwarnings("ignore", message="Using UFloat objects with std_dev==0")

# ==============================================================================
# 1. KONFIGURACE A POMOCNÉ FUNKCE
# ==============================================================================

def round_uncertainty(value, error):
    """Zaokrouhlí nejistotu striktně na 1 platnou cifru a hodnotu na stejný řád."""
    if error == 0 or math.isnan(error):
        return f"{value} \\pm 0"
    
    order = math.floor(math.log10(abs(error)))
    err_rounded = round(error, -order)
    
    if err_rounded >= 10**(order+1):
        order += 1
        err_rounded = round(error, -order)
        
    val_rounded = round(value, -order)
    
    if order < 0:
        decimals = -order
        return f"{val_rounded:.{decimals}f} \\pm {err_rounded:.{decimals}f}"
    else:
        return f"{int(val_rounded)} \\pm {int(err_rounded)}"

def format_ufloat(u_val):
    return round_uncertainty(u_val.n, u_val.s)

def exp_decay(t, A0, delta):
    """Teoretická funkce pro obálku tlumených kmitů."""
    return A0 * np.exp(-delta * t)

# ==============================================================================
# 2. NAČTENÍ DAT (Z data.txt)
# ==============================================================================

def load_data(filename="data.txt"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    
    # OPRAVA: 'Vlastni' nyní používá seznamy pro ukládání všech řádků
    data = {
        'Vlastni': {'T_0': [], 'T_1': []},
        'Utlum': {'t':[], 'A':[]},
        'Nucene': {'f': [], 'A': [], 'phi':[]}
    }
    
    if not os.path.exists(filepath):
        print(f"Kritická chyba: Soubor nebyl nalezen na cestě:\n{filepath}")
        sys.exit(1)

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    current_section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'): continue
        
        if line.startswith('['):
            current_section = line.strip('[]')
            continue
            
        parts =[p.strip() for p in line.split('|')]
        
        # OPRAVENÁ LOGIKA: Dynamické přidávání do polí T_0 a T_1
        if current_section == 'Vlastni_kmity' and len(parts) >= 4:
            name = parts[0]
            t_celk = float(parts[1])
            err_celk = float(parts[2])
            N_kmitu = float(parts[3])
            
            # Výpočet periody a nejistoty pro jeden konkrétní řádek (T = t/N)
            val = t_celk / N_kmitu
            err = err_celk / N_kmitu
            
            # Pokud se omylem zadá jiný název (např. překlep T_2), skript vytvoří nový klíč
            if name not in data['Vlastni']:
                data['Vlastni'][name] =[]
                
            data['Vlastni'][name].append(ufloat(val, err))
            
        elif current_section == 'Utlum' and len(parts) >= 2:
            data['Utlum']['t'].append(float(parts[0]))
            data['Utlum']['A'].append(float(parts[1]))
            
        elif current_section == 'Nucene_kmity' and len(parts) >= 3:
            data['Nucene']['f'].append(float(parts[0]))
            data['Nucene']['A'].append(float(parts[1]))
            data['Nucene']['phi'].append(float(parts[2]))
            
    # Převod sekcí útlumu a nucených kmitů na numpy pole
    for key in['Utlum', 'Nucene']:
        for subkey in data[key]:
            data[key][subkey] = np.array(data[key][subkey])
            
    return data

# ==============================================================================
# 3. ZPRACOVÁNÍ A VÝPOČTY FYZIKÁLNÍCH VELIČIN
# ==============================================================================

data = load_data()

# --- Bezpečnostní kontroly dat ---
if not data['Vlastni'].get('T_0') or not data['Vlastni'].get('T_1'):
    print("Chyba: V sekci [Vlastni_kmity] chybí data pro T_0 nebo T_1.")
    sys.exit(1)
    
if len(data['Utlum']['t']) == 0:
    print("Chyba: Nebyla nalezena žádná data v sekci [Utlum].")
    sys.exit(1)
    
if len(data['Nucene']['f']) == 0:
    print("Chyba: Nebyla nalezena žádná data v sekci [Nucene_kmity].")
    sys.exit(1)

# --- A. Volné kmity (Zprůměrování všech zaznamenaných period) ---
# T_0_list obsahuje ufloat objekty všech měření. Součet ufloatů zachová šíření nejistot.
T_0_list = data['Vlastni']['T_0']
T_0 = sum(T_0_list) / len(T_0_list)

T_1_list = data['Vlastni']['T_1']
T_1 = sum(T_1_list) / len(T_1_list)

omega = (2 * np.pi) / T_0
omega_1_meas = (2 * np.pi) / T_1

# --- B. Regrese - Konstanta tlumení ---
t_data = data['Utlum']['t']
A_data = data['Utlum']['A']

# Regrese bezpečně proloží body, i když je v datech více pokusů pod sebou
popt, pcov = curve_fit(exp_decay, t_data, A_data, p0=[A_data[0], 0.1])
A0_fit = ufloat(popt[0], np.sqrt(pcov[0,0]))
delta = ufloat(popt[1], np.sqrt(pcov[1,1]))

omega_1_teor = usqrt(omega**2 - delta**2)

# --- C. Nucené kmity a rezonance ---
f_nuc = data['Nucene']['f']
A_nuc = data['Nucene']['A']
phi_nuc = data['Nucene']['phi']

Omega_nuc = 2 * np.pi * f_nuc

max_idx = np.argmax(A_nuc)

if 0 < max_idx < len(f_nuc) - 1:
    step_left = f_nuc[max_idx] - f_nuc[max_idx - 1]
    step_right = f_nuc[max_idx + 1] - f_nuc[max_idx]
    f_step_err = max(step_left, step_right) / 2.0
else:
    f_step_err = 0.05 

f_err_total = np.sqrt(f_step_err**2 + 0.005**2)
f_res_exp_u = ufloat(f_nuc[max_idx], f_err_total)
Omega_res_exp_u = 2 * np.pi * f_res_exp_u

Omega_res_teor = (omega**2) / usqrt(omega**2 - 2 * delta**2)

# ==============================================================================
# 4. EXPORT VÝSLEDKŮ DO TEXTOVÉHO SOUBORU
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, "vysledky_protokol.txt"), "w", encoding='utf-8') as fout:
    fout.write("VÝSLEDKY EXPERIMENTU PRO LABORATORNÍ PROTOKOL\n")
    fout.write("===============================================\n\n")
    
    fout.write("1. PARAMETRY VOLNÝCH KMITŮ (Zprůměrováno z měření)\n")
    fout.write(f"Zaznamenáno pokusů pro T_0:         {len(T_0_list)}\n")
    fout.write(f"Zaznamenáno pokusů pro T_1:         {len(T_1_list)}\n")
    fout.write(f"Výsledná perioda netlumených T_0:   {format_ufloat(T_0)} s\n")
    fout.write(f"Úhlová frekvence netlumená omega:   {format_ufloat(omega)} rad/s\n\n")
    
    fout.write("2. URČENÍ TLUMENÍ Z EXPONENCIÁLNÍ REGRESE\n")
    fout.write(f"Počáteční amplituda A_0:            {format_ufloat(A0_fit)} N\n")
    fout.write(f"Konstanta tlumení delta:            {format_ufloat(delta)} s^-1\n\n")
    
    fout.write("3. ÚHLOVÁ FREKVENCE TLUMENÝCH KMITŮ (POROVNÁNÍ)\n")
    fout.write(f"Změřená (z průměru T_1) omega_1:    {format_ufloat(omega_1_meas)} rad/s\n")
    fout.write(f"Vypočtená (z teorie) omega_1:       {format_ufloat(omega_1_teor)} rad/s\n\n")
    
    fout.write("4. REZONANCE ZRYCHLENÍ\n")
    fout.write(f"Experimentální vrchol Omega_res,a:  {format_ufloat(Omega_res_exp_u)} rad/s\n") 
    fout.write(f"Teoretická očekávaná Omega_res,a:   {format_ufloat(Omega_res_teor)} rad/s\n")

print(f"Výpočty dokončeny. Výsledky uloženy do 'vysledky_protokol.txt'.")

# ==============================================================================
# 5. VYKRESLENÍ GRAFŮ
# ==============================================================================

plt.rcParams.update({'font.size': 12})

# --- Graf 1 ---
plt.figure(figsize=(8, 5))
plt.scatter(t_data, A_data, color='black', label='Naměřená maxima', zorder=5)
# Jemná osa upravená i pro vícenásobná měření
t_fine = np.linspace(0, max(t_data)*1.1, 100)
plt.plot(t_fine, exp_decay(t_fine, A0_fit.n, delta.n), 'r-', label=f'Exponenciální regrese\n$\\delta = {format_ufloat(delta)}$ s$^{{-1}}$')
plt.xlabel(r'$t \mathrm{/s}$')
plt.ylabel(r'$F \mathrm{/N}$') 
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'graf1_utlum.png'), dpi=300)
plt.close()

# --- Graf 2 ---
plt.figure(figsize=(8, 5))
# Pokud bys křivku zhustil tak, že by hodnoty nešly po sobě chronologicky, sortujeme je pro čáru:
sort_idx = np.argsort(Omega_nuc)
plt.plot(Omega_nuc[sort_idx], A_nuc[sort_idx], 'ko-', label='Naměřená data')

plt.axvline(Omega_res_exp_u.n, color='red', linestyle='--', label=f'Exp. rezonance $\\Omega_{{res}} = {Omega_res_exp_u.n:.2f}$ rad/s')
plt.axvline(omega.n, color='blue', linestyle=':', label=f'Vlastní netlumená $\\omega = {omega.n:.2f}$ rad/s')

plt.xlabel(r'$\Omega \mathrm{/(rad \cdot s^{-1})}$')
plt.ylabel(r'$A_F \mathrm{/N}$')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'graf2_rezonance.png'), dpi=300)
plt.close()

# --- Graf 3 ---
plt.figure(figsize=(8, 5))
plt.plot(Omega_nuc[sort_idx], phi_nuc[sort_idx], 's-', color='darkgreen', label='Naměřené fázové posunutí')
plt.axvline(omega.n, color='blue', linestyle=':', label=f'Vlastní netlumená $\\omega = {omega.n:.2f}$ rad/s')

plt.xlabel(r'$\Omega \mathrm{/(rad \cdot s^{-1})}$')
plt.ylabel(r'$\varphi \mathrm{/rad}$')
plt.legend() 
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'graf3_faze.png'), dpi=300)
plt.close()

print("Grafy vygenerovány.")