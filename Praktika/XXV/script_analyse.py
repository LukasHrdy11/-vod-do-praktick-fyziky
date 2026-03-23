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

def to_float(s):
    """Bezpečný převod stringu na float (poradí si s čárkou i tečkou)."""
    return float(s.replace(',', '.'))

def exp_decay(t, A0, delta, C):
    """Teoretická funkce pro obálku tlumených kmitů."""
    return A0 * np.exp(-delta * t) + C

def lorentz_oscillator(Omega, C, Omega_res, gamma):
    """Teoretická rezonanční křivka tlumeného oscilátoru (Lorentzův profil)."""
    return C / np.sqrt((Omega_res**2 - Omega**2)**2 + (gamma * Omega)**2)

# ==============================================================================
# 2. NAČTENÍ DAT A ODCHYLEK (Z data.txt)
# ==============================================================================

def load_data(filename="data.txt"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    
    data = {
        'Odchylky': {'err_time': 0.0, 'err_freq': 0.0, 'err_force': 0.0},
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
            
        parts = [p.strip() for p in line.split('|')]
        
        if current_section == 'Odchylky' and len(parts) >= 2:
            name = parts[0].lower()
            val = to_float(parts[1])
            if 'čas' in name:
                data['Odchylky']['err_time'] = val
            elif 'frekvenc' in name:
                data['Odchylky']['err_freq'] = val
            elif 'síl' in name:
                data['Odchylky']['err_force'] = val
                
        elif current_section == 'Vlastni_kmity' and len(parts) >= 3:
            name = parts[0]
            t_celk = to_float(parts[1])
            N_kmitu = to_float(parts[2])
            
            err_celk = data['Odchylky']['err_time']
            t_celk_u = ufloat(t_celk, err_celk)
            val_u = t_celk_u / N_kmitu
            
            if name not in data['Vlastni']:
                data['Vlastni'][name] = []
                
            data['Vlastni'][name].append(val_u)
            
        elif current_section == 'Utlum' and len(parts) >= 2:
            data['Utlum']['t'].append(to_float(parts[0]))
            data['Utlum']['A'].append(to_float(parts[1]))
            
        elif current_section == 'Nucene_kmity' and len(parts) >= 3:
            data['Nucene']['f'].append(to_float(parts[0]))
            data['Nucene']['A'].append(to_float(parts[1]))
            data['Nucene']['phi'].append(to_float(parts[2]))
            
    for key in ['Utlum', 'Nucene']:
        for subkey in data[key]:
            data[key][subkey] = np.array(data[key][subkey])
            
    return data

# ==============================================================================
# 3. ZPRACOVÁNÍ A VÝPOČTY FYZIKÁLNÍCH VELIČIN
# ==============================================================================

data = load_data()
err_time = data['Odchylky']['err_time']
err_freq = data['Odchylky']['err_freq']
err_force = data['Odchylky']['err_force']

# Přepočet chyby frekvence na chybu úhlové frekvence
err_omega = 2 * np.pi * err_freq

# --- A. Volné kmity ---
T_0_list = data['Vlastni']['T_0']
T_0 = sum(T_0_list) / len(T_0_list)

T_1_list = data['Vlastni']['T_1']
T_1 = sum(T_1_list) / len(T_1_list)

omega = (2 * np.pi) / T_0
omega_1_meas = (2 * np.pi) / T_1

# --- B. Regrese - Konstanta tlumení ---
t_data = data['Utlum']['t']
A_data = data['Utlum']['A']

# Využití chyby síly (F) namísto staré amplitudy
sigma_A = np.full(len(A_data), err_force)
# Odhad počátečních parametrů: [A0, delta, C]
# C odhadneme jako minimum z naměřených dat, A0 jako rozdíl maxima a C
C_guess = min(A_data)
A0_guess = max(A_data) - C_guess

popt, pcov = curve_fit(
    exp_decay, 
    t_data, 
    A_data, 
    p0=[A0_guess, 0.1, C_guess], 
    sigma=sigma_A, 
    absolute_sigma=True
)

A0_fit = ufloat(popt[0], np.sqrt(pcov[0,0]))
delta = ufloat(popt[1], np.sqrt(pcov[1,1]))
C_fit = ufloat(popt[2], np.sqrt(pcov[2,2])) # Nový parametr C
omega_1_teor = usqrt(omega**2 - delta**2)

# --- C. Nucené kmity a rezonance (Lorentzův fit) ---
f_nuc = data['Nucene']['f']
A_nuc = data['Nucene']['A']
phi_nuc = data['Nucene']['phi']

Omega_nuc = 2 * np.pi * f_nuc
sigma_A_nuc = np.full(len(A_nuc), err_force)

err_phi_nuc = 2 * np.pi * f_nuc * err_time

# Najdeme index největší naměřené amplitudy (slouží pro výchozí odhad parametrů)
max_idx = np.argmax(A_nuc)

# Odhad počátečních parametrů [C, Omega_res, gamma]
p0_lorentz = [A_nuc[max_idx] * (omega.n**2), omega.n, 2 * delta.n]

try:
    # Fitování Lorentzovy křivky
    popt_L, pcov_L = curve_fit(
        lorentz_oscillator, 
        Omega_nuc, 
        A_nuc, 
        p0=p0_lorentz, 
        sigma=sigma_A_nuc, 
        absolute_sigma=True,
        maxfev=10000
    )
    # Extrahujeme Omega_res z parametrů fitu (je na indexu 1)
    Omega_res_fit_u = ufloat(popt_L[1], np.sqrt(pcov_L[1, 1]))
    fit_success = True
except Exception as e:
    print(f"Varování: Fitování Lorentzovy křivky selhalo ({e}).")
    # Záložní plán: použije jen nejvyšší naměřený bod, pokud fit selže (zde už s přesnou distribuovanou chybou frekvence)
    Omega_res_fit_u = 2 * np.pi * ufloat(f_nuc[max_idx], err_freq)
    fit_success = False

Omega_res_teor = (omega**2) / usqrt(omega**2 - 2 * delta**2)

# ==============================================================================
# 4. EXPORT VÝSLEDKŮ
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, "vysledky_protokol.txt"), "w", encoding='utf-8') as fout:
    fout.write("VÝSLEDKY EXPERIMENTU PRO LABORATORNÍ PROTOKOL\n")
    fout.write("===============================================\n\n")
    
    fout.write("1. PARAMETRY VOLNÝCH KMITŮ\n")
    fout.write(f"Výsledná perioda netlumených T_0:   {format_ufloat(T_0)} s\n")
    fout.write(f"Úhlová frekvence netlumená omega:   {format_ufloat(omega)} rad/s\n\n")
    
    fout.write("2. URČENÍ TLUMENÍ Z REGRESE\n")
    fout.write(f"Počáteční amplituda A_0:            {format_ufloat(A0_fit)} N\n")
    fout.write(f"Konstanta tlumení delta:            {format_ufloat(delta)} s^-1\n\n")
    fout.write(f"Absolutní člen (posun) C:           {format_ufloat(C_fit)} N\n\n")

    fout.write("3. ÚHLOVÁ FREKVENCE TLUMENÝCH KMITŮ\n")
    fout.write(f"Změřená (z průměru T_1) omega_1:    {format_ufloat(omega_1_meas)} rad/s\n")
    fout.write(f"Vypočtená (z teorie) omega_1:       {format_ufloat(omega_1_teor)} rad/s\n\n")
    
    fout.write("4. REZONANCE ZRYCHLENÍ\n")
    if fit_success:
        fout.write(f"Zjištěno proložením Lorentzovy křivky napříč všemi body:\n")
    fout.write(f"Experimentální vrchol Omega_res,a:  {format_ufloat(Omega_res_fit_u)} rad/s\n") 
    fout.write(f"Teoretická očekávaná Omega_res,a:   {format_ufloat(Omega_res_teor)} rad/s\n")

# ==============================================================================
# 5. VYKRESLENÍ GRAFŮ
# ==============================================================================

plt.rcParams.update({'font.size': 12})

# --- Graf 1: Útlum ---
plt.figure(figsize=(8, 5))
plt.errorbar(t_data, A_data, xerr=err_time, yerr=err_force, fmt='ko', label='Naměřená maxima', zorder=5, capsize=3, elinewidth=1)

t_fine = np.linspace(0, max(t_data)*1.1, 100)
plt.plot(t_fine, exp_decay(t_fine, A0_fit.n, delta.n, C_fit.n), 'r-', label=f'Exponenciální regrese\n$\\delta = {format_ufloat(delta)}$ s$^{{-1}}$\n$C = {format_ufloat(C_fit)}$ N\n$ A_0 = {format_ufloat(A0_fit)}$ N')
plt.xlabel(r'$t \mathrm{/s}$')
plt.ylabel(r'$F \mathrm{/N}$') 
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'graf1_utlum.png'), dpi=300)
plt.close()

# --- Graf 2: Rezonance ---
plt.figure(figsize=(8, 5))
sort_idx = np.argsort(Omega_nuc)

# Vizualizace naměřených dat s chybovými kříži využívajícími správné odchylky
plt.errorbar(Omega_nuc[sort_idx], A_nuc[sort_idx], xerr=err_omega, yerr=err_force, fmt='ko', label='Naměřená data', zorder=5, capsize=3, elinewidth=1)

# Vykreslení fitované Lorentzovy křivky
if fit_success:
    span = max(Omega_nuc) - min(Omega_nuc)
    Omega_fine = np.linspace(min(Omega_nuc) - span*0.2, max(Omega_nuc) + span*0.2, 500)
    plt.plot(Omega_fine, lorentz_oscillator(Omega_fine, *popt_L), 'm-', label=f'Fit rezonance (Lorentz)\n$\\Omega_{{res}} = {format_ufloat(Omega_res_fit_u)}$ rad/s')

plt.axvline(omega.n, color='blue', linestyle=':', label=f'Vlastní netlumená $\\omega = {omega.n:.2f}$ rad/s')

plt.xlabel(r'$\Omega \mathrm{/(rad \cdot s^{-1})}$')
plt.ylabel(r'$A_F \mathrm{/N}$')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'graf2_rezonance.png'), dpi=300)
plt.close()

# --- Graf 3: Fáze ---
plt.figure(figsize=(8, 5))
plt.errorbar(Omega_nuc[sort_idx], phi_nuc[sort_idx], xerr=err_omega, yerr=err_phi_nuc[sort_idx], fmt='s-', color='darkgreen', label='Naměřené fázové posunutí', capsize=3, elinewidth=1)
plt.axvline(omega.n, color='blue', linestyle=':', label=f'Vlastní netlumená $\\omega = {omega.n:.2f}$ rad/s')

plt.xlabel(r'$\Omega \mathrm{/(rad \cdot s^{-1})}$')
plt.ylabel(r'$\varphi \mathrm{/rad}$')
plt.legend() 
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'graf3_faze.png'), dpi=300)
plt.close()

print("Grafy vygenerovány a výsledky uloženy.")