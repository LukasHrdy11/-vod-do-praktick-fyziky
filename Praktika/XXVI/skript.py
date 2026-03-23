import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from uncertainties import ufloat
from uncertainties import unumpy as unp
import os

# ==========================================
# KONFIGURACE A NASTAVENÍ GRAFŮ
# ==========================================
SOUBOR_DATA = 'data.txt'
SOUBOR_VYSTUP = 'vysledky_protokol.txt'

plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# ==========================================
# POMOCNÉ FUNKCE
# ==========================================
def nacti_sekci(filename, sekce):
    data =[]
    in_section = False
    if not os.path.exists(filename):
        print(f"Chyba: Soubor {filename} nebyl nalezen.")
        return np.array([])
        
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith(f"[{sekce}]"):
                in_section = True
                continue
            if in_section:
                if line.startswith('['): 
                    break
                if not line.startswith('#') and line != '':
                    try:
                        hodnoty = [float(x) for x in line.split()]
                        data.append(hodnoty)
                    except ValueError:
                        pass
    return np.array(data)

def fmt(u_obj):
    """Zformátuje ufloat objekt na 1 platnou cifru nejistoty (např. 1.23 ± 0.04)"""
    if np.isnan(u_obj.n): return "NaN"
    return f"{u_obj:.1u}".replace('+/-', ' ± ')

def ufloat_ze_3_mereni(mereni, chyba_B):
    """Vypočítá průměr a celkovou nejistotu (Typ A + Typ B) ze seznamu 3 měření"""
    prumer = np.mean(mereni)
    # Výběrová směrodatná odchylka průměru (Typ A)
    u_A = np.std(mereni, ddof=1) / np.sqrt(len(mereni))
    # Typ B z rovnoměrného rozdělení (chyba / sqrt(3))
    u_B = chyba_B / np.sqrt(3)
    u_C = np.sqrt(u_A**2 + u_B**2)
    return ufloat(prumer, u_C)

# ==========================================
# NAČTENÍ CHYB A DAT
# ==========================================
chyby_data = nacti_sekci(SOUBOR_DATA, 'PRISTROJE_A_CHYBY')[0]
u_F = chyby_data[0]
u_l = chyby_data[1]
u_f_B = chyby_data[2]
u_t = chyby_data[3]

# Tloušťka struny
d_data = nacti_sekci(SOUBOR_DATA, 'TLOUSTKA_STRUNY')[0]
d_struny = ufloat_ze_3_mereni(d_data, chyby_data[4])

with open(SOUBOR_VYSTUP, 'w', encoding='utf-8') as f_out:
    f_out.write("====== VÝSLEDKY ANALÝZY PRO PROTOKOL ======\n\n")
    f_out.write(f"Změřená tloušťka struny d = {fmt(d_struny)} mm\n\n")

    # ------------------------------------------
    # ÚKOL 1: f = f(F) (zpracování základních frekvencí n=1)
    # ------------------------------------------
    data1 = nacti_sekci(SOUBOR_DATA, 'UKOL_1')
    if data1.size > 0:
        F_list, f1_list = [], []
        
        for radek in data1:
            if radek[1] == 1: # Filtrujeme pouze základní frekvence (n=1)
                F_val = ufloat(radek[0], u_F/np.sqrt(3))
                f_val = ufloat_ze_3_mereni(radek[2:5], u_f_B)
                F_list.append(F_val)
                f1_list.append(f_val)
                
        F_unp = np.array(F_list)
        f_unp = np.array(f1_list)
        
        # Transformace x = sqrt(F) - uncertainties knihovna sama přenese chybu
        x1_unp = unp.sqrt(F_unp)
        y1_unp = f_unp
        
        # Regrese (na nominálních hodnotách pro vykreslení)
        x1_nom = unp.nominal_values(x1_unp)
        y1_nom = unp.nominal_values(y1_unp)
        res1 = linregress(x1_nom, y1_nom)
        
        k1 = ufloat(res1.slope, res1.stderr)
        q1 = ufloat(res1.intercept, res1.intercept_stderr)
        
        # Generování grafu
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(x1_nom, y1_nom, xerr=unp.std_devs(x1_unp), yerr=unp.std_devs(y1_unp), 
                    fmt='o', color='blue', label='Naměřená data', capsize=3)
        ax.plot(x1_nom, res1.slope*x1_nom + res1.intercept, 'r-', label='Lineární regrese')
        ax.set_xlabel(r'$\sqrt{F}$ / $\mathrm{N}^{1/2}$')
        ax.set_ylabel(r'$f_1$ / $\mathrm{Hz}$')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('graf_ukol1.png', dpi=300)
        plt.close()
        
        # Výpis
        f_out.write("--- ÚKOL 1: Závislost frekvence na napětí struny ---\n")
        f_out.write(f"Rovnice regrese: f = k * sqrt(F) + q\n")
        f_out.write(f"Směrnice k = {fmt(k1)} Hz/N^(1/2)\n")
        f_out.write(f"Úsek q     = {fmt(q1)} Hz\n\n")
        f_out.write("F [N]\t sqrt(F)[N^(1/2)]\t f1 [Hz]\n")
        for i in range(len(F_list)):
            f_out.write(f"{fmt(F_unp[i])}\t {fmt(x1_unp[i])}\t {fmt(y1_unp[i])}\n")
        f_out.write("\n")

    # ------------------------------------------
    # ÚKOL 2: f = f(l)
    # ------------------------------------------
    data2 = nacti_sekci(SOUBOR_DATA, 'UKOL_2')
    if data2.size > 0:
        l_list, f2_list = [], []
        
        for radek in data2:
            if radek[1] == 1: # Opět bereme základní harmonickou
                l_val = ufloat(radek[0], u_l/np.sqrt(3))
                f_val = ufloat_ze_3_mereni(radek[2:5], u_f_B)
                l_list.append(l_val)
                f2_list.append(f_val)
                
        l_unp = np.array(l_list)
        f_unp = np.array(f2_list)
        
        # Transformace x = 1/l
        x2_unp = 1.0 / l_unp
        y2_unp = f_unp
        
        x2_nom = unp.nominal_values(x2_unp)
        y2_nom = unp.nominal_values(y2_unp)
        res2 = linregress(x2_nom, y2_nom)
        
        k2 = ufloat(res2.slope, res2.stderr)
        q2 = ufloat(res2.intercept, res2.intercept_stderr)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(x2_nom, y2_nom, xerr=unp.std_devs(x2_unp), yerr=unp.std_devs(y2_unp), 
                    fmt='s', color='green', label='Naměřená data', capsize=3)
        ax.plot(x2_nom, res2.slope*x2_nom + res2.intercept, 'r-', label='Lineární regrese')
        ax.set_xlabel(r'$1/l$ / $\mathrm{m}^{-1}$')
        ax.set_ylabel(r'$f_1$ / $\mathrm{Hz}$')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('graf_ukol2.png', dpi=300)
        plt.close()
        
        f_out.write("--- ÚKOL 2: Závislost frekvence na délce struny ---\n")
        f_out.write(f"Rovnice regrese: f = k * (1/l) + q\n")
        f_out.write(f"Směrnice k = {fmt(k2)} m/s (pozn. toto je c/2)\n")
        f_out.write(f"Úsek q     = {fmt(q2)} Hz\n\n")
        f_out.write("l [m]\t 1/l [m^-1]\t f1 [Hz]\n")
        for i in range(len(l_list)):
            f_out.write(f"{fmt(l_unp[i])}\t {fmt(x2_unp[i])}\t {fmt(y2_unp[i])}\n")
        f_out.write("\n")

    # ------------------------------------------
    # ÚKOL 3: Zázněje
    # ------------------------------------------
    data3 = nacti_sekci(SOUBOR_DATA, 'UKOL_3')
    if data3.size > 0:
        f_out.write("--- ÚKOL 3: Studium rázů (Zázněje) ---\n")
        f_out.write("f_A [Hz]\t f_B [Hz]\t f_teorie [Hz]\t N\t t [s]\t f_mereno [Hz]\n")
        
        for radek in data3:
            fA = ufloat(radek[0], u_f_B/np.sqrt(3))
            fB = ufloat(radek[1], u_f_B/np.sqrt(3))
            N_razu = radek[2]
            t_mereno = ufloat(radek[3], u_t/np.sqrt(3))
            
            # Výpočet pomocí uncertainties
            f_teorie = abs(fA - fB)
            f_mereno = N_razu / t_mereno
            
            f_out.write(f"{fmt(fA)}\t {fmt(fB)}\t {fmt(f_teorie)}\t {int(N_razu)}\t {fmt(t_mereno)}\t {fmt(f_mereno)}\n")
        f_out.write("\n")

print("Analýza dokončena! Zkontroluj soubor 'vysledky_protokol.txt' a vygenerované PNG grafy pro úkoly 1 a 2.")