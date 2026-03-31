import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat, unumpy
import warnings

warnings.filterwarnings('ignore') # Potlačí varování při fitování malého vzorku

# =============================================================================
# 1) KONFIGURACE A VIZUÁLNÍ STYL
# =============================================================================
DATA_FILE = "data.txt"
RESULTS_FILE = "vysledky_analyzy.txt"

OUT_DIR = "grafy"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# Globální nastavení vzhledu grafů pro čistý, profesionální vzhled do protokolu
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'legend.framealpha': 0.9,
    'lines.linewidth': 1.5,
    'lines.markersize': 7,
    'figure.autolayout': True,
    'axes.grid': True,
    'grid.alpha': 0.5,
    'grid.linestyle': '--'
})

# =============================================================================
# 2) POMOCNÉ FUNKCE
# =============================================================================
def format_unc(val, err):
    """Zaokrouhlí hodnotu a nejistotu na 1 platnou cifru nejistoty."""
    if err == 0 or math.isnan(err) or math.isinf(err):
        return f"{val:.4g}"
    try:
        place = -int(math.floor(math.log10(err)))
        round_err = round(err, place)
        round_val = round(val, place)
        if place <= 0:
            return f"{int(round_val)} ± {int(round_err)}"
        else:
            return f"{round_val:.{place}f} ± {round_err:.{place}f}"
    except Exception:
        return f"{val} ± {err}"

def lin_fit_affine(x, a, b):
    """ Funkce pro afinní fit: y = ax + b """
    return a * x + b

def format_eq_b(b, dec=2):
    """ Zformátuje absolutní člen pro legendu (vyhne se zápisu '+ (-x)') """
    if b >= 0:
        return f"+ {b:.{dec}f}"
    else:
        return f"- {abs(b):.{dec}f}"

# =============================================================================
# 3) NAČÍTÁNÍ DAT
# =============================================================================
metadata = {}
errors = {}
stat_data = []
dyn_data = []

if not os.path.exists(DATA_FILE):
    print(f"Chyba: Soubor '{DATA_FILE}' nebyl nalezen. Vytvořte jej prosím s potřebnými daty.")
    exit(1)

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

state = 0 
for line in lines:
    line = line.split('#')[0].strip()
    if not line:
        continue
    
    if line == "[DATA_STATIKA]":
        state = 1
        continue
    elif line == "[DATA_DYNAMIKA]":
        state = 2
        continue
        
    if state == 0:
        if ':' in line:
            key, val = [part.strip() for part in line.split(':', 1)]
            try:
                errors[key] = float(val)
            except ValueError:
                metadata[key] = val
    elif state == 1:
        if 'pruzina' in line.lower(): continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 4:
            try:
                stat_data.append({
                    'pruzina': parts[0],
                    'm_g': float(parts[1]),
                    'h_nezat': float(parts[2]),
                    'h_zat': float(parts[3])
                })
            except ValueError:
                pass
    elif state == 2:
        if 'pruzina' in line.lower(): continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 5:
            try:
                dyn_data.append({
                    'pruzina': parts[0],
                    'm_g': float(parts[1]),
                    't1': float(parts[2]),
                    't2': float(parts[3]),
                    'n': float(parts[4])
                })
            except ValueError:
                pass

err_m = errors.get('chyba_m_g', 0.1) / 1000
err_h = errors.get('chyba_h_mm', 1.0) / 1000
err_t = errors.get('chyba_t_s', 0.04)
err_g = errors.get('chyba_g_ms2', 0.001)
g_tab = ufloat(errors.get('g_tabulkove_ms2', 9.811), err_g)

# =============================================================================
# 4) VÝPOČTY A ŠÍŘENÍ NEJISTOT
# =============================================================================
stat_results = {}
dyn_results = {}
g_results = []

# --- A) STATICKÁ METODA ---
pruziny_stat = sorted(list(set([d['pruzina'] for d in stat_data])))

plt.figure(figsize=(8, 6))
ax = plt.gca()
for p in pruziny_stat:
    F_vals, y0_vals = [], []
    F_errs, y0_errs = [], []
    
    for d in stat_data:
        if d['pruzina'] == p:
            m_u = ufloat(d['m_g'] / 1000, err_m)
            F_u = m_u * g_tab
            
            h1_u = ufloat(d['h_nezat'] / 100, err_h)
            h2_u = ufloat(d['h_zat'] / 100, err_h)
            y0_u = abs(h2_u - h1_u)
            
            F_vals.append(F_u.n)
            F_errs.append(F_u.s)
            y0_vals.append(y0_u.n)
            y0_errs.append(y0_u.s)

    F_vals, y0_vals = np.array(F_vals), np.array(y0_vals)
    F_errs, y0_errs = np.array(F_errs), np.array(y0_errs)
    
    # Graf: osa X je síla F, osa Y je prodloužení y0
    scatter = ax.errorbar(F_vals, y0_vals, xerr=F_errs, yerr=y0_errs, fmt='o', capsize=3, elinewidth=1.2, label=f'Pružina {p} (data)')
    color = scatter[0].get_color()
    
    if len(y0_vals) >= 2:
        try:
            # Fitování y = a*F + b, kde a = 1/k
            popt, pcov = curve_fit(lin_fit_affine, F_vals, y0_vals, sigma=y0_errs, absolute_sigma=True)
            
            # Parametr a (směrnice) a b (posun)
            a_stat = ufloat(popt[0], np.sqrt(pcov[0, 0]))
            b_stat = ufloat(popt[1], np.sqrt(pcov[1, 1]))
            
            # Výpočet tuhosti k = 1 / a (včetně automatického přenosu chyb)
            if a_stat.n != 0:
                k_stat = 1 / a_stat
            else:
                k_stat = ufloat(0, 0)
            
            # Vykreslení fitu (x je F, y je prodloužení)
            x_fit = np.linspace(0, max(F_vals)*1.05, 100)
            y_fit = lin_fit_affine(x_fit, *popt)
            eq_label = f'{p} fit: y = {popt[0]:.4f}F {format_eq_b(popt[1], 4)}'
            ax.plot(x_fit, y_fit, '-', color=color, alpha=0.8, label=eq_label)
        except RuntimeError:
            k_stat, b_stat = ufloat(0, 0), ufloat(0, 0)
    else:
        k_stat, b_stat = ufloat(0, 0), ufloat(0, 0)

    stat_results[p] = {'k': k_stat, 'b': b_stat}

ax.set_xlabel('F / N')
ax.set_ylabel('y / m')
# Bez titulku pro protokol
ax.legend(loc='best')
plt.savefig(os.path.join(OUT_DIR, 'graf_statika.png'), dpi=300)
plt.close()

# --- B) DYNAMICKÁ METODA ---
pruziny_dyn = sorted(list(set([d['pruzina'] for d in dyn_data])))

plt.figure(figsize=(8, 6))
ax = plt.gca()
for p in pruziny_dyn:
    inv_sqrt_m_vals, w_vals = [], []
    inv_sqrt_m_errs, w_errs = [], []
    
    for d in dyn_data:
        if d['pruzina'] == p:
            m_u = ufloat(d['m_g'] / 1000, err_m)
            inv_sqrt_m_u = 1 / unumpy.sqrt(m_u)
            
            t1_u = ufloat(d['t1'], err_t)
            t2_u = ufloat(d['t2'], err_t)
            n_kmity = d['n']
            if n_kmity <= 0: continue
                
            T_u = (t2_u - t1_u) / n_kmity
            w_u = 2 * math.pi / T_u
            
            inv_sqrt_m_vals.append(inv_sqrt_m_u.n)
            inv_sqrt_m_errs.append(inv_sqrt_m_u.s)
            w_vals.append(w_u.n)
            w_errs.append(w_u.s)
            
            for s_d in stat_data:
                if s_d['pruzina'] == p and s_d['m_g'] == d['m_g']:
                    y0_u = abs(ufloat(s_d['h_zat']/100, err_h) - ufloat(s_d['h_nezat']/100, err_h))
                    g_u = y0_u * (w_u**2)
                    g_results.append(g_u)

    inv_sqrt_m_vals = np.array(inv_sqrt_m_vals)
    inv_sqrt_m_errs = np.array(inv_sqrt_m_errs)
    w_vals = np.array(w_vals)
    w_errs = np.array(w_errs)
    
    scatter = ax.errorbar(inv_sqrt_m_vals, w_vals, yerr=w_errs, xerr=inv_sqrt_m_errs, fmt='o', capsize=3, elinewidth=1.2, label=f'Pružina {p} (data)')
    color = scatter[0].get_color()
    
    if len(inv_sqrt_m_vals) >= 2:
        try:
            popt, pcov = curve_fit(lin_fit_affine, inv_sqrt_m_vals, w_vals, sigma=w_errs, absolute_sigma=True)
            sqrt_k_dyn = ufloat(popt[0], np.sqrt(pcov[0, 0]))
            b_dyn = ufloat(popt[1], np.sqrt(pcov[1, 1]))
            k_dyn = sqrt_k_dyn**2
            
            x_fit = np.linspace(min(inv_sqrt_m_vals)*0.95, max(inv_sqrt_m_vals)*1.05, 100)
            eq_label = rf'{p} fit: $\omega$ = {popt[0]:.1f} / $\sqrt{{m}}$ {format_eq_b(popt[1])}'
            ax.plot(x_fit, lin_fit_affine(x_fit, *popt), '-', color=color, alpha=0.8, label=eq_label)
        except RuntimeError:
            k_dyn, b_dyn = ufloat(0, 0), ufloat(0, 0)
    else:
        k_dyn, b_dyn = ufloat(0, 0), ufloat(0, 0)
        
    dyn_results[p] = {'k': k_dyn, 'b': b_dyn}

ax.set_xlabel(r'1/$\sqrt{m}$ / kg$^{-1/2}$')
ax.set_ylabel(r'$\omega$ / s$^{-1}$')
# Bez titulku pro protokol
ax.legend(loc='best')
plt.savefig(os.path.join(OUT_DIR, 'graf_dynamika_omega_vs_m.png'), dpi=300)
plt.close()

# --- C) TÍHOVÉ ZRYCHLENÍ ---
if g_results:
    g_mean_arr = unumpy.uarray([v.n for v in g_results], [v.s for v in g_results])
    weights = 1 / (unumpy.std_devs(g_mean_arr)**2)
    if np.any(np.isinf(weights)):
        g_final = ufloat(np.mean(unumpy.nominal_values(g_mean_arr)), 0)
    else:
        g_final_val = np.sum(unumpy.nominal_values(g_mean_arr) * weights) / np.sum(weights)
        g_final_err = np.sqrt(1 / np.sum(weights))
        g_final = ufloat(g_final_val, g_final_err)
else:
    g_final = ufloat(0, 0)

# --- D) GRAF ZÁVISLOSTI OMEGA NA ODMOCNINĚ Z K ---
plt.figure(figsize=(8, 6))
ax = plt.gca()
masses = sorted(list(set([d['m_g'] for d in dyn_data])))

for m_val in masses:
    sqrt_k_vals = []
    sqrt_k_errs = []
    w_vals = []
    w_errs = []
    
    for d in dyn_data:
        if d['m_g'] == m_val:
            p = d['pruzina']
            if p in stat_results and stat_results[p]['k'].n > 0:
                k_u = stat_results[p]['k']
                sqrt_k_u = k_u**0.5
                
                t1_u = ufloat(d['t1'], err_t)
                t2_u = ufloat(d['t2'], err_t)
                n_kmity = d['n']
                if n_kmity > 0:
                    T_u = (t2_u - t1_u) / n_kmity
                    w_u = 2 * math.pi / T_u
                    
                    sqrt_k_vals.append(sqrt_k_u.n)
                    sqrt_k_errs.append(sqrt_k_u.s)
                    w_vals.append(w_u.n)
                    w_errs.append(w_u.s)
    
    # Podmínka pro vykreslení: více než jeden datový bod pro danou hmotnost
    if len(sqrt_k_vals) > 1:
        sqrt_k_vals = np.array(sqrt_k_vals)
        sqrt_k_errs = np.array(sqrt_k_errs)
        w_vals = np.array(w_vals)
        w_errs = np.array(w_errs)
        
        scatter = ax.errorbar(sqrt_k_vals, w_vals, xerr=sqrt_k_errs, yerr=w_errs, fmt='s', capsize=3, elinewidth=1.2, label=f'm = {m_val} g (data)')
        color = scatter[0].get_color()
        
        try:
            popt, pcov = curve_fit(lin_fit_affine, sqrt_k_vals, w_vals, sigma=w_errs, absolute_sigma=True)
            x_fit = np.linspace(min(sqrt_k_vals)*0.95, max(sqrt_k_vals)*1.05, 100)
            eq_label = rf'Fit m={m_val}g: $\omega$ = {popt[0]:.2f} $\sqrt{{k}}$ {format_eq_b(popt[1])}'
            ax.plot(x_fit, lin_fit_affine(x_fit, *popt), '--', color=color, alpha=0.8, label=eq_label)
        except RuntimeError:
            pass

ax.set_xlabel(r'$\sqrt{k}$ / (N/m)$^{1/2}$')
ax.set_ylabel(r'$\omega$ / s$^{-1}$')
ax.legend(loc='upper left', fontsize=9)
plt.savefig(os.path.join(OUT_DIR, 'graf_dynamika_omega_vs_k.png'), dpi=300)
plt.close()

# =============================================================================
# 5) VÝSTUP VÝSLEDKŮ
# =============================================================================
with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
    f.write("=== VÝSLEDKY EXPERIMENTU ===\n")
    f.write(f"Zpracováno pro experiment: {metadata.get('Nazev_experimentu', 'N/A')}\n")
    f.write(f"Experimentátor: {metadata.get('Experimentator', 'N/A')}\n")
    f.write("="*65 + "\n\n")
    
    f.write("--- 1. TUHOST PRUŽIN (Statická metoda) ---\n")
    f.write("Rovnice regrese: y = (1/k)*F + b\n")
    f.write(f"{'Pružina':<10} | {'Tuhost k (N/m)':<25} | {'Posun b (m)':<25}\n")
    f.write("-" * 65 + "\n")
    for p in pruziny_stat:
        k_str = format_unc(stat_results[p]['k'].n, stat_results[p]['k'].s)
        b_str = format_unc(stat_results[p]['b'].n, stat_results[p]['b'].s)
        f.write(f"{p:<10} | {k_str:<25} | {b_str:<25}\n")
    
    f.write("\n--- 2. TUHOST PRUŽIN (Dynamická metoda) ---\n")
    f.write("Rovnice regrese: \u03C9 = \u221Ak * (1/\u221Am) + b\n")
    f.write(f"{'Pružina':<10} | {'Tuhost k (N/m)':<25} | {'Posun b (1/s)':<25}\n")
    f.write("-" * 65 + "\n")
    for p in pruziny_dyn:
        k_str = format_unc(dyn_results[p]['k'].n, dyn_results[p]['k'].s)
        b_str = format_unc(dyn_results[p]['b'].n, dyn_results[p]['b'].s)
        f.write(f"{p:<10} | {k_str:<25} | {b_str:<25}\n")

    f.write("\n--- 3. MÍSTNÍ TÍHOVÉ ZRYCHLENÍ (Kombinovaná metoda) ---\n")
    f.write(f"Vypočtené g = {format_unc(g_final.n, g_final.s)} m/s^2\n")
    f.write(f"(Tabelované g = {format_unc(g_tab.n, g_tab.s)} m/s^2)\n")

print(f"\nAnalýza úspěšně dokončena! Výsledky uloženy do '{RESULTS_FILE}'.")
print(f"Grafy vygenerovány do složky '{OUT_DIR}/'.")