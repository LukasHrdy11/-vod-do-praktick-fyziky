import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.stats import linregress
import re
import os

def load_data(filename="data.txt"):
    """Jednoduchý parser pro načtení proměnných a tabulek ze souboru data.txt."""
    data = {}
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parsování jednorázových proměnných
    for line in lines:
        line = line.strip()
        if '=' in line and not line.startswith('#') and not line.startswith('['):
            key, val = line.split('=')
            try:
                data[key.strip()] = float(val.split('#')[0].strip())
            except ValueError:
                pass # Ignorujeme nečíselné hodnoty (např. jména)

    # Parsování Tabulky 1 (Matematické kyvadlo)
    t10_m = []
    in_tab1 = False
    for line in lines:
        if '[Tabulka_1_Matematicke_kyvadlo]' in line: in_tab1 = True; continue
        if in_tab1 and line.startswith('['): break
        if in_tab1 and ',' in line and not line.startswith('#') and not line.startswith('N'):
            parts = line.split(',')
            t10_m.append(float(parts[1].strip()))
    data['t10_M'] = np.array(t10_m)

    # Parsování Tabulky 2 (Reverzní kyvadlo)
    l_p, t_nahore, t_dole = [], [], []
    in_tab2 = False
    for line in lines:
        if '[Tabulka_2_Reverzni_kyvadlo]' in line: in_tab2 = True; continue
        if in_tab2 and line.startswith('['): break
        if in_tab2 and ',' in line and not line.startswith('#') and not line.startswith('l_p'):
            parts = line.split(',')
            l_p.append(float(parts[0].strip()))
            t_nahore.append(float(parts[1].strip()))
            t_dole.append(float(parts[2].strip()))
    data['l_p'] = np.array(l_p)
    data['t_nahore'] = np.array(t_nahore)
    data['t_dole'] = np.array(t_dole)

    return data

def simulate_period(g, I, M, d, theta0_deg=5.0, gamma=0.005):
    """
    Numerická integrace diferenciální rovnice kyvadla:
    I * d^2(theta)/dt^2 + gamma * d(theta)/dt + M * g * d * sin(theta) = 0
    Vrací periodu kmitů.
    """
    theta0 = np.radians(theta0_deg)
    
    def eq_of_motion(t, y):
        theta, omega = y
        dtheta_dt = omega
        domega_dt = - (M * g * d / I) * np.sin(theta) - gamma * omega
        return [dtheta_dt, domega_dt]

    # Událost pro detekci průchodu nulou (z kladných do záporných hodnot)
    def zero_crossing(t, y): return y[0]
    zero_crossing.direction = -1
    zero_crossing.terminal = False

    # Simulujeme na dostatečně dlouhý časový interval
    T_approx = 2 * np.pi * np.sqrt(I / (M * g * d))
    t_span = (0, T_approx * 5)
    
    sol = solve_ivp(eq_of_motion, t_span, [theta0, 0], 
                    events=zero_crossing, max_step=T_approx/1000)
    
    # Perioda je rozdíl mezi dvěma po sobě jdoucími průchody nulou stejným směrem
    if len(sol.t_events[0]) >= 2:
        return sol.t_events[0][1] - sol.t_events[0][0]
    return T_approx

def main():
    data = load_data('data.txt')
    
    # --- 1. MATEMATICKÉ KYVADLO (Fyzikální model) ---
    # Převod na SI jednotky
    m_k = data['m_k_g'] / 1000
    m_t = data['m_t_g'] / 1000
    D_k = data['D_k_mm'] / 1000
    R = D_k / 2
    h_h = data['h_h_mm'] / 1000
    l_z = data['l_z_mm'] / 1000
    
    L_str = l_z + h_h  # celková délka provázku
    L_M_ideal = L_str + R # idealizovaná délka
    
    M_celk = m_k + m_t
    
    # Těžiště (od osy otáčení)
    d_tez_str = L_str / 2
    d_tez_k = L_str + R
    d_tez_celk = (m_t * d_tez_str + m_k * d_tez_k) / M_celk
    
    # Momenty setrvačnosti (Steinerova věta)
    I_str = (1/3) * m_t * L_str**2
    I_k = (2/5) * m_k * R**2 + m_k * d_tez_k**2
    I_celk = I_str + I_k
    I_ideal = m_k * L_M_ideal**2
    
    # Experimentální perioda
    T_M_exp = np.mean(data['t10_M']) / 10

    # Hledání skutečného g přes digitální dvojče (root-finding)
    # Hledáme takové g, pro které numerická simulace vrátí přesně změřenou periodu
    def g_root_func(g_guess):
        return simulate_period(g_guess, I_celk, M_celk, d_tez_celk) - T_M_exp
    
    g_M_sim = brentq(g_root_func, 9.0, 10.0)

    # --- 2. REVERZNÍ KYVADLO ---
    l_p = data['l_p']
    T_n = data['t_nahore'] / 10
    T_d = data['t_dole'] / 10
    
    # Lineární regrese pro nalezení průsečíku (T_n(l) a T_d(l))
    slope_n, int_n, _, _, _ = linregress(l_p, T_n)
    slope_d, int_d, _, _, _ = linregress(l_p, T_d)
    
    # Průsečík: slope_n * l + int_n = slope_d * l + int_d
    l_p0 = (int_d - int_n) / (slope_n - slope_d)
    T_r = slope_n * l_p0 + int_n
    
    L_r = data['L_r_mm'] / 1000
    g_r = 4 * np.pi**2 * L_r / (T_r**2)

    # --- TABULKOVÁ HODNOTA A CHYBY ---
    g_tab = 9.811
    err_g = 0.001
    
    # Generování výstupního textu
    output_text = f"""============================================================
 VÝSLEDKY SIMULACE EXPERIMENTU: MĚŘENÍ TÍHOVÉHO ZRYCHLENÍ
============================================================

--- 1. MATEMATICKÉ KYVADLO ---
Zprůměrovaná perioda:       T_M = {T_M_exp:.4f} s
Délka idealiz. kyvadla:     L_M = {L_M_ideal:.4f} m
Vypočtené tíhové zrychlení: g_M = {g_M_sim:.4f} m/s^2

--- 2. CHYBOVÁ ANALÝZA IDEALIZACE ---
a) Srovnání momentů setrvačnosti:
   Idealizovaný model: I_ideal = {I_ideal:.6f} kg*m^2
   Reálné kyvadlo:     I_celk  = {I_celk:.6f} kg*m^2
   Rozdíl idealizace:  delta_I = {abs(I_ideal - I_celk):.6f} kg*m^2 ({abs(I_ideal - I_celk)/I_ideal * 100:.2f} %)

b) Srovnání délek:
   Délka matematického k.: L_M     = {L_M_ideal:.5f} m
   Skutečná poloha těžiště: d_tez  = {d_tez_celk:.5f} m
   Posun těžiště:          delta_L = {abs(L_M_ideal - d_tez_celk):.5f} m ({abs(L_M_ideal - d_tez_celk)/L_M_ideal * 100:.2f} %)

--- 3. REVERZNÍ KYVADLO ---
Společná poloha čočky:  l_p0 = {l_p0:.2f} mm
Interpolovaná perioda:  T_r  = {T_r:.5f} s
Redukovaná délka:       L_r  = {L_r:.4f} m
Vypočtené tíhové zrychlení: g_r  = {g_r:.4f} m/s^2

--- 4. POROVNÁNÍ S TABULKOVOU HODNOTOU A DISKUZE ---
Tabulková hodnota: g_tab = (9,811 \u00b1 0,001) m/s^2

a) MATEMATICKÉ KYVADLO:
   Absolutní odchylka: Delta g_M = {abs(g_M_sim - g_tab):.4f} m/s^2
   Relativní odchylka:           = {abs(g_M_sim - g_tab)/g_tab * 100:.2f} %
   Shoda v rámci nejistot:   z_M = {abs(g_M_sim - g_tab) / err_g:.1f} sigma
   Relativní přesnost měření:    = {abs(g_M_sim - g_tab)/g_tab * 100:.2f} %

b) REVERZNÍ KYVADLO:
   Absolutní odchylka: Delta g_r = {abs(g_r - g_tab):.4f} m/s^2
   Relativní odchylka:           = {abs(g_r - g_tab)/g_tab * 100:.2f} %
   Shoda v rámci nejistot:   z_r = {abs(g_r - g_tab) / err_g:.1f} sigma
   Relativní přesnost měření:    = {abs(g_r - g_tab)/g_tab * 100:.2f} %
"""

    # Uložení do souboru
    with open("vysledky_simulace.txt", "w", encoding="utf-8") as f:
        f.write(output_text)
    
    print("Simulace dokončena. Výsledky uloženy do 'vysledky_simulace.txt'.")

if __name__ == "__main__":
    main()