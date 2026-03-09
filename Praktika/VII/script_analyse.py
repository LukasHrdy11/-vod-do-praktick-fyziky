import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# KONFIGURACE
# ==============================================================================
INPUT_FILE = "data.txt"
OUTPUT_TXT = "vysledky.txt"
OUTPUT_PLOT_IDEAL = "graf_kappa_idealni.png"
OUTPUT_PLOT_FULL = "graf_kappa_kompletni.png"
# ==============================================================================

def calc_omega(n, t_list, delta_t_syst):
    """Vypočítá střední periodu, omega a jejich celkové nejistoty."""
    t_arr = np.array(t_list)
    t_mean = np.mean(t_arr)
    
    if len(t_arr) > 1:
        t_stat = np.std(t_arr, ddof=1) / np.sqrt(len(t_arr))
    else:
        t_stat = 0.0
        
    t_err = np.sqrt(t_stat**2 + delta_t_syst**2)
    
    T = t_mean / n
    dT = t_err / n
    w = 2 * np.pi / T
    dw = w * (dT / T)
    
    return w, dw, T, dT, t_mean, t_err

def calc_kappa(w1, dw1, w2, dw2):
    """Vypočítá stupeň vazby kappa a jeho nejistotu pomocí zákona přenosu chyb."""
    kappa = (w2**2 - w1**2) / (w2**2 + w1**2)
    denominator = (w1**2 + w2**2)**2
    dk_dw1 = (-4 * w1 * w2**2) / denominator
    dk_dw2 = (4 * w1**2 * w2) / denominator
    dkappa = np.sqrt((dk_dw1 * dw1)**2 + (dk_dw2 * dw2)**2)
    return kappa, dkappa

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"CHYBA: Soubor {INPUT_FILE} nebyl nalezen!")
        return

    constants = {}
    uncoupled_data = {}
    fixed_data = {'A': {}, 'B': {}}
    kappa_vs_h = []

    # 1. PARSOVÁNÍ SOUBORU
    current_section = None
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                continue
                
            if current_section == "CONSTANTS":
                key, val = line.split('=')
                constants[key.strip()] = float(val.strip())
                
            elif current_section == "UNCOUPLED":
                # OPRAVA: Nahrazujeme čárky mezerami, aby se zachovala čísla jako 18.8
                parts = line.replace(',', ' ').split()
                name = parts[0]
                n = float(parts[1])
                t_list = [float(x) for x in parts[2:]]
                uncoupled_data[name] = (n, t_list)
            
            elif current_section == "FIXED_DISTANCE":
                # OPRAVA: Nahrazujeme čárky mezerami
                parts = line.replace(',', ' ').split()
                spring = parts[0]
                mode = int(parts[1])
                n = float(parts[2])
                t_list = [float(x) for x in parts[3:]]
                fixed_data[spring][mode] = (n, t_list)
                
            elif current_section == "KAPPA_VS_H":
                # OPRAVA: Nahrazujeme čárky mezerami
                parts = line.replace(',', ' ').split()
                if len(parts) >= 3:
                    h = float(parts[0])
                    t1_list = [float(parts[1])]
                    t2_list = [float(x) for x in parts[2:]]
                    kappa_vs_h.append((h, t1_list, t2_list))

    dt_norm = constants.get('delta_t', 0.2)
    dt_T4 = constants.get('delta_t4', 1.5)

    # 2. VÝPOČTY A ZÁPIS VÝSLEDKŮ
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as out:
        out.write("=========================================================\n")
        out.write("VÝSLEDKY MĚŘENÍ: Studium kmitů vázaných oscilátorů\n")
        out.write("=========================================================\n\n")

        # --- ÚKOL 1: Nevázaná kyvadla ---
        if uncoupled_data:
            out.write("--- 1. ÚKOL: NEVÁZANÁ KYVADLA (Kalibrace) ---\n")
            for name, (n, t_list) in uncoupled_data.items():
                w, dw, T, dT, t_mean, t_err = calc_omega(n, t_list, dt_norm)
                out.write(f"{name}:\n")
                out.write(f"  Perioda T0 = {T:.3f} +/- {dT:.3f} s\n")
                out.write(f"  Frekvence w0 = {w:.3f} +/- {dw:.3f} s^-1\n\n")

        # --- ÚKOL 2 a 3: Vázaná kyvadla ---
        for spring in ['A', 'B']:
            if spring not in fixed_data or not fixed_data[spring]:
                continue
                
            out.write(f"--- 2. ÚKOL: VÁZANÁ KYVADLA - PRUŽINA {spring} ---\n")
            w_dict, dw_dict = {}, {}
            
            for mode in range(1, 5):
                if mode in fixed_data[spring]:
                    n, t_list = fixed_data[spring][mode]
                    current_dt = dt_T4 if mode == 4 else dt_norm
                    w, dw, T, dT, t_mean, t_err = calc_omega(n, t_list, current_dt)
                    
                    w_dict[mode] = w
                    dw_dict[mode] = dw
                    
                    if mode == 4:
                        out.write(f"Mód 4 (Ts):\n  Perioda obálky T4 = {T:.3f} +/- {dT:.3f} s\n")
                    else:
                        out.write(f"Mód {mode} (T{mode}):\n  Perioda T{mode} = {T:.3f} +/- {dT:.3f} s\n")
                        
                    out.write(f"  Frekvence w{mode} = {w:.3f} +/- {dw:.3f} s^-1\n\n")
            
            if 1 in w_dict and 2 in w_dict:
                w3_teo = (w_dict[1] + w_dict[2]) / 2
                dw3_teo = 0.5 * np.sqrt(dw_dict[1]**2 + dw_dict[2]**2)
                
                w4_teo = (w_dict[2] - w_dict[1]) / 2
                dw4_teo = 0.5 * np.sqrt(dw_dict[2]**2 + dw_dict[1]**2)
                
                out.write("POROVNÁNÍ RÁZŮ (Teorie vs Měření):\n")
                if 3 in w_dict:
                    out.write(f"  w3 (teorie) = {w3_teo:.3f} +/- {dw3_teo:.3f} s^-1\n")
                    out.write(f"  w3 (exper.) = {w_dict[3]:.3f} +/- {dw_dict[3]:.3f} s^-1\n")
                if 4 in w_dict:
                    out.write(f"  w4 (teorie) = {w4_teo:.3f} +/- {dw4_teo:.3f} s^-1\n")
                    out.write(f"  w4 (exper.) = {w_dict[4]:.3f} +/- {dw_dict[4]:.3f} s^-1\n")
                
                kappa, dkappa = calc_kappa(w_dict[1], dw_dict[1], w_dict[2], dw_dict[2])
                out.write(f"\nSTUPEŇ VAZBY:\n  kappa = {kappa:.4f} +/- {dkappa:.4f}\n\n")
                out.write("-" * 57 + "\n\n")

        # --- ÚKOL 5: Závislost kappa na h ---
        if kappa_vs_h:
            out.write("=========================================================\n")
            out.write("5. ÚKOL: ZÁVISLOST STUPNĚ VAZBY KAPPA NA VZDÁLENOSTI H\n")
            out.write("=========================================================\n")
            out.write(f"{'h [m]':>8} | {'w1 [s^-1]':>12} | {'w2 [s^-1]':>12} | {'kappa':>10} | {'d_kappa':>10}\n")
            out.write("-" * 65 + "\n")
            
            h_vals, kappa_vals, dkappa_vals = [], [], []
            n_kmitu = 10 # Počet kmitů pro tuto sekci
            
            for row in kappa_vs_h:
                h, t1_list, t2_list = row
                w1, dw1, _, _, _, _ = calc_omega(n_kmitu, t1_list, dt_norm)
                w2, dw2, _, _, _, _ = calc_omega(n_kmitu, t2_list, dt_norm)
                k, dk = calc_kappa(w1, dw1, w2, dw2)
                
                h_vals.append(h)
                kappa_vals.append(k)
                dkappa_vals.append(dk)
                
                out.write(f"{h:8.3f} | {w1:12.3f} | {w2:12.3f} | {k:10.4f} | {dk:10.4f}\n")
                
            h_arr = np.array(h_vals)
            k_arr = np.array(kappa_vals)
            dk_arr = np.array(dkappa_vals)
            
            # FILTRACE DAT (Ideální polovina vs Celé kyvadlo)
            mask_ideal = h_arr <= 0.4
            h_ideal = h_arr[mask_ideal]
            k_ideal = k_arr[mask_ideal]
            dk_ideal = dk_arr[mask_ideal]
            
            # Regrese 1: Jen ideální data
            if len(h_ideal) > 0:
                p_ideal = np.polyfit(h_ideal, k_ideal, 2)
                out.write(f"\nRovnice regrese (h <= 0.4 m): kappa = {p_ideal[0]:.4f}*h^2 + {p_ideal[1]:.4f}*h + {p_ideal[2]:.4f}\n")
            else:
                p_ideal = [0, 0, 0]

            # Regrese 2: Všechna data
            p_full = np.polyfit(h_arr, k_arr, 2)
            out.write(f"Rovnice regrese (Celý rozsah): kappa = {p_full[0]:.4f}*h^2 + {p_full[1]:.4f}*h + {p_full[2]:.4f}\n")

            # GRAF 1: IDEÁLNÍ POLOVINA
            if len(h_ideal) > 0:
                plt.figure(figsize=(9, 6))
                # OPRAVA: Přidáno r před string pro LaTeX
                plt.errorbar(h_ideal, k_ideal, yerr=dk_ideal, fmt='ko', capsize=4, label=r'Naměřená data ($h \leq 0{,}4\,\mathrm{m}$)', zorder=5)
                h_fit_ideal = np.linspace(0, 0.45, 100)
                k_fit_ideal = np.polyval(p_ideal, h_fit_ideal)
                plt.plot(h_fit_ideal, k_fit_ideal, 'r-', label=f'Ideální kvadratická regrese')
                plt.xlabel(r'Vzdálenost pružiny od závěsu $h$/m', fontsize=12)
                plt.ylabel(r'Stupeň vazby $\kappa$', fontsize=12)
                plt.grid(True, linestyle=':', alpha=0.7)
                plt.legend()
                plt.xlim(left=0) 
                plt.ylim(bottom=0)
                plt.savefig(OUTPUT_PLOT_IDEAL, dpi=300, bbox_inches='tight')
                plt.close()
            
            # GRAF 2: KOMPLETNÍ ROZSAH A ODCHYLKA OD TEORIE
            plt.figure(figsize=(9, 6))
            plt.errorbar(h_arr, k_arr, yerr=dk_arr, fmt='ko', capsize=4, label='Všechna naměřená data', zorder=5)
            h_fit_full = np.linspace(0, max(h_arr)*1.05, 100)
            # Křivka přes všechna data
            plt.plot(h_fit_full, np.polyval(p_full, h_fit_full), 'b-', label='Proložení všech dat (neodpovídá teorii)')
            # Extrapolace ideální křivky (ukazuje odchylku)
            plt.plot(h_fit_full, np.polyval(p_ideal, h_fit_full), 'r--', linewidth=2, label='Extrapolace teoretického modelu z 1. poloviny')
            
            plt.xlabel(r'Vzdálenost pružiny od závěsu $h$/m', fontsize=12)
            plt.ylabel(r'Stupeň vazby $\kappa$', fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.legend()
            plt.xlim(left=0) 
            plt.ylim(bottom=0)
            plt.savefig(OUTPUT_PLOT_FULL, dpi=300, bbox_inches='tight')
            plt.close()

    print(f"HOTOVO! Vygenerovány dva grafy: '{OUTPUT_PLOT_IDEAL}' a '{OUTPUT_PLOT_FULL}'.")

if __name__ == "__main__":
    main()