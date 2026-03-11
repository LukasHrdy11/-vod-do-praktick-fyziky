import numpy as np

# 1. Načtení dat ze souboru (průměry v nm)
D = np.loadtxt('data.txt')

# 2. Přepočet na poloměry (zajímá nás poloměr částice r)
r = D / 2

# 3. Výpočet průměrného poloměru
r_mean = np.mean(r)

# 4. Výpočet nejistoty typu A
# Marco správně uvádí, že počítáme nejistotu výběru (výběrovou směrodatnou odchylku), 
# protože sledujeme "konkrétní kuličku vybranou ze vzorku", nikoliv ideální průměrnou kuličku.
# Parametr ddof=1 zajistí dělení (N-1) podle vzorce ze zadání.
sigma_A = np.std(r, ddof=1)

# 5. Zavedení nejistoty typu B
# Zahrnuje nepřesnost odečtu uživatele, rozmazání okrajů a krok programu (viz Marco.pdf).
sigma_B = 2.0  # v nm

# 6. Výpočet celkové nejistoty poloměru (kvadratický součet)
sigma_r = np.sqrt(sigma_A**2 + sigma_B**2)

# 7. Výpis výsledků
print("--- VÝSLEDKY MĚŘENÍ POLOMĚRU ---")
print(f"Počet měření (N): {len(r)}")
print(f"Průměrný poloměr (r): {r_mean:.1f} nm")
print(f"Výběrová nejistota typu A (sigma_A): {sigma_A:.1f} nm")
print(f"Odhadnutá nejistota typu B (sigma_B): {sigma_B:.1f} nm")
print(f"Celková nejistota (sigma_r): {sigma_r:.1f} nm")
print("-" * 32)
# Správně zaokrouhlený výsledek s odpovídajícím počtem platných číslic
print(f"Výsledek do protokolu: r = ({r_mean:.0f} +- {sigma_r:.0f}) nm")