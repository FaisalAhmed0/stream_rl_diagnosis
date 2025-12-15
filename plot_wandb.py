import wandb
import pandas as pd
import matplotlib.pyplot as plt

api = wandb.Api()

plt.rcParams.update({'font.size': 20})

## Minigrid 16x16
# run_paths = [
#     "streaming-x-diagnosis/Stream_AC_Minigrid/o6ih2q69",
#     "streaming-x-diagnosis/Stream_AC_Minigrid/4pnj2mzg",
#     "streaming-x-diagnosis/Stream_Q_Minigrid_2/x012idia",
#     "streaming-x-diagnosis/Stream_Q_Minigrid_2/1tr243h9"
# ]

## Memory S7
run_paths = [
    "streaming-x-diagnosis/Stream_Q_Minigrid/92zy7aie",
    "streaming-x-diagnosis/Stream_Q_Minigrid_2/wh7coixs",
    "streaming-x-diagnosis/Stream_AC_Minigrid/bs3l2y0h",
    "streaming-x-diagnosis/Stream_AC_Minigrid/ln8oxzyo"
]

## Memory S13
# run_paths = [
#     "streaming-x-diagnosis/Stream_AC_Minigrid/ngmoin0k",    
# ]


# --- PARAMÈTRE A MODIFIER ---
WINDOW_SIZE = 100  # Moyenne sur les 100 derniers points
# ----------------------------

plt.figure(figsize=(12, 6))
print(f"Génération du graphique avec une moyenne glissante de {WINDOW_SIZE}...")

for run_p in run_paths:
    try:
        run = api.run(run_p)
        
        # 1. Récupération
        history = run.history(keys=['_step', 'training/episodic_return'], samples=1000000)
        
        # 2. Nettoyage des NaNs
        history = history.dropna(subset=['training/episodic_return'])
        
        if not history.empty:
            history = history.sort_values('_step')
            
            # 3. Calcul de la Rolling Mean
            # On crée une nouvelle colonne pour la moyenne
            history['rolling_return'] = history['training/episodic_return'].rolling(window=WINDOW_SIZE, min_periods=1).mean()
            
            # 4. Tracé
            # A. On trace d'abord la donnée brute en très transparent (alpha=0.2)
            # On récupère l'objet 'p' pour réutiliser la même couleur pour la moyenne
            p = plt.plot(history['_step'], history['training/episodic_return'], alpha=0.2)
            color = p[0].get_color() # On capture la couleur attribuée
            
            # B. On trace la moyenne glissante avec la même couleur, mais plus épaisse
            if "Q" in run.name :
                stop = 8
            else :
                stop = 9
            if "0.8" in run.name:
                plt.plot(history['_step'], history['rolling_return'], 
                        label=f"{run.name[:stop]}(0.8)", 
                        color=color, 
                        linewidth=2)
            else:
                plt.plot(history['_step'], history['rolling_return'], 
                        label=f"{run.name[:stop]}(0.0)", 
                        color=color, 
                        linewidth=2)
            
            print(f"-> Ajouté : {run.name}")
        else:
            print(f"-> Pas de données pour {run.name}")
            
    except Exception as e:
        print(f"Erreur sur {run_p}: {e}")

plt.xlabel("Timesteps")
# plt.ylabel("Episodic Return")
plt.xlim((0, 1_000_000))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)

output_filename = "comparaison_runs_rolling2.png"
plt.savefig(output_filename, dpi=300)
print(f"\nGraphique enregistré sous : {output_filename}")

plt.close()