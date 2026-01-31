import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Q3 import Q3

def plot_q3_figures():
    # 1. Run Simulation to get Data
    solver = Q3() # Use defaults or specific parameters
    
    # We want to show the convergence, so maybe start with low inventory to trigger dynamic response?
    solver.WS = 5e6 # Start low
    
    print("Running Simulation for Q3 Plots...")
    # Get detailed history from a run
    df_res = solver.run_simulation(start_year=2050, duration=50)
    
    # --- Figure 1: Dynamic Inventory & Dispatch Timeline ---
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # Primary Axis (Demand & Supply) - "Dispatch"
    # Wait, the user asked for "Monthly Demand W(t)" vs "Inventory WS(t)"
    # But our simulation is Yearly. Let's label Axis as "Annual Demand"
    
    color_dem = '#1f77b4' # Blue
    color_ws = '#ff7f0e' # Orange
    
    # Assume Demand W(t) is relatively smooth, but let's plot it
    # We might also want to show 'Supply' to prove I(t) > W(t)
    
    # Calculate Supply = Ws_next - Ws_prev + Demand
    # Or just use the 'a' to infer high supply?
    # Let's derive Supply roughly:
    df_res['Supply_Est'] = df_res['Ws'].diff().fillna(0) + df_res['Demand']
    # Fix first point roughly
    df_res.loc[0, 'Supply_Est'] = df_res.loc[0, 'Ws'] - solver.WS + df_res.loc[0, 'Demand']
    
    l1 = ax1.plot(df_res['Year'], df_res['Demand']/1e6, color=color_dem, linestyle='--', linewidth=2, label='Annual Demand $W(t)$')
    # Optional: Plot Supply Expected?
    l2 = ax1.fill_between(df_res['Year'], df_res['Demand']/1e6, df_res['Supply_Est']/1e6, color=color_dem, alpha=0.1, label='Surplus Supply')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Water Volume (Million Tons)', color=color_dem)
    ax1.tick_params(axis='y', labelcolor=color_dem)
    ax1.grid(True, alpha=0.3)
    
    # Secondary Axis (Inventory)
    ax2 = ax1.twinx()
    l3 = ax2.plot(df_res['Year'], df_res['Ws']/1e6, color=color_ws, linewidth=3, label='Inventory Level $WS(t)$')
    ax2.set_ylabel('Inventory $WS(t)$ (Million Tons)', color=color_ws)
    ax2.tick_params(axis='y', labelcolor=color_ws)
    
    # Combined Legend
    lines = l1 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    plt.title('Dynamic Inventory Balance & Dispatch Strategy (2050-2100)')
    plt.tight_layout()
    plt.savefig('Q3_Fig1_Dynamic_Inventory.png')
    print("Saved Q3_Fig1_Dynamic_Inventory.png")
    plt.close()

    # --- Figure 2: Convergence of Allocation Sequence A_t' ---
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    # Plot a_t
    # Use a scatter for points and a smooth line for trend
    sns.lineplot(data=df_res, x='Year', y='a', color='#2ca02c', linewidth=2, label='Optimal Allocation $a_t$', ax=ax)
    ax.scatter(df_res['Year'], df_res['a'], color='#2ca02c', s=30, alpha=0.6)
    
    # Static Optimal Baseline
    # Calculate static optimal for the final demand level?
    # Or just average of last 10 years as "Steady State"
    steady_state_a = df_res['a'].tail(10).mean()
    ax.axhline(steady_state_a, color='red', linestyle='--', linewidth=1.5, label=f'Steady State $a^* \\approx {steady_state_a:.2f}$')
    
    # Annotations
    ax.annotate('Initial Response\n(High SE Ratio)', 
                xy=(2050, df_res.iloc[0]['a']), xytext=(2055, df_res.iloc[0]['a'] + 0.2),
                arrowprops=dict(facecolor='black', shrink=0.05))
                
    ax.annotate('Steady State Convergence', 
                xy=(2090, steady_state_a), xytext=(2075, steady_state_a - 0.2),
                arrowprops=dict(facecolor='black', shrink=0.05))

    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel('Allocation Ratio ($a$)')
    ax.set_xlabel('Year')
    ax.set_title('Convergence of Space Elevator Allocation Sequence $A_t\'$')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('Q3_Fig2_Convergence.png')
    print("Saved Q3_Fig2_Convergence.png")
    plt.close()

if __name__ == "__main__":
    plot_q3_figures()
