import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Q3 import Q3

# --- Trace Class for SA Visualization ---
class Q3_Trace(Q3):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trace_history = []  # Store steps: {iter, T, a, E, accepted}

    # Override the SA method to capture history
    def simulated_annealing(self, W_demand, current_ws):
        # Clear history for this run
        self.trace_history = []
        
        # Init
        a_curr = 0.5
        T = self.T_start
        
        # Baseline
        tc, cc, _ = self.calculate_mc_vectorized(0.8, W_demand, 10)
        c_base = np.mean(cc)
        t_base = np.mean(tc)
        if t_base < 1e-6: t_base = 1.0
        
        E_curr = self.energy_function(a_curr, W_demand, current_ws, c_base, t_base)
        best_a = a_curr
        
        iteration = 0
        
        # Capture Initial State
        self.trace_history.append({
            'iter': iteration,
            'T': T,
            'a': a_curr,
            'E': E_curr,
            'accepted': True,
            'current_a': a_curr,
            'current_E': E_curr
        })

        while T > self.T_end:
            # We record every step
            for _ in range(5):  # Inner loop
                iteration += 1
                sigma = 0.1 if T > 50 else 0.01
                a_new = np.clip(a_curr + np.random.normal(0, sigma), 0.0, 1.0)
                
                E_new = self.energy_function(a_new, W_demand, current_ws, c_base, t_base)
                
                dE = E_new - E_curr
                accepted = False
                if dE < 0 or np.random.rand() < np.exp(-dE / T):
                    a_curr = a_new
                    E_curr = E_new
                    accepted = True
                    if E_curr < 1e8: # Only track valid solutions
                        best_a = a_new
                
                # Record Step
                self.trace_history.append({
                    'iter': iteration,
                    'T': T,
                    'a': a_new,      # The proposed a
                    'E': E_new,      # The proposed E
                    'accepted': accepted,
                    'current_a': a_curr, # The state after decision
                    'current_E': E_curr
                })
            
            T *= self.gamma
            
        return best_a

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
    
    color_dem = '#1f77b4' # Blue
    color_ws = '#ff7f0e' # Orange
    
    # Calculate Supply roughly
    df_res['Supply_Est'] = df_res['Ws'].diff().fillna(0) + df_res['Demand']
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

    # --- Figure 3: SA Analysis (New) by request ---
    print("Running SA Trace for Fig 3...")
    tracer = Q3_Trace()
    tracer.WS = 5e6
    W_demand_sample = tracer.predict_demand(2050)
    tracer.simulated_annealing(W_demand_sample, tracer.WS)
    
    df_trace = pd.DataFrame(tracer.trace_history)
    df_valid = df_trace[df_trace['E'] < 1e8].copy()
    df_accepted = df_trace[df_trace['accepted'] == True]
    E_max_plot = df_valid['E'].max() * 1.1 if not df_valid.empty else 100

    fig3 = plt.figure(figsize=(14, 10))
    gs = fig3.add_gridspec(2, 2)
    
    # Subplot 1: Energy Convergence
    ax1 = fig3.add_subplot(gs[0, :])
    ax1.plot(df_accepted['iter'], df_accepted['E'], color='#1f77b4', linewidth=1, label='Accepted Energy E(a)', alpha=0.8)
    ax1b = ax1.twinx()
    ax1b.plot(df_trace['iter'], df_trace['T'], color='red', linestyle=':', label='Temperature T', alpha=0.5)
    ax1b.set_ylabel('Temperature (Log Scale)', color='red')
    ax1b.set_yscale('log')
    
    ax1.set_title('SA Energy Convergence & Cooling Schedule')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Energy Function E(a)')
    ax1.set_ylim(0, E_max_plot)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Trajectory (a vs Energy)
    ax2 = fig3.add_subplot(gs[1, 0])
    sc = ax2.scatter(df_valid['a'], df_valid['E'], c=df_valid['iter'], cmap='viridis', s=20, alpha=0.3, label='Explored States')
    plt.colorbar(sc, ax=ax2, label='Iteration')
    
    ax2.plot(df_accepted['current_a'], df_accepted['current_E'], color='red', linewidth=1.5, alpha=0.8, label='Optimization Path')
    # Start/End
    if not df_accepted.empty:
        ax2.scatter(df_accepted.iloc[0]['current_a'], df_accepted.iloc[0]['current_E'], color='green', marker='^', s=100, label='Start', zorder=5)
        ax2.scatter(df_accepted.iloc[-1]['current_a'], df_accepted.iloc[-1]['current_E'], color='black', marker='*', s=150, label='Final', zorder=5)
    
    ax2.set_title('Optimization Trajectory (Search Space)')
    ax2.set_xlabel('Allocation Ratio a')
    ax2.set_ylabel('Energy')
    ax2.set_ylim(0, E_max_plot)
    ax2.set_xlim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Acceptance Distribution
    ax3 = fig3.add_subplot(gs[1, 1])
    ax3.scatter(df_trace.loc[~df_trace['accepted'], 'a'], df_trace.loc[~df_trace['accepted'], 'T'], 
                color='red', alpha=0.1, s=10, label='Rejected')
    ax3.scatter(df_trace.loc[df_trace['accepted'], 'a'], df_trace.loc[df_trace['accepted'], 'T'], 
                color='green', alpha=0.5, s=10, label='Accepted')
    
    ax3.set_title('Acceptance vs Temperature')
    ax3.set_xlabel('Allocation Ratio a')
    ax3.set_ylabel('Temperature (Log Scale)')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Q3_Fig3_SA_Analysis.png')
    print("Saved Q3_Fig3_SA_Analysis.png")
    plt.close()

if __name__ == "__main__":
    plot_q3_figures()
