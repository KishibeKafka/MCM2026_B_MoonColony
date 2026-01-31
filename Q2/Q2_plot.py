import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Q2 import Q2

# --- Graph Settings ---
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid", {'font.sans-serif': ['SimHei', 'Arial']})

# 风险溢价龙卷风图 (Risk Premium Tornado Chart)

def plot_risk_tornado(solver):
    print("Generating Risk Premium Tornado Chart...")

    # --- 1. Define Cost Calculation Logic ---
    def calc_cost(pes, pr, weather_months, cfix_val):
        # Effective params
        gamma = weather_months / 12.0
        # Expected effective capacity factor due to cable probability
        # E[efficiency] approx 1 - pes
        exp_delta = 1.0 - pes
        
        # Effective rates
        rate_se = solver.Qse * exp_delta * gamma
        rate_rocket = solver.Qrocket * (1 - pr) * gamma
        
        # Total effective rate
        rate_total = rate_se + rate_rocket
        
        # Years needed
        if rate_total <= 0: return float('inf')
        T = solver.M / rate_total
        
        # Costs
        # Split Volume proportional to capacity
        vol_se = solver.M * (rate_se / rate_total)
        vol_rk = solver.M * (rate_rocket / rate_total)
        
        # Cost SE
        # Ops + Repairs
        # Expectation of repairs: T * 12 * pes events? 
        # Using simplified heuristic: Cost ~ Ops + Repair_Risk
        c_se_ops = vol_se * solver.Cse
        c_se_repair = T * 12 * pes * cfix_val
        
        # Cost Rocket
        # Ops (including failures requiring extra launches)
        c_rk_ops = vol_rk * solver.C_RM / (1-pr)
        
        return (c_se_ops + c_se_repair + c_rk_ops)

    # --- 2. Base Case ---
    base_pes = 0.05
    base_pr = 0.10
    base_weath = 10.0 # Standard weather
    base_cfix = solver.Cfix
    
    C_base = calc_cost(base_pes, base_pr, base_weath, base_cfix)
    
    # --- 3. Sensitivity Parameters ---
    # Define Low/High values. 
    params = [
        {
            'label': 'Cable Repair Cost', 
            'param': 'cfix',
            'base': base_cfix,
            'low_val': base_cfix * 0.5, 
            'high_val': base_cfix * 2.0,
            'func': lambda x: calc_cost(base_pes, base_pr, base_weath, x)
        },
        {
            'label': 'Cable Damage Rate', 
            'param': 'pes',
            'base': base_pes,
            'low_val': 0.01, 
            'high_val': 0.15,
            'func': lambda x: calc_cost(x, base_pr, base_weath, base_cfix)
        },
        {
            'label': 'Rocket Failure Rate', 
            'param': 'pr',
            'base': base_pr,
            'low_val': 0.05, 
            'high_val': 0.20,
            'func': lambda x: calc_cost(base_pes, x, base_weath, base_cfix)
        }
    ]
    
    plot_data = []
    for p in params:
        cost_low = p['func'](p['low_val'])
        cost_high = p['func'](p['high_val'])
        
        # Delta from base
        delta_low = (cost_low - C_base) / 1e9 # Billions
        delta_high = (cost_high - C_base) / 1e9
        
        plot_data.append({
            'label': p['label'],
            'high_change': delta_high, # Change when param is High
            'low_change': delta_low,   # Change when param is Low
            'width': abs(delta_high - delta_low)
        })
        
    # Sort by total width (Smallest at top to match typical Tornado or specific request)
    plot_data.sort(key=lambda x: x['width'], reverse=False)

    labels = [x['label'] for x in plot_data]
    high_changes = [x['high_change'] for x in plot_data]
    low_changes = [x['low_change'] for x in plot_data]
    
    # --- 4. Plotting ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    y_pos = np.arange(len(labels))
    bar_height = 0.6
    
    # Colors: Red for High Parameter, Green for Low Parameter
    color_high = '#d62728' # Red
    color_low = '#2ca02c'  # Green
    
    # Plot bars
    ax.barh(y_pos, high_changes, height=bar_height, color=color_high, label='High Limit Parameter', align='center', alpha=0.9, edgecolor='black', linewidth=0.5)
    ax.barh(y_pos, low_changes, height=bar_height, color=color_low, label='Low Limit Parameter', align='center', alpha=0.9, edgecolor='black', linewidth=0.5)
    
    # Vertical line at 0
    ax.axvline(0, color='black', linewidth=1.2)
    
    # Grid
    ax.grid(axis='x', linestyle='-', alpha=0.3, color='gray')
    ax.grid(axis='y', linestyle='--', alpha=0.1) 
    
    # Formatting X Axis
    ax.set_xlabel('Change in Total Cost (Billion $)', fontsize=12, weight='bold')
    
    # Formatting Y Axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12, weight='bold')
    
    # Title
    base_cost_b = C_base / 1e9
    ax.set_title(f'FIG 3: Cost Sensitivity Tornado Chart (Base Cost = ${base_cost_b:.2f}B)', fontsize=14, weight='bold', pad=15)
    
    # Legend
    ax.legend(loc='upper right', frameon=True, fontsize=10, facecolor='white', edgecolor='gray')
    
    plt.tight_layout()
    plt.savefig('tornado.png', dpi=300)
    plt.show()

def generate_q2_plots():
    l_max, p_2050 = 593, 21464577.32
    
    # Init Solver
    solver = Q2(FGI=l_max, C_RB=p_2050)

    plot_risk_tornado(solver)

if __name__ == "__main__":
    generate_q2_plots()