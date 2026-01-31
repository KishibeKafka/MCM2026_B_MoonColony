import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Import Q1 class and helper functions
from Q1 import Q1, get_parameters, optimize_scheme_2

# --- Graph Settings ---
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("white", {'font.sans-serif': ['SimHei', 'Arial']})

# -----------------------------------------------------------
# 1. Resource Allocation Surface (Dual-Axis Plot)
# -----------------------------------------------------------
def plot_resource_allocation(solver):
    print("Generating Resource Allocation Plot...")
    alpha_values = np.linspace(0, 1, 500)
    times = []
    costs = []
    
    for a in alpha_values:
        t, c = solver.calculate_scenario_metrics(a)
        times.append(t)
        costs.append(c)
        
    times = np.array(times)
    costs = np.array(costs)
    costs_b = costs / 1e9  # Billion USD
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Time (Red)
    color_t = 'tab:red'
    ax1.set_xlabel('Proportion of Cargo by Space Elevator ($a$)', fontsize=12)
    ax1.set_ylabel('Time to Completion (Years)', color=color_t, fontsize=12)
    ax1.plot(alpha_values, times, color=color_t, linewidth=2.5, label='Time')
    ax1.tick_params(axis='y', labelcolor=color_t)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Plot Cost (Blue)
    ax2 = ax1.twinx() 
    color_c = 'tab:blue'
    ax2.set_ylabel('Total Cost (Billion USD)', color=color_c, fontsize=12)
    ax2.plot(alpha_values, costs_b, color=color_c, linewidth=2.5, linestyle='-', label='Cost')
    ax2.tick_params(axis='y', labelcolor=color_c)
    
    # Highlight Optimal intersection roughly (visual guide)
    # The actual optimal depends on weights, but visual intersection is clear trade-off
    
    plt.title('Trade-off Analysis: Time vs Cost by Space Elevator Ratio ($a$)', fontsize=14)
    fig.tight_layout()
    plt.savefig('resource_allocation.png')
    plt.show()

# -----------------------------------------------------------
# 2. Sensitivity Analysis (Radar Chart)
# -----------------------------------------------------------
def plot_sensitivity_radar(FGI_base, Price_base):
    print("Generating Sensitivity Radar Chart...")
    
    # Baseline
    solver_base = Q1(FGI=FGI_base, C_RB=Price_base)
    # _, _, _, cost_base = optimize_scheme_2(solver_base, 0.5, 0.5)
    # Use explicit optimal finder from Scheme 1 logic or Scheme 2 (should be similar for min cost)
    # Or simply calculate cost of 'Current Optimal' configuration? 
    # Let's re-optimize for fair comparison
    _, _, _, cost_base = optimize_scheme_2(solver_base, 0.5, 0.5)

    
    # Parameters to test: Ce, MI, b, FGI
    # We will test +50% improvement (Cost reduction or Tech increase)
    # "Improvement" directions:
    # Ce (Cost SE): -50%
    # MI (Payload): +50%
    # b (Maint): -50% (from 0.05 to 0.025)
    # FGI (Freq): +50%
    
    scenarios = {
        'SE Unit Cost ($C_e$)':   {'param': 'Ce',    'change': -0.5}, 
        'Rocket Payload ($m_i$)': {'param': 'MI',    'change': +0.5},
        'Maintenance ($b$)':      {'param': 'b_factor', 'change': -0.5},
        'Launch Freq ($fg_{max}$)': {'param': 'FGI',   'change': +0.5}
    }
    
    impacts = []
    labels = []
    
    for name, conf in scenarios.items():
        # Create temp solver with modified param
        # Need to manually poke the Q1 class since params are processed in __init__
        # Strategy: Re-instantiate Q1 with hacked inputs or modify attributes + recalc
        
        # New Values - Initialize with BASE
        fgi_new = FGI_base
        solver_new = Q1(FGI=FGI_base, C_RB=Price_base)
        
        # Apply Modification
        if conf['param'] == 'FGI':
             # Re-init purely because FGI affects Qrocket in __init__
            fgi_new = FGI_base * (1 + conf['change'])
            solver_new = Q1(FGI=fgi_new, C_RB=Price_base)

        elif conf['param'] == 'Ce':
            solver_new.Ce = solver_base.Ce * (1 + conf['change'])
            # Recalc dependent
            solver_new.Cse = solver_new.Ce + solver_new.Ce * (1 - solver_new.b_factor)
            
        elif conf['param'] == 'MI':
            solver_new.MI = solver_base.MI * (1 + conf['change'])
            # Recalc dependent Qrocket = FGI * MI * Locs
            solver_new.Qrocket = solver_new.FGI * solver_new.MI * solver_new.NUM_LAUNCH_LOCATIONS
            # Also affects Cost? Term3 = ... * C_RB / MI. Yes.
            
        elif conf['param'] == 'b_factor':
            solver_new.b_factor = solver_base.b_factor * (1 + conf['change'])
            # Recalc dependent
            solver_new.Cse = solver_new.Ce + solver_new.Ce * (1 - solver_new.b_factor)
            
        # Get new optimized cost
        _, _, _, cost_new = optimize_scheme_2(solver_new, 0.5, 0.5)
        
        # Calculate % Change in Cost (Expected negative for improvement)
        pct_change = (cost_new - cost_base) / cost_base * 100
        # We plot magnitude of savings? Or pure change?
        # Let's plot "Sensitivity Score" = Abs(% Change in Total Cost) per 20% change
        impacts.append(abs(pct_change))
        labels.append(name)
        print(f"  {name}: {pct_change:.2f}% Cost Change")

    # Radar Plot
    # Close the loop
    values = impacts + [impacts[0]]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += [angles[0]]  # Close loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable + labels
    plt.xticks(angles[:-1], labels, size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    # plt.yticks([5, 10, 15, 20], ["5%", "10%", "15%", "20%"], color="grey", size=10)
    # plt.ylim(0, max(values)*1.1)
    
    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#1f77b4')
    ax.fill(angles, values, '#1f77b4', alpha=0.25)
    
    plt.title('Sensitivity Analysis: Cost Reduction Potential\n(Impact of 50% Technical Improvement)', y=1.08, fontsize=15, fontweight='bold')
    
    # Add annotation explaining the metric
    plt.figtext(0.5, 0.02, "Values represent % reduction in Total Cost given a 50% improvement in the parameter.\nLarger area indicates higher sensitivity.", 
                ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout()
    plt.savefig('Q1_Sensitivity_Radar.png')
    plt.show()

# -----------------------------------------------------------
# 3. Cumulative Delivery Progress (Stacked Area + Lines)
# -----------------------------------------------------------
def plot_cumulative_progress(solver, opt_a):
    print("Generating Cumulative Progress Plot...")
    
    years = np.linspace(0, 50, 200) # 50 Year horizon
    M = solver.M / 1e6 # Convert to Million Tons
    
    # Scenario A (SE Only, a=1)
    # Rate: Qse
    rate_a = solver.Qse / 1e6
    y_a = np.minimum(M, rate_a * years)
    
    # Scenario B (Rocket Only, a=0)
    # Rate: Qrocket
    rate_b = solver.Qrocket / 1e6
    y_b = np.minimum(M, rate_b * years)
    
    # Scenario C (Optimal Hybrid)
    # SE Part: Cap = a*M, Rate = Qse
    # Rocket Part: Cap = (1-a)*M, Rate = Qrocket
    # They run strictly parallel from t=0. 
    # Stop when their specific quota is done.
    
    cap_se_c = opt_a * M
    cap_rk_c = (1 - opt_a) * M
    
    y_c_se = np.minimum(cap_se_c, rate_a * years)
    y_c_rk = np.minimum(cap_rk_c, rate_b * years)
    y_c_total = y_c_se + y_c_rk
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Comparison Lines (Scenario A & B)
    ax.plot(years, y_a, 'g-.', linewidth=2, label='Scenario A (SE Only)', alpha=0.8)
    ax.plot(years, y_b, 'b-.', linewidth=2, label='Scenario B (Rocket Only)', alpha=0.8)
    
    # 2. Stacked Area for Scenario C
    ax.stackplot(years, y_c_se, y_c_rk, 
                 labels=['Scenario C (SE Contribution)', 'Scenario C (Rocket Contribution)'],
                 colors=['#2ca02c', '#ff7f0e'], alpha=0.4)
                 
    # 3. Total Progress Line for C
    ax.plot(years, y_c_total, 'k-', linewidth=2, label='Scenario C (Total)')
    
    # 4. Target Line
    ax.axhline(y=M, color='red', linestyle=':', linewidth=2, label='Target (100 MT)')
    
    # 5. Intersects
    # Find when C hits M
    # Ideally: T_c = max( (aM)/Qse, ((1-a)M)/Qrocket )
    # Let's verify plotting matches calc
    t_complete = np.argmax(y_c_total >= M * 0.999) # Approx index
    if t_complete > 0:
        t_val = years[t_complete]
        ax.axvline(x=t_val, color='k', linestyle='--', alpha=0.5)
        ax.text(t_val, M*1.02, f'{t_val:.1f} Y', ha='center', fontweight='bold')

    ax.set_title('Cumulative Delivery Progress Comparison (Scenario C)', fontsize=14)
    ax.set_xlabel('Years', fontsize=12)
    ax.set_ylabel('Cumulative Cargo (Million Tons)', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 40) # Limit x axis for clarity
    
    plt.tight_layout()
    plt.savefig('progress.png')
    plt.show()

def generate_refactored_plots():
    print("Initializing...")
    launch_max, price_2050 = get_parameters()
    
    # Base Solver
    solver = Q1(FGI=launch_max, C_RB=price_2050)
    
    # Get optimal for Base
    _, opt_a, _, _ = optimize_scheme_2(solver, 0.5, 0.5)
    print(f"Optimal Alpha Base: {opt_a:.4f}")
    
    # Plot 1
    plot_resource_allocation(solver)
    
    # Plot 2
    plot_sensitivity_radar(launch_max, price_2050)
    
    # Plot 3
    plot_cumulative_progress(solver, opt_a)

if __name__ == "__main__":
    generate_refactored_plots()
