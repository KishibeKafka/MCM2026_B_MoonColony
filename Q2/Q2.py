import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Base Parameters (From Q1) 
M = 1e8             # 100 Million Tons
MI = 125           # Rocket Payload
FGMAX_VAL = 110 # Max launches per site
NUM_SITES = 10

# Costs (Base)
C_RB = 21464577.32    # Rocket Fixed Cost
C_RM = 5000         # Rocket Variable Cost / ton
Ce_Base = 1000       # SE Unit Cost
b_factor = 0.05
# Effective SE Unit Cost (Operational)
C_unit_SE = Ce_Base + Ce_Base * (1 - b_factor)

# SE Capacity Base
NUM_SE_PORTS = 3
Qeu = 179000
Qed = 200000
Rate_SE = NUM_SE_PORTS * min(Qeu, Qed)

# Q2 Specific Parameters (Estimates)
Cfix = C_RB        # Cost to repair one cable

# Model Functions
def get_capacity_factors(Pes):
    """
    假设是两条缆绳是独立损坏的
    Probabilities:
      P(delta=0)   = Pes^2
      P(delta=0.5) = 2Pes - 2Pes^2
      P(delta=1)   = (1 - Pes)^2
    """    
    exp_delta = 0 * (Pes**2) + 0.5 * (2 * Pes - 2 * Pes**2) + 1 * ((1 - Pes)**2)
    return exp_delta

MI_ARR = np.full(NUM_SITES, MI)

def calculate_q2_metrics(fgi_array, Pes, Pr, n_weather_months):
    """
    Calculates Time and Cost under imperfect conditions.
    """
    rate_rocket = np.sum(fgi_array * MI_ARR)
    a = Rate_SE / (Rate_SE + rate_rocket)

    gamma = n_weather_months / 12.0
    
    # Time Calculation
    # Qe = sigam（3,1) [ Qeu * delat * gama-weather * (1 - Pe) ]
    exp_delta = get_capacity_factors(Pes)
    Q_elev = Rate_SE * gamma * exp_delta
    
    # Annual Rocket Throughput
    # User formula: sigma(10,1)[fgi * mi * gamma] * (1 - Pr)
    max_rocket_launches = NUM_SITES * FGMAX_VAL
    Q_rocket = max_rocket_launches * MI * gamma * (1 - Pr)
    
    total_rate = Q_elev + Q_rocket
    
    if total_rate <= 0:
        return float('inf'), float('inf')
        
    # Time to transport M
    T = M / total_rate
    
    # Cost Calculation
    c_rocket_var = (1 - a) * M * C_RM / (1 - Pr)
    
    c_rocket_fix = C_RB * ((1 - a) * M) / (MI * (1 - Pr))
    
    c_se_op = a * M * C_unit_SE # same as Q1
    
    # Cfix = T * 6 * Pes
    c_se_repair = Cfix * T * 6 * Pes
    
    C_total = c_rocket_var + c_rocket_fix + c_se_op + c_se_repair
    
    return T, C_total, a

# Analysis
# --- 3. Sensitivity Analysis (Heatmap) ---
def sensitivity_analysis(steps=20):
    print("\nStarting Sensitivity Analysis...")
    pes_values = np.linspace(0.001, 0.2, steps)
    pr_values = np.linspace(0.001, 0.2, steps)
    
    normal_weather_months = 12
    fgi_curr = np.full(NUM_SITES, FGMAX_VAL)  # Max rocket usage for sensitivity
    # Store results
    cost_sensitivity = np.zeros((steps, steps))
    time_sensitivity = np.zeros((steps, steps))

    for i, pes in enumerate(pes_values):
        for j, pr in enumerate(pr_values):
            # Calculate metrics directly using fixed strategy (Full Capacity)
            t_curr, c_curr, _ = calculate_q2_metrics(fgi_curr, pes, pr, normal_weather_months)
            
            cost_sensitivity[i, j] = c_curr
            time_sensitivity[i, j] = t_curr

    # --- Analysis Output ---
    print("-" * 30)
    print("Sensitivity Analysis Results:")
    print(f"Range assessed: Pr [{np.min(pr_values):.3f}, {np.max(pr_values):.3f}], Pes [{np.min(pes_values):.3f}, {np.max(pes_values):.3f}]")
    
    max_cost_idx = np.unravel_index(np.argmax(cost_sensitivity, axis=None), cost_sensitivity.shape)
    min_cost_idx = np.unravel_index(np.argmin(cost_sensitivity, axis=None), cost_sensitivity.shape)
    
    print(f"Cost Range: ${np.min(cost_sensitivity)/1e9:.2f}B - ${np.max(cost_sensitivity)/1e9:.2f}B")
    print(f"  Lowest Cost at Pr={pr_values[min_cost_idx[1]]:.3f}, Pes={pes_values[min_cost_idx[0]]:.3f}")
    print(f"  Highest Cost at Pr={pr_values[max_cost_idx[1]]:.3f}, Pes={pes_values[max_cost_idx[0]]:.3f}")
    
    max_time_idx = np.unravel_index(np.argmax(time_sensitivity, axis=None), time_sensitivity.shape)
    min_time_idx = np.unravel_index(np.argmin(time_sensitivity, axis=None), time_sensitivity.shape)
    
    print(f"Time Range: {np.min(time_sensitivity):.1f} Years - {np.max(time_sensitivity):.1f} Years")
    print(f"  Fastest Time at Pr={pr_values[min_time_idx[1]]:.3f}, Pes={pes_values[min_time_idx[0]]:.3f}")
    print(f"  Slowest Time at Pr={pr_values[max_time_idx[1]]:.3f}, Pes={pes_values[max_time_idx[0]]:.3f}")
    
    # Simple Sensitivity Grade (Avg change per step)
    grad_cost_pes = np.mean(np.diff(cost_sensitivity, axis=0)) # Change along Pes
    grad_cost_pr = np.mean(np.diff(cost_sensitivity, axis=1)) # Change along Pr
    print(f"Avg Cost Sensitivity to Pes: ${grad_cost_pes/1e9:.3f}B per step")
    print(f"Avg Cost Sensitivity to Pr: ${grad_cost_pr/1e9:.3f}B per step")
    print("-" * 30)

    # Plot Heatmaps (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Cost Heatmap
    p1 = sns.heatmap(cost_sensitivity / 1e9, ax=axes[0], 
                     xticklabels=np.round(pr_values, 2), 
                     yticklabels=np.round(pes_values, 2), cmap='viridis_r') # reversed: dark=high cost
    axes[0].set_title('Cost (Billion $)')
    axes[0].set_xlabel('Rocket Failure Rate (Pr)')
    axes[0].set_ylabel('Cable Damage Rate (Pes)')
    axes[0].invert_yaxis()
    
    # 2. Time Heatmap
    p2 = sns.heatmap(time_sensitivity, ax=axes[1], 
                     xticklabels=np.round(pr_values, 2), 
                     yticklabels=np.round(pes_values, 2), cmap='magma_r')
    axes[1].set_title('Time to Complete (Years)')
    axes[1].set_xlabel('Rocket Failure Rate (Pr)')
    axes[1].set_yticks([]) # Hide y ticks for cleaner look
    axes[1].invert_yaxis()
    
    # Reduce ticks frequency for readability
    # Handle X ticks for all axes
    for ax in axes:
        xticks = ax.get_xticks()
        if len(xticks) > 0:
            ax.set_xticks(xticks[::4])
            ax.set_xticklabels(np.round(pr_values[::4], 2))
    
    # Handle Y ticks ONLY for the first axis (others are hidden)
    yticks = axes[0].get_yticks()
    if len(yticks) > 0:
        axes[0].set_yticks(yticks[::4])
        axes[0].set_yticklabels(np.round(pes_values[::4], 2))
    
    axes[1].set_yticks([])

    plt.tight_layout()
    plt.savefig('heatmap.png')
    plt.show()

# --- 4. Monte Carlo Simulation (Year-by-Year Iteration) ---
def monte_carlo_simulation(num_simulations=1000):
    print(f"\nStarting Monte Carlo Simulation ({num_simulations} runs)...")
    
    sim_times = []
    sim_costs = []
    
    # Parameters for distributions
    mean_weather_months = 12
    std_weather_months = 1
    
    # Base strategy: Aim for Scenario C optimal ratio initially, but adjust dynamically
    # For simulation, we assume unlimited demand to finish M as fast as possible or cost-effective?
    # Logic: Dynamic filling of gap.
    
    for sim_id in range(num_simulations):
        mass_delivered = 0
        current_year = 0
        total_cost = 0
        
        while mass_delivered < M:
            current_year += 1
            
            # 1. Environment Generation
            # Weather (Normal distribution clipped 1-12)
            g_weather = np.clip(np.random.normal(mean_weather_months, std_weather_months), 1, 12) / 12.0
            
            # Rocket Failure Probability sample for this year
            pr_sample = np.random.beta(2, 20) # Mean ~0.1
            
            # 3 SE Ports Status (Independent)
            # Pes sample for this year/cable
            pes_sample = np.random.beta(2, 38) # Mean ~0.05
            
            # Calculate SE Capacity
            # Each port has 2 cables? Assuming abstract capacity factor
            # exp_delta = get_capacity_factors(pes_sample) 
            # Or simulate discrete events for 3 ports:
            se_delivered_this_year = 0
            repair_events = 0
            
            for port in range(NUM_SE_PORTS):
                # Roll for cables. Assume 2 cables per port?
                # Simplified: Port efficiency delta 
                # P(normal) = (1-pes)^2, P(0.5) = ..., P(0) = ...
                rand_val = np.random.random()
                p_full = (1 - pes_sample)**2
                p_half = 2 * pes_sample * (1 - pes_sample)
                # p_zero = pes_sample**2
                
                if rand_val < p_full:
                    delta = 1.0
                elif rand_val < p_full + p_half:
                    delta = 0.5
                    repair_events += 1 # 1 cable fix
                else:
                    delta = 0.0
                    repair_events += 2 # 2 cables fix (or 1 major?)
                
                # Single Port Capacity = Rate_SE_Base / 3
                port_cap = (Rate_SE / NUM_SE_PORTS) * delta * g_weather
                se_delivered_this_year += port_cap
            
            # 2. Strategy Response (Fill Gap)
            remaining_mass = M - mass_delivered
            
            # Decide Rockets
            # "If gap exists, increase rocket frequency"
            # Here gap is basically "remaining mass". 
            # We want to use max rockets if SE is down? Or just enough?
            # User logic: "Rocket base increase launch frequency to fill gap"
            # This implies a target annual rate. Let's assume target is M / 25 years (Base plan).
            target_annual = M / 25.0 
            gap = max(0, target_annual - se_delivered_this_year)
            
            # Rocket Capacity Calculation
            # Launches needed = gap / (MI * (1 - pr_sample))
            # But limited by FGMAX
            max_launches_total = NUM_SITES * FGMAX_VAL
            
            launches_needed = int(np.ceil(gap / (MI * (1 - pr_sample)))) if (1 - pr_sample) > 0 else max_launches_total
            launches_actual = min(launches_needed, max_launches_total)
            
            # Rocket Delivery
            # Binomial for success? Or expected value?
            # Monte Carlo suggests Binomial for actual successes
            success_launches = np.random.binomial(launches_actual, 1 - pr_sample)
            rocket_delivered_this_year = success_launches * MI * g_weather # Weather affects rocket too
            
            # Total Mass
            total_delivered_year = se_delivered_this_year + rocket_delivered_this_year
            mass_delivered += total_delivered_year
            
            # 3. Cost Calculation
            # SE Ops (Fixed per year or per ton?)
            # Q1 formula: a * M * Ce. implies per ton.
            cost_se_ops = se_delivered_this_year * C_unit_SE
            
            # SE Repair
            cost_se_fix = repair_events * Cfix # Assumed Cfix is per cable/event
            
            # Rocket Variable
            cost_rocket_var = rocket_delivered_this_year * C_RM
            
            # Rocket Fixed
            cost_rocket_fix = launches_actual * C_RB
            
            total_cost += cost_se_ops + cost_se_fix + cost_rocket_var + cost_rocket_fix
            
            # Fail-safe for infinite loop
            if current_year > 1000:
                break
                
        sim_times.append(current_year)
        sim_costs.append(total_cost)

    # --- Analysis Output ---
    times_arr = np.array(sim_times)
    costs_arr = np.array(sim_costs)
    costs_b = costs_arr / 1e9
    
    print("-" * 30)
    print(f"Monte Carlo Analysis Results ({num_simulations} runs):")
    
    print("Completion Time (Years):")
    print(f"  Mean: {np.mean(times_arr):.2f}")
    print(f"  Median: {np.median(times_arr):.2f}")
    print(f"  Std Dev: {np.std(times_arr):.2f}")
    print(f"  95% CI: [{np.percentile(times_arr, 2.5):.2f}, {np.percentile(times_arr, 97.5):.2f}]")
    print(f"  Range: {np.min(times_arr)} - {np.max(times_arr)}")
    
    print("Total Cost (Billion $):")
    print(f"  Mean: ${np.mean(costs_b):.2f}B")
    print(f"  Median: ${np.median(costs_b):.2f}B")
    print(f"  Std Dev: ${np.std(costs_b):.2f}B")
    print(f"  95% CI: [${np.percentile(costs_b, 2.5):.2f}B, ${np.percentile(costs_b, 97.5):.2f}B]")
    print("-" * 30)

    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(sim_times, kde=True, color='skyblue')
    plt.title('Distribution of Completion Time (Years)')
    plt.xlabel('Years')
    
    plt.subplot(1, 2, 2)
    sns.histplot(np.array(sim_costs)/1e9, kde=True, color='salmon')
    plt.title('Distribution of Total Cost (Billion $)')
    plt.xlabel('Cost (B)')
    
    plt.tight_layout()
    plt.savefig('monte_carlo_results.png')
    plt.show()

sensitivity_analysis()
monte_carlo_simulation()

