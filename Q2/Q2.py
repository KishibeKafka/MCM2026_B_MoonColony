import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Q1.Q1 import Q1

class Q2(Q1):
    def __init__(self, FGI, C_RB):
        super().__init__(FGI, 21464577.32)
        print(f"DEBUG: Q2 Init - Qse={self.Qse}, Qrocket={self.Qrocket}")
        self.Cfix = C_RB  # Cost to repair one cable

    # Model Functions
    def get_capacity_factors(self, Pes):
        """
        假设是两条缆绳是独立损坏的
        Probabilities:
          P(delta=0)   = Pes^2
          P(delta=0.5) = 2Pes - 2Pes^2
          P(delta=1)   = (1 - Pes)^2
        """    
        exp_delta = 0 * (Pes**2) + 0.5 * (2 * Pes - 2 * Pes**2) + 1 * ((1 - Pes)**2)
        return exp_delta

    def calculate_scenario_metrics(self, alpha, Pes, Pr, n_weather_months):
        """
        Calculates Time and Cost under imperfect conditions.
        Overrides Q1 method to account for reliability and repair costs.
        """
        # Calculate dynamic rocket rate based on fgi_array input
        # Note: self.Qse is base annual capacity.
        # Q2 Logic: Capacity is affected by Reliability and Weather
        
        gamma = n_weather_months / 12.0
        
        # SE Throughput (Expected)
        # Q_elev = Rate_SE * gamma * E[delta]
        exp_delta = self.get_capacity_factors(Pes)
        
        Qe = self.Qse * gamma * exp_delta * (1 - Pes) # t/year
        Qr = self.Qrocket * gamma * (1 - Pr) # t/year

        t_se = (alpha * self.M) / Qe if Qe > 0 else float('inf')
        t_rocket = ((1 - alpha) * self.M) / Qr if Qr > 0 else float('inf')
        # If alpha is 0, t_se is 0 (not inf). If alpha is 1, t_rocket is 0.
        if alpha == 0: t_se = 0
        if alpha == 1: t_rocket = 0
        T = max(t_se, t_rocket)

        c_rocket_var = ((1 - alpha) * self.M) * self.C_RM / (1 - Pr)
        
        c_rocket_fix = self.C_RB * ((1 - alpha) * self.M) / (self.MI * (1 - Pr))

        c_se_op = alpha * self.M * self.Cse
        
        num_cables = self.NUM_SE_PORTS * 2
        c_se_repair = self.Cfix * T * num_cables * Pes
        
        C_total = c_rocket_var + c_rocket_fix + c_se_op + c_se_repair
        
        return T, C_total

# Analysis
# --- 3. Sensitivity Analysis (Heatmap) ---
# Analysis
# --- 3. Sensitivity Analysis (Heatmap) ---
def sensitivity_analysis(steps=20):
    print("\nStarting Sensitivity Analysis...")
    pes_values = np.linspace(0.001, 0.2, steps)
    pr_values = np.linspace(0.001, 0.2, steps)
    
    normal_weather_months = 12
    # Initialize Solver
    FGMAX_VAL = 110 
    C_RB_VAL = 21464577.32 
    solver = Q2(FGI=FGMAX_VAL, C_RB=C_RB_VAL)
    
    # Store results
    cost_sensitivity = np.zeros((steps, steps))
    time_sensitivity = np.zeros((steps, steps))

    for i, pes in enumerate(pes_values):
        for j, pr in enumerate(pr_values):
            # Calculate metrics
            gamma = normal_weather_months / 12.0
            exp_delta = solver.get_capacity_factors(pes)
            
            # Max available throughputs
            Qe_max = solver.Qse * gamma * exp_delta * (1 - pes)
            Qr_max = solver.Qrocket * gamma * (1 - pr)
            
            total_rate = Qe_max + Qr_max
            if total_rate <= 0:
                t_curr, c_curr = float('inf'), float('inf')
            else:
                # Alpha is naturally the share of SE in this max-effort scenario
                effective_alpha = Qe_max / total_rate
                t_curr, c_curr = solver.calculate_scenario_metrics(effective_alpha, pes, pr, normal_weather_months)
            
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
    
    # Initialize Solver for parameters
    FGMAX_VAL = 110 
    C_RB_VAL = 21464577.32 
    solver = Q2(FGI=FGMAX_VAL, C_RB=C_RB_VAL)
    
    sim_times = []
    sim_costs = []
    
    # Parameters for distributions
    mean_weather_months = 12
    std_weather_months = 1
    
    for sim_id in range(num_simulations):
        mass_delivered = 0
        current_year = 0
        total_cost = 0
        
        while mass_delivered < solver.M:
            current_year += 1
            
            # 1. Environment Generation
            # Weather (Normal distribution clipped 1-12)
            g_weather = np.clip(np.random.normal(mean_weather_months, std_weather_months), 1, 12) / 12.0
            
            # Failure Probabilities (Stochastic per year)
            pr_sample = np.random.beta(2, 20) # Mean ~0.1
            pes_sample = np.random.beta(2, 38) # Mean ~0.05
            
            # 2. SE Capacity (Stochastic)
            se_delivered_this_year = 0
            repair_events = 0
            
            # Assume 3 Ports
            for port in range(solver.NUM_SE_PORTS):
                rand_val = np.random.random()
                p_full = (1 - pes_sample)**2
                p_half = 2 * pes_sample * (1 - pes_sample)
                
                if rand_val < p_full:
                    delta = 1.0 
                elif rand_val < p_full + p_half:
                    delta = 0.5
                    repair_events += 1 
                else:
                    delta = 0.0
                    repair_events += 2 
                
                # Single Port Capacity 
                port_cap = (solver.Qse / solver.NUM_SE_PORTS) * delta * g_weather
                se_delivered_this_year += port_cap
            
            # 3. Strategy Response (Fill Gap)
            remaining_mass = solver.M - mass_delivered
            
            target_annual = solver.M / 25.0 
            gap = max(0, target_annual - se_delivered_this_year)
            
            # Rocket Capacity Calculation
            max_launches_total = solver.NUM_LAUNCH_LOCATIONS * solver.FGI
            
            if (1 - pr_sample) > 0:
                launches_needed = int(np.ceil(gap / (solver.MI * (1 - pr_sample))))
            else:
                launches_needed = max_launches_total
                
            launches_actual = min(launches_needed, max_launches_total)
            
            # Actual Rocket Delivery (Binomial)
            success_launches = np.random.binomial(launches_actual, 1 - pr_sample)
            rocket_delivered_this_year = success_launches * solver.MI * g_weather 
            
            # Update State
            mass_delivered += se_delivered_this_year + rocket_delivered_this_year
            
            # 4. Cost Calculation
            # Costs using solver attributes
            cost_se_ops = se_delivered_this_year * solver.Cse
            cost_se_fix = repair_events * solver.Cfix 
            cost_rocket_var = rocket_delivered_this_year * solver.C_RM
            cost_rocket_fix = launches_actual * solver.C_RB
            
            total_cost += cost_se_ops + cost_se_fix + cost_rocket_var + cost_rocket_fix
            
            if current_year > 1000: break
                
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

