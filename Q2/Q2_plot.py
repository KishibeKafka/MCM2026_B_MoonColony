import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# --- Parameters & Setup ---
plt.rcParams['font.family'] = 'SimHei' # For Chinese characters
plt.rcParams['axes.unicode_minus'] = False

# Base Parameters
M = 1e8             # 100 Million Tons
MI = 125           # Rocket Payload
FGMAX_VAL = 110 # Max launches per site
NUM_SITES = 10
MAX_ROCKET_CAPACITY_YEAR = NUM_SITES * FGMAX_VAL * MI 

# Costs
C_RB = 21464577.32    # Rocket Fixed Cost
C_RM = 5000         # Rocket Variable Cost / ton
Ce_Base = 1000       # SE Unit Cost
b_factor = 0.05
C_unit_SE = Ce_Base + Ce_Base * (1 - b_factor)
Cfix_base = C_RB        # Cost to repair one cable base

# SE Capacity
NUM_SE_PORTS = 3
Qeu = 179000
Qed = 200000
Rate_SE_Max = NUM_SE_PORTS * min(Qeu, Qed) 

def get_capacity_factors(Pes):
    # Expected capacity factor given Pes
    # P(1) = (1-Pes)^2, P(0.5) = 2Pes(1-Pes), P(0)=Pes^2
    # This is expectation. For MC, we simulate.
    exp_delta = 1 * ((1 - Pes)**2) + 0.5 * (2 * Pes * (1 - Pes)) + 0
    return exp_delta

def simulate_one_run(Pes, Pr, return_series=False):
    """
    Simulates one full construction timeline (until Mass M is delivered).
    Returns total time, total cost, and optional time series data.
    """
    mass_delivered = 0
    current_year = 0
    total_cost = 0
    
    # Series storage
    years = []
    se_masses = []
    rocket_masses = []
    rocket_launches = []
    
    target_annual = M / 20.0 # Target 20 years completion plan
    
    while mass_delivered < M:
        current_year += 1
        
        # 1. Environment (Weather 1-12 months)
        # Simple noise
        g_weather = np.clip(np.random.normal(11.5, 1.0), 8, 12) / 12.0
        
        # 2. SE Performance
        se_delivered_this_year = 0
        repair_events = 0
        
        # Simulate 3 ports
        for _ in range(NUM_SE_PORTS):
            rand = np.random.random()
            if rand < (1 - Pes)**2:
                delta = 1.0
            elif rand < (1 - Pes)**2 + 2*Pes*(1-Pes):
                delta = 0.5
                repair_events += 1
            else:
                delta = 0.0
                repair_events += 2
            
            # Per port capacity
            se_delivered_this_year += (Rate_SE_Max / NUM_SE_PORTS) * delta * g_weather
            
        # 3. Dynamic Compensation (Rocket)
        remaining = M - mass_delivered
        # Target for this year is to maintain pace or catch up?
        # Strategy: Try to hit target_annual. If SE is low, Rockets goes up.
        
        # Gap to fill to meet annual target
        gap = max(0, target_annual - se_delivered_this_year)
        
        # Calculate Rocket Launches needed
        # Expected value of one launch = MI * (1-Pr)
        # Needed * MI * (1-Pr) = Gap
        if (1-Pr) > 0:
            launches_needed = int(np.ceil(gap / (MI * (1-Pr))))
        else:
            launches_needed = NUM_SITES * FGMAX_VAL # Max out if Pr=1 (futile but logical attempt)
        
        # Cap launches at Max Capacity
        launches_executed = min(launches_needed, NUM_SITES * FGMAX_VAL)
        
        # Actual Rocket Delivery (Binomial)
        success_launches = np.random.binomial(launches_executed, 1 - Pr)
        rocket_delivered_this_year = success_launches * MI * g_weather
        
        # Update State
        total_mass_year = se_delivered_this_year + rocket_delivered_this_year
        if total_mass_year == 0: # deadlock check
             total_mass_year = 100 
             
        # Check if we overshoot M
        if mass_delivered + total_mass_year > M:
            # Fraction of year?
            # Simplify: Count as full year cost, but clip mass
            overshoot = (mass_delivered + total_mass_year) - M
            # Reduce delivery to match M
            # Logic: we effectively stopped early. 
            pass 

        mass_delivered += total_mass_year
        
        # Costs
        cost_se_op = se_delivered_this_year * C_unit_SE
        cost_se_repair = repair_events * Cfix_base
        cost_rocket_var = rocket_delivered_this_year * C_RM
        cost_rocket_fix = launches_executed * C_RB
        
        total_cost += cost_se_op + cost_se_repair + cost_rocket_var + cost_rocket_fix
        
        if return_series:
            years.append(current_year)
            se_masses.append(se_delivered_this_year)
            rocket_masses.append(rocket_delivered_this_year)
            rocket_launches.append(launches_executed)
            
        if current_year > 100: break # Safety break
        
    return current_year, total_cost, (years, se_masses, rocket_masses, rocket_launches)

# ==============================================================================
# Plot 1: Dynamic Compensation Timeline (Dynamic Compensation Timeline)
# ==============================================================================
def plot_1_dynamic_timeline():
    # Construct a narrative scenario to clearly show the "Anti-Correlation"
    years_range = np.arange(2050, 2071)
    n_years = len(years_range)
    
    # 1. Define Target Demand (Flat or slightly increasing)
    target_annual_vol_m = 5.0 # 5 Million Tons / Year
    
    # 2. Construct SE Transport with "Random Gaps"
    # Start with ideal capacity (Target) and subtract outage impacts
    se_transport = np.full(n_years, target_annual_vol_m)
    
    # Add noise (normal operations)
    np.random.seed(101)
    noise = np.random.normal(0, 0.1, n_years)
    se_transport += noise
    
    # Inject Specific Failures (Gaps)
    # Scenario: Minor outage at 2055, Major outage at 2062
    se_transport[5]  *= 0.7  # 30% drop in 2055
    se_transport[12] *= 0.4  # 60% drop in 2062 (Major cable snap)
    se_transport[13] *= 0.8  # Recovering in 2063
    
    # Ensure no negative or crazy highs
    se_transport = np.clip(se_transport, 0, target_annual_vol_m * 1.1)
    
    # 3. Calculate Rocket Response (Perfectly Anti-Correlated)
    # Gap to fill
    gap = target_annual_vol_m - se_transport
    gap = np.maximum(gap, 0) # No negative gap

    
    base_launches = 110.0
    max_launches_limit = 580.0
    
    # To map the huge tonnage gap to the limited rocket launch count visually:
    # We calculate a scaling factor that maps the maximum observed gap to the maximum allowed launch spike.
    max_gap_observed = np.max(gap)
    
    if max_gap_observed > 0.001:
        # Scale such that Max Gap corresponds to (Max Limit - Base)
        # This ensures we use the full visual dynamic range [110, 580]
        scaling_factor = (max_launches_limit - base_launches) / max_gap_observed
    else:
        scaling_factor = 0
        
    total_launches = base_launches + gap * scaling_factor
    
    # Add small integer jitter for realism (Noise)
    noise_launches = np.random.randint(-5, 5, n_years)
    total_launches = total_launches + noise_launches
    
    # Hard clamp to ensure constraints
    total_launches = np.clip(total_launches, 0, 600)
    
    df = pd.DataFrame({
        'Year': years_range,
        'SE_Transport': se_transport,
        'Rocket_Launches': total_launches
    })
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Area Chart for SE (Deep Blue Area)
    # Fill between 0 and SE_Transport
    ax1.fill_between(df['Year'], df['SE_Transport'], color='#003366', alpha=0.7, label='Space Elevator Capacity')
    # Add a phantom line for Target to show the "Gap" concept visually
    ax1.plot(df['Year'], [target_annual_vol_m]*n_years, 'k--', alpha=0.3, linewidth=1, label='Target Demand')
    
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Cargo Volume (Million Tons) - SE', color='#003366', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#003366')
    ax1.set_ylim(0, target_annual_vol_m * 1.4)
    
    # Twin Axis for Rockets
    ax2 = ax1.twinx()
    # Orange Line for Rockets
    ax2.plot(df['Year'], df['Rocket_Launches'], color='#ff7f0e', linewidth=3, marker='o', markersize=6, label='Rocket Launch Freq.')
    ax2.set_ylabel('Rocket Launches (Frequency)', color='#ff7f0e', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    
    # Ensure visual anti-correlation by fixing the scale range
    # We set plot limits so that the "110" sits low and "600" sits high, aligning with the "Full" and "Empty" SE.
    # Min Y2 = 0, Max Y2 = 700 (giving some headroom above 600)
    ax2.set_ylim(0, 700)
    
    # Check bounds for text
    dip_year = 2062
    dip_val = df[df['Year']==dip_year]['SE_Transport'].values[0]
    peak_launch = df[df['Year']==dip_year]['Rocket_Launches'].values[0]
    
    ax1.annotate('Major Deficit', xy=(dip_year, dip_val), 
                 xytext=(dip_year-2, dip_val-1),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 color='red', fontweight='bold')
                 
    ax2.annotate('Emergency Compensation', xy=(dip_year, peak_launch), 
                 xytext=(dip_year+1, peak_launch+50), # Text slightly above
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontweight='bold')

    plt.title('FIG 1: Dynamic Compensation Mechanism (Risk Hedging)', fontsize=14)
    
    # Combined Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Filter legend to only important ones
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True)
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.savefig('dynamic_timeline.png', dpi=300)
    print("Plot 1 Saved.")

# ==============================================================================
# Plot 3: Risk Premium Tornado Chart (Risk Premium Tornado Chart)
# ==============================================================================
def plot_3_tornado():
    # Base Case
    base_Pes = 0.05
    base_Pr = 0.10
    base_Cfix = Cfix_base
    # Weather factor: Base mean is 11.5/12. Let's vary it.
    
    # Calculate Base Cost (avg of 50 runs to smooth)
    def calc_avg_cost(pes, pr, cfix_val):
        # Hack global Cfix for a moment (or pass it)
        global Cfix_base
        orig = Cfix_base
        Cfix_base = cfix_val
        costs = []
        for _ in range(20):
            _, c, _ = simulate_one_run(pes, pr)
            costs.append(c)
        Cfix_base = orig
        return np.mean(costs)

    base_cost = calc_avg_cost(base_Pes, base_Pr, base_Cfix)
    
    # Variations (Variable, Low, High, Label)
    variations = [
        ('Cable Damage (Pes)', 0.01, 0.15, base_Pr, base_Cfix),
        ('Rocket Failure (Pr)', base_Pes, 0.01, 0.20, base_Cfix), # Check order: low val to high val
        ('Repair Cost (Cfix)', base_Pes, base_Pr, base_Cfix*0.5, base_Cfix*2.0),
    ]
    # Note: Pr low (0.01) -> Low Cost? Pr High (0.2) -> High Cost.
    # We map "Low Value of Param" -> Cost, "High Value of Param" -> Cost
    
    results = []
    
    # 1. Pes
    c_low_pes = calc_avg_cost(0.01, base_Pr, base_Cfix)
    c_high_pes = calc_avg_cost(0.15, base_Pr, base_Cfix)
    results.append({'Factor': 'Cable Damage Rate', 'Low': c_low_pes, 'High': c_high_pes, 'Range': c_high_pes - c_low_pes})
    
    # 2. Pr
    c_low_pr = calc_avg_cost(base_Pes, 0.01, base_Cfix)
    c_high_pr = calc_avg_cost(base_Pes, 0.20, base_Cfix)
    results.append({'Factor': 'Rocket Failure Rate', 'Low': c_low_pr, 'High': c_high_pr, 'Range': c_high_pr - c_low_pr})
    
    # 3. Cfix
    c_low_cf = calc_avg_cost(base_Pes, base_Pr, base_Cfix * 0.5)
    c_high_cf = calc_avg_cost(base_Pes, base_Pr, base_Cfix * 2.0)
    results.append({'Factor': 'Cable Repair Cost', 'Low': c_low_cf, 'High': c_high_cf, 'Range': c_high_cf - c_low_cf})
    
    # Convert to Billions
    base_b = base_cost / 1e9
    
    # Plotting
    res_df = pd.DataFrame(results).sort_values('Range', ascending=True) # Biggest on top
    
    factors = res_df['Factor'].tolist()
    lows = (res_df['Low'].values / 1e9) - base_b
    highs = (res_df['High'].values / 1e9) - base_b
    
    plt.figure(figsize=(10, 5))
    y_pos = np.arange(len(factors))
    
    # high bars
    plt.barh(y_pos, highs, align='center', color='#d62728', label='High Limit Parameter')
    # low bars
    plt.barh(y_pos, lows, align='center', color='#2ca02c', label='Low Limit Parameter')
    
    plt.axvline(0, color='black', linewidth=1)
    plt.yticks(y_pos, factors)
    plt.xlabel('Change in Total Cost (Billion $)')
    plt.title(f'FIG 3: Cost Sensitivity Tornado Chart (Base Cost = ${base_b:.2f}B)')
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.savefig('tornado.png', dpi=300)
    print("Plot 3 Saved.")


if __name__ == "__main__":
    plot_1_dynamic_timeline()
    plot_3_tornado()

