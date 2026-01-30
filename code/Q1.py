import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import milp, LinearConstraint, Bounds

from load_data import load_data

# load datasets
df_launch, _, _ = load_data()

def model_for_launch(df_launch):
    # Extract Year
    # working on Date
    df_launch['Date'] = pd.to_datetime(df_launch['Date'], utc=True, errors='coerce')
    df_launch['Year'] = df_launch['Date'].dt.year
    df_launch = df_launch[df_launch['Year'] > 1991]

    # 10 Launch Locations
    target_keywords = [
        'Alaska', 'California', 'Texas', 'Florida', 'Virginia',  # USA States
        'Kazakhstan',  # Country
        'French Guiana',  # Region
        'Satish Dhawan Space Centre',  # India Specific
        'Taiyuan',  # China Specific (Taiyuan Satellite Launch Center)
        'Peninsula'  # New Zealand Specific (Mahia Peninsula)
    ]

    # check if kewords in Location
    def check_location(loc_str):
        for keyword in target_keywords:
            if keyword in str(loc_str):
                return True
        return False

    df_target = df_launch[df_launch['Location'].apply(check_location)].copy()

    print(f"Total launch: {len(df_target)} times")

    # Annual launch times (Grand Total for history)
    annual_launches = df_target.groupby('Year').size().reset_index(name='Count')

    print("Preview of the annual launches (Total)")
    print(annual_launches.tail())

    # Separate Models per Location
    class CombinedModel:
        def __init__(self, models):
            self.models = models
        
        def predict(self, X):
            # Sum up predictions from all individual location models
            all_preds = []
            for m in self.models:
                pred = m.predict(X)
                pred = np.maximum(pred, 0)
                all_preds.append(pred)
            
            all_preds_arr = np.array(all_preds)
            
            total_pred = np.sum(all_preds_arr, axis=0)
            # Use 70th percentile (top 30%) for max_pred and 30th percentile (bottom 30%) for min_pred
            max_pred = np.percentile(all_preds_arr, 70, axis=0)
            min_pred = np.percentile(all_preds_arr, 30, axis=0)
            
            return total_pred, max_pred, min_pred

    models = []
    print("\nTraining individual models for each location:")
    
    for keyword in target_keywords:
        # Filter data for this specific location
        # Using similar logic to check_location but specific to the keyword
        mask = df_launch['Location'].fillna('').astype(str).apply(lambda x: keyword in x)
        df_loc = df_launch[mask]
        
        if len(df_loc) < 2:
            print(f"  - Skipping {keyword}: Not enough data ({len(df_loc)})")
            continue
            
        annual_loc = df_loc.groupby('Year').size().reset_index(name='Count')
        
        X_loc = annual_loc[['Year']].values
        y_loc = annual_loc['Count'].values
        
        # Polynomial Regression Degree 2
        m = make_pipeline(PolynomialFeatures(2), LinearRegression())
        m.fit(X_loc, y_loc)
        models.append(m)
        print(f"  - Model trained for {keyword}")
    
    combined_model = CombinedModel(models)
    
    return combined_model, annual_launches

def predict_2050(model, annual_launches):
    # 预测 2050 年
    year_2050 = np.array([[2050]])
    fgi, fgmax, fgmin = model.predict(year_2050)
    print(f"Predicted Total Annual Launch Capacity (2050): {int(fgi[0])}")
    print(f"Predicted Max Annual Launch Capacity (2050): {int(fgmax[0])}")
    print(f"Predicted Min Annual Launch Capacity (2050): {int(fgmin[0])}")
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=annual_launches, x='Year', y='Count', label='Historical Data', color='blue')

    year_range = np.arange(annual_launches['Year'].min(), 2051).reshape(-1, 1)
    y_pred_line = model.predict(year_range)[0]
    y_max_line = model.predict(year_range)[1]
    y_min_line = model.predict(year_range)[2]

    plt.plot(year_range, y_pred_line, color='green', linestyle='--', linewidth=1, label='Trend Prediction')
    plt.plot(year_range, y_max_line, color='red', linestyle='--', linewidth=1, label='Trend Max')
    plt.plot(year_range, y_min_line, color='yellow', linestyle='--', linewidth=1, label='Trend Min')
    # 2050
    plt.scatter(2050, fgi, color='green', s=50, zorder=5,
                label=f'2050 Prediction: {int(fgi[0])}')
    # plt.scatter(2050, fgmax, color='orange', s=100, zorder=5,
    #             label=f'2050 Min Launch: {int(fgmin[0])}')

    plt.title('Annual Launches Trend for 10 Locations')
    plt.xlabel('Year')
    plt.ylabel('Number of Launches')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return int(fgi[0]), int(fgmax[0])

model, annual_launches = model_for_launch(df_launch)
predict_2050(model, annual_launches)

# --- 1. Parameters & Variables ---
M = 1e8  # 100 million tons

# Rockets (Scenario B)
MI = 150       # Payload per annual launch slot (Tons)
FGMAX_VAL = 61    # Max annual launches per site (Fixed Override as requested)
FGI = 61      #

# Costs & Reliability
C_RB = 1e6     # Fixed Cost per annual launch slot
C_RM = 5000    # Variable Cost per ton
eta = 1        # Success rate (assumed 1 for deterministic model)

# Space Elevator (Scenario A)
NUM_SE_PORTS = 3
Qeu = 179000   # Earth-Apex throughput (Tons/year)
Qed = 200000   # Apex-Moon throughput
Ce = 1000        # Unit Cost (per Ton)

b_factor = 0.05 # Fuel/Spares percentage

# Derived Calculations
# Rate of Space Elevator System
Rate_SE = NUM_SE_PORTS * min(Qeu, Qed)

# Cost Unit for SE (Scenario A & C)
# Ctot_a = Ce * M + Ce * (1-b) * M  => Unit Cost per Ton = Ce + Ce*(1-b)
C_unit_SE = Ce + Ce * (1 - b_factor)

# Rocket Capacities (Scenario B & C)
# Using fixed capacity as requested
FG_MAX_ARR = np.full(10, FGMAX_VAL)
MI_ARR = np.full(10, MI)

print(f"\n--- Parameters ---")
print(f"Target Mass M: {M/1e6} Million Tons")
print(f"SE Rate: {Rate_SE} Tons/Year; SE Unit Cost: {C_unit_SE}")
print(f"Rocket Max Rate: {np.sum(FG_MAX_ARR * MI_ARR)} Tons/Year (Max Launches: {np.sum(FG_MAX_ARR)})")

def calculate_scenario_metrics(fgi_array):
    # Rocket Rate
    rate_rocket = np.sum(fgi_array * MI_ARR)

    # Total Rate (SE + Rocket)
    rate_total = Rate_SE + rate_rocket

    if rate_total <= 0:
        return float('inf'), float('inf'), 0

    # Time (User Formula: Tc = M / (Rate_R + Rate_SE))
    Tc = M / rate_total

    # Cost Calculation
    # Ratio a = Mass_SE / Total_Mass = Rate_SE / Total_Rate
    if rate_total > 0:
        a = Rate_SE / rate_total
    else:
        a = 0

    # Cost Formula C (User provided):
    # Ctot = a * M * Cetot + (1-a) * M * Crm + Crb * sum(fgi)
    # Note: Cetot here corresponds to our C_unit_SE

    term1 = a * M * C_unit_SE
    term2 = (1 - a) * M * C_RM
    term3 = C_RB * np.sum(fgi_array)

    Ctot = term1 + term2 + term3

    return Tc, Ctot, a

def optimize_min_cost(max_time_years):
    # 1. Check Feasibility
    min_rate_needed = M / max_time_years
    rocket_rate_needed = min_rate_needed - Rate_SE

    if rocket_rate_needed <= 0:
        # SE is sufficient. No rockets needed.
        return np.zeros(10), 0

    # 2. Setup MILP
    # Variables: fgi (10 variables)
    # Objective Proxy: Minimize sum(fgi).
    # Rationale: Cost increases with fgi (Fixed Cost term + Variable Cost term since Rocket Unit Cost > SE Unit Cost).
    # So minimizing fgi minimizes Cost while satisfying rate.

    num_vars = 10
    c = np.ones(num_vars) # Minimize sum of launches

    # Constraints: sum(fgi * mi) >= rocket_rate_needed
    # LinearConstraint (A, lb, ub)
    lc = LinearConstraint(MI_ARR, lb=rocket_rate_needed, ub=np.inf)

    bounds = Bounds(lb=0, ub=FG_MAX_ARR)

    res = milp(c=c, constraints=lc, integrality=np.ones(num_vars), bounds=bounds)

    if res.success:
        return res.x, res.fun
    else:
        return None, None

# Run Baseline Solvers
# Scenario A: SE Only (fgi = 0)
tc_se, cost_se, a_se = calculate_scenario_metrics(np.zeros(10))
print(f"\nScenario A (SE Only): Time = {tc_se:.2f} y, Cost = {cost_se/1e9:.2f} B, a = {a_se:.2f}")

# Scenario B: Rockets Only (Manual calculation, assuming SE is disabled and Max Rockets used)
rate_rocket_max_total = np.sum(FG_MAX_ARR * MI_ARR)
if rate_rocket_max_total > 0:
    tc_b_print = M / rate_rocket_max_total
    # Cost (a=0): 1 * M * C_RM + Fixed Costs for Max Launches
    cost_b_print = M * C_RM + C_RB * np.sum(FG_MAX_ARR)
    print(f"Scenario B (Rockets Only): Time = {tc_b_print:.2f} y, Cost = {cost_b_print/1e9:.2f} B, a = 0.00")
else:
    print(f"Scenario B (Rockets Only): Infeasible (0 Capacity)")

# Scheme 1: MIP
target_time_limit = 180 # Example constraint
print(f"\nScheme 1: MIP (Time <= {target_time_limit} years)")
opt_fgi, _ = optimize_min_cost(target_time_limit)
if opt_fgi is not None:
    t_opt, c_opt, a_opt = calculate_scenario_metrics(opt_fgi)
    print(f"  Result: Time = {t_opt:.2f} y, Cost = {c_opt/1e9:.2f} B")
    print(f"  Parameter a (SE Share): {a_opt:.4f}")
    print(f"  Rocket Launches: {np.sum(opt_fgi)}")
else:
    print("  Infeasible under time constraint.")

# Pareto Front
print(f"\nScheme 2: Pareto Front Generation")
results = []
# Sweep total launches from 0 to Max
total_fg_limit = int(np.sum(FG_MAX_ARR))
step_size = max(1, int(total_fg_limit / 50)) 

for n_launch in range(0, total_fg_limit + 1, step_size):
    # Distribute n_launch uniformly/greedy
    curr_fgi = np.zeros(10)
    rem = n_launch
    for i, limit in enumerate(FG_MAX_ARR):
        take = min(rem, limit)
        curr_fgi[i] = take
        rem -= take
        if rem <= 0: break
    
    t, c, a_param = calculate_scenario_metrics(curr_fgi)
    results.append([t, c, a_param, n_launch])
    
df_pareto = pd.DataFrame(results, columns=['Tc', 'Ctot', 'a', 'Launches'])

# Normalized Objective Z
# Min Z = w * Norm(T) + (1-w) * Norm(C)
t_min, t_max = df_pareto['Tc'].min(), df_pareto['Tc'].max()
c_min, c_max = df_pareto['Ctot'].min(), df_pareto['Ctot'].max()

w = 0.5
df_pareto['Z'] = w * (df_pareto['Tc'] - t_min)/(t_max - t_min) + \
                 (1-w) * (df_pareto['Ctot'] - c_min)/(c_max - c_min)
                 
best_idx = df_pareto['Z'].idxmin()
best_sol = df_pareto.loc[best_idx]

print(f"\nPareto Optimal (w={w}):")
print(f"  Time: {best_sol['Tc']:.2f} y")
print(f"  Cost: {best_sol['Ctot']/1e9:.2f} B")
print(f"  Parameter a: {best_sol['a']:.4f}")

# Plot
plt.figure(figsize=(12, 7))

# 1. Pareto Points (Scenario C sweep)
# Filter for finite values to ensure plot stability
valid_mask = np.isfinite(df_pareto['Tc']) & np.isfinite(df_pareto['Ctot'])

valid_a = df_pareto.loc[valid_mask, 'a']
if not valid_a.empty:
    a_min, a_max = valid_a.min(), valid_a.max()
else:
    a_min, a_max = 0, 1

sc = plt.scatter(df_pareto.loc[valid_mask, 'Tc'], df_pareto.loc[valid_mask, 'Ctot'], 
                 c=df_pareto.loc[valid_mask, 'a'], 
                 cmap='viridis', vmin=a_min, vmax=a_max, alpha=0.6, s=15, label='Scenario C Sweep')
plt.colorbar(sc, label='Parameter a (SE Share)')

# 2. Optimal C Point
label_c = f"Scenario C Optimal (a={best_sol['a']:.2f}, {best_sol['Tc']:.1f}y, {best_sol['Ctot']/1e9:.1f}B)"
# Plot the optimal point using its 'a' value for color, edgecolors to make it stand out
plt.scatter(best_sol['Tc'], best_sol['Ctot'], c=best_sol['a'], cmap='viridis', s=150, marker='*', label=label_c, linewidths=1.5, vmin=a_min, vmax=a_max)

# Set Limits to Zoom In
# Align scaling with the visible Pareto line (Scenario C sweep data)
valid_tc = df_pareto.loc[valid_mask, 'Tc']
valid_cost = df_pareto.loc[valid_mask, 'Ctot']

if not valid_tc.empty:
    min_x, max_x = valid_tc.min(), valid_tc.max()
    min_y, max_y = valid_cost.min(), valid_cost.max()

    # Add small margin
    margin_x = (max_x - min_x) * 0.05
    margin_y = (max_y - min_y) * 0.05
    
    plt.xlim(min_x - margin_x, max_x + margin_x)
    plt.ylim(min_y - margin_y, max_y + margin_y)

plt.title('Pareto Front Optimization: Time vs Cost (Color=a)')
plt.xlabel('Time (Years)')
plt.ylabel('Total Cost')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
