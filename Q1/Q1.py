import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import milp, LinearConstraint, Bounds

from Q1.load_data import load_data

# load datasets
df_launch, _, _ = load_data()

def model_for_launch(df_launch):
    # 10 Launch Locations
    target_keywords = [
        'Alaska', 'CA', 'TX', 'FL', 'Virginia',  # USA States
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
            max_pred = all_preds_arr.max(axis=0)
            
            return total_pred, max_pred

    # Piecewise Linear Model Class (Breakpoint 2020)
    class PiecewiseLinearRegression:
        def __init__(self, breakpoint=2020):
            self.breakpoint = breakpoint
            self.model = LinearRegression()
            
        def _transform(self, X):
            # Feature 1: x
            # Feature 2: max(0, x - breakpoint)
            # This creates a hinge function that allows slope change at breakpoint
            X_transformed = np.hstack([X, np.maximum(0, X - self.breakpoint)])
            return X_transformed

        def fit(self, X, y):
            X_new = self._transform(X)
            self.model.fit(X_new, y)

        def predict(self, X):
            X_new = self._transform(X)
            return self.model.predict(X_new)

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
        
        # Piecewise Linear Regression (Breakpoint 2020)
        m = PiecewiseLinearRegression(breakpoint=2020)
        m.fit(X_loc, y_loc)
        models.append(m)
        print(f"  - Model trained for {keyword} (Piecewise Linear)")
    
    combined_model = CombinedModel(models)
    
    return combined_model, annual_launches

def predict_2050_fgi(model, annual_launches):
    # 预测 2050 年
    year_2050 = np.array([[2050]])
    fgi, fgmax = model.predict(year_2050)
    print(f"Predicted Total Annual Launch Capacity (2050): {int(fgi[0])}")
    print(f"Max Annual Launch Capacity in Locations: {int(fgmax[0])}")
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=annual_launches, x='Year', y='Count', label='Historical Data', color='blue')

    year_range = np.arange(annual_launches['Year'].min(), 2051).reshape(-1, 1)
    y_pred_line = model.predict(year_range)[0]
    y_max_line = model.predict(year_range)[1]

    plt.plot(year_range, y_pred_line, color='green', linestyle='--', linewidth=1, label='Trend Total')
    plt.plot(year_range, y_max_line, color='red', linestyle='--', linewidth=1, label='Trend Max')
    # 2050
    plt.scatter(2050, fgi, color='green', s=50, zorder=5,
                label=f'2050 Prediction: {int(fgi[0])}')

    plt.title('Annual Launches Trend for 10 Locations')
    plt.xlabel('Year')
    plt.ylabel('Number of Launches')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('launch_times.png')
    plt.show()
    print(f"Predicted total launch times is {round(fgi[0])}, max launch times is {round(fgmax[0])}")
    return round(fgi[0]), round(fgmax[0])

def model_for_price(df_launch):
    # Filter out entries with missing Price
    df_price = df_launch.dropna(subset=['Price']).copy()
    # Clean Price column: remove '$' and ',' and convert to float
    df_price['Price'] = df_price['Price'].astype(str).str.replace(r'[$,]', '', regex=True)
    df_price['Price'] = pd.to_numeric(df_price['Price'], errors='coerce')
    df_price = df_price.dropna(subset=['Price'])
    # Calculate average price per year
    annual_avg_price = df_price.groupby('Year')['Price'].mean().reset_index()
    # Log transform
    annual_avg_price['Log Price'] = np.log(annual_avg_price['Price'])
    X = annual_avg_price['Year'].values.reshape(-1, 1)
    y = annual_avg_price['Log Price'].values
    # Train Regression Model (Linear on Log Price)
    model = LinearRegression()
    model.fit(X, y)
    return model, annual_avg_price

def predict_price_2050(model, annual_avg_price):
    year_2050 = np.array([[2050]])
    log_price_2050 = model.predict(year_2050)
    price_2050 = np.exp(log_price_2050)[0]
    print(f"Predicted Average Launch Price (2050): ${price_2050:,.2f}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=annual_avg_price, x='Year', y='Price', label='Historical Data', color='blue')
    year_range = np.arange(annual_avg_price['Year'].min(), 2051).reshape(-1, 1)
    log_pred_line = model.predict(year_range)
    y_pred_line = np.exp(log_pred_line)
    plt.plot(year_range, y_pred_line, color='green', linestyle='--', linewidth=1, label='Trend Prediction')
    plt.scatter(2050, price_2050, color='green', s=50, zorder=5,
                label=f'2050 Prediction: ${price_2050:,.2f}')
    plt.title('Average Launch Price Trend')
    plt.xlabel('Year')
    plt.ylabel('Average Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('launch_price.png')
    plt.show()
    return price_2050

def get_parameters():
    m, al = model_for_launch(df_launch)
    l_tot, l_max = predict_2050_fgi(m, al)
    pm, ap = model_for_price(df_launch)
    p_2050 = predict_price_2050(pm, ap)
    return l_max, p_2050

class Q1:
    def __init__(self, FGI, C_RB):
        self.NUM_SE_PORTS = 3
        self.NUM_LAUNCH_LOCATIONS = 10
        self.M = 1e8
        self.MI = 125
        self.FGI = FGI
        self.C_RB = C_RB
        self.C_RM = 5000    # Variable Cost per ton (Assumed cargo integration cost)
        self.NUM_SE_PORTS = 3
        self.Qeu = 179000   # Earth-Apex throughput (Tons/year)
        self.Qed = 200000   # Apex-Moon throughput
        self.Ce = 1000        # Unit Cost (per Ton) - Reasonable estimate for mature SE
        self.b_factor = 0.05 # Fuel/Spares percentage

        self.Qse = self.NUM_SE_PORTS * min(self.Qeu, self.Qed) # t/year
        self.Cse = self.Ce + self.Ce * (1 - self.b_factor)

        self.Qrocket = self.FGI * self.MI * self.NUM_LAUNCH_LOCATIONS # t/year
        print(f"\n--- Parameters ---")
        print(f"Target Mass M: {self.M/1e6} Million Tons")
        print(f"SE Capacity: {self.Qse:,.0f} t/year")
        print(f"Rocket Capacity (Total): {self.Qrocket:,.0f} t/year")

    def calculate_scenario_metrics(self, alpha):
        # alpha * M for SE and (1-alpha) * M for Rockets
        # Time = max {a * M/Qse, (1-a)*M/Qrocket}
        # Handle edge cases for 0 capacity if needed, but Qse/Qrocket > 0 typically
        t_se = (alpha * self.M) / self.Qse if self.Qse > 0 else float('inf')
        t_rocket = ((1 - alpha) * self.M) / self.Qrocket if self.Qrocket > 0 else float('inf')
        
        # If alpha is 0, t_se is 0 (not inf). If alpha is 1, t_rocket is 0.
        if alpha == 0: t_se = 0
        if alpha == 1: t_rocket = 0
            
        T = max(t_se, t_rocket)
        
        # Cost Formula
        term1 = alpha * self.M * self.Cse
        term2 = (1 - alpha) * self.M * self.C_RM
        term3 = (1 - alpha) * self.M * self.C_RB / self.MI
        C = term1 + term2 + term3
        return T, C

# Scheme 1: Mixed Integer Programming (Approximation via Linear Constraints)
def optimize_scheme_1(solver, t_max):
    # Constants
    M = solver.M
    Q1 = solver.Qse
    Q2 = solver.Qrocket
    alpha_values = np.linspace(0, 1, 1001)
    results = []
    
    for val_a in alpha_values:
        t_val, c_val = solver.calculate_scenario_metrics(val_a)
        if (t_val <= t_max):
            results.append({
                'a': val_a,
                'T': t_val,
                'C': c_val
            })

    df = pd.DataFrame(results)
    if df.empty:
        return None, None, None, "Infeasible: No solutions meet the time constraint"
    best_idx = df['C'].idxmin()
    best_sol = df.loc[best_idx]
    
    return best_sol['a'], best_sol['T'], best_sol['C'], "Optimal"

# Scheme 2: Pareto Front with Normalization
def optimize_scheme_2(solver, w1=0.5, w2=0.5):    
    # Search Space
    alpha_values = np.linspace(0, 1, 1001)
    results = []
    
    for val_a in alpha_values:
        t_val, c_val = solver.calculate_scenario_metrics(val_a)        
        results.append({
            'a': val_a,
            'T': t_val,
            'C': c_val,
        })
    df = pd.DataFrame(results)
    
    # Find Anchors
    T_min = df['T'].min()
    C_min = df['C'].min()
    T_max = df['T'].max()
    C_max = df['C'].max()

    df['Z_time'] = (df['T'] - T_min) / (T_max - T_min)
    df['Z_cost'] = (df['C'] - C_min) / (C_max - C_min)
    df['Z'] = w1 * df['Z_time'] + w2 * df['Z_cost']

    # Find Optimal
    best_idx = df['Z'].idxmin()
    best_sol = df.loc[best_idx]
    
    return df, best_sol['a'], best_sol['T'], best_sol['C']

# --- Execution ---
if __name__ == "__main__":
    launch_max, price_2050 = get_parameters()
    q1_solver = Q1(FGI=launch_max, C_RB=price_2050)

    # Scenario A & B
    t_a, c_a = q1_solver.calculate_scenario_metrics(1.0)
    print(f"\nScenario A (SE Only): Time = {t_a:.2f} y, Cost = {c_a/1e9:.2f} B, a = 1.00")

    t_b, c_b = q1_solver.calculate_scenario_metrics(0.0)
    print(f"Scenario B (Rockets Only): Time = {t_b:.2f} y, Cost = {c_b/1e9:.2f} B, a = 0.00")

    # Scenario C
    # Scheme 1
    target_time = 120
    print(f"\n--- Scheme 1: Constrained Optimization (MILP Logic) ---")
    print(f"Objective: Min Cost s.t. Time <= {target_time} years")
    opt_a, opt_t, opt_c, status = optimize_scheme_1(q1_solver, target_time)

    if opt_a is not None:
        print(f"  Status: {status}")
        print(f"  Optimal a (SE Share): {opt_a:.4f}")
        print(f"  Resulting Time: {opt_t:.2f} years")
        print(f"  Resulting Cost: ${opt_c/1e9:.2f} Billion")
    else:
        print(f"  Status: {status}")

    # Scheme 2
    print(f"\n--- Scheme 2: Pareto Front Normalization ---")
    w1, w2 = 0.5, 0.5
    df_pareto, opt_a, opt_t, opt_c = optimize_scheme_2(q1_solver, w1, w2)

    print(f"\nPareto Optimal Solution (w1={w1}, w2={w2}):")
    print(f"  Parameter a: {opt_a:.4f}")
    print(f"  Time: {opt_t:.2f} y")
    print(f"  Cost: ${opt_c/1e9:.2f} B")

    # Plot
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(df_pareto['T'], df_pareto['C'], 
                     c=df_pareto['a'], cmap='viridis', 
                     label='Pareto Frontier')
    plt.colorbar(sc, label='SE ratio (a)')
    plt.scatter(opt_t, opt_c, color='red', marker='*', s=200,
                label=f"Scenario C (t={opt_t:.2f}y c={opt_c/1e9:.2f}b a={opt_a:.2f})")
    plt.scatter(t_a, c_a, color='green', marker='*', s=200,
                label=f"Scenario A (t={t_a:.2f}y c={c_a/1e9:.2f}b a=1.0)")
    plt.scatter(t_b, c_b, color='blue', marker='*', s=200,
                label=f"Scenario B (t={t_b:.2f}y c={c_b/1e9:.2f}b a=0.0)")

    plt.title('Pareto Frontier: Time vs Cost')
    plt.xlabel('Time (Years)')
    plt.ylabel('Total Cost ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('pareto.png')
    plt.show()
