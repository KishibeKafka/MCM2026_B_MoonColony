import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

from Q1.Q1 import Q1
from Q1.load_data import load_data

class Q3(Q1):
    def __init__(self, FGI=593, C_RB=21464577.32, water_per_cap=541.87):
        super().__init__(FGI, C_RB)
        
        # Load trained parameters (Optional usage)
        self.water_per_cap = water_per_cap 
        
        # --- Capacity Scaling for 2050 Era ---
        # Base Q1 capacity ~1.2M. Demand ~20M. Need scaling.
        # Assuming infrastructure growth to meet demand baseline.
        self.scale_factor = 20.0
        self.Qse_base = self.Qse * self.scale_factor
        self.Qrocket_base = self.Qrocket * self.scale_factor
        
        # Constants
        self.WS_max = 1e8 # 100 M tons
        self.WS = self.WS_max * 0.5 # Init 50%
        
        # SA Hyperparams (Optimized for Speed)
        self.T_start = 100.0
        self.T_end = 1e-3 # Stop slightly earlier
        self.gamma = 0.90 # Faster cooling
        self.N_mc = 500   # High samples, but vectorized
        self.nabta = 0.6 

    def predict_demand(self, year):
        # Stable population logic (as per previous context snippets)
        P_stable = 1e5 
        w_p = self.water_per_cap
        r_cyc = 0.9
        
        W_person = w_p * P_stable * (1 - r_cyc)
        W_build = W_person * 0.25
        return W_person + W_build

    def calculate_mc_vectorized(self, alpha, task_weight, n_samples):
        # Generate noise arrays
        # Uniform [0.8, 1.0]
        noise_se = np.random.uniform(0.8, 1.0, n_samples)
        noise_rk = np.random.uniform(0.8, 1.0, n_samples)
        
        Qse_eff = self.Qse_base * noise_se
        Qrk_eff = self.Qrocket_base * noise_rk
        
        # Time Calculation
        # Avoid div zero
        t_se = np.divide(alpha * task_weight, Qse_eff, out=np.full_like(Qse_eff, 1e6), where=Qse_eff>1e-6)
        t_rk = np.divide((1-alpha) * task_weight, Qrk_eff, out=np.full_like(Qrk_eff, 1e6), where=Qrk_eff>1e-6)
        
        if alpha <= 1e-9: t_se[:] = 0
        if alpha >= 1.0 - 1e-9: t_rk[:] = 0
            
        Tc_arr = np.maximum(t_se, t_rk)
        
        # Cost Calculation (Formula: Constant * Weight basically)
        # Ctot = a *W *Ce + Ce*(1-b)*W + (1-a)*W*Crm + Crb*(1-a)*M/mi
        # Calculate scalar cost first
        term1 = alpha * task_weight * self.Ce
        term2 = self.Ce * (1 - self.b_factor) * task_weight
        term3 = (1 - alpha) * task_weight * self.C_RM
        term4 = (1 - alpha) * task_weight * self.C_RB / self.MI
        C_scalar = term1 + term2 + term3 + term4
        
        C_arr = np.full(n_samples, C_scalar)
        
        Cap_arr = Qse_eff + Qrk_eff
        return Tc_arr, C_arr, Cap_arr

    def energy_function(self, alpha, task_weight, current_ws, C_base, T_base):
        
        Tc_arr, C_arr, Cap_arr = self.calculate_mc_vectorized(alpha, task_weight, self.N_mc)
        
        # Penalties
        P_arr = np.zeros(self.N_mc)
        
        # 1. Time Constraint: T > 1.05
        P_arr[Tc_arr > 1.05] += 1e9
        
        # 2. WS Constraints
        # WS_next = WS + Supply - Demand
        # Supply = min(Demand, Capacity) ?? 
        # If we *ordered* 'task_weight', and 'Capacity' < 'task_weight', we get Capacity.
        # If 'Capacity' > 'task_weight', we get 'task_weight'.
        supplied = np.minimum(task_weight, Cap_arr)
        ws_next_arr = current_ws + supplied - task_weight
        
        # Bounds check
        mask_low = ws_next_arr < 0
        mask_high = ws_next_arr > self.WS_max
        
        P_arr[mask_low] += 1e9
        P_arr[mask_high] += 1e9
        
        # Normalize
        score = self.nabta * (C_arr / C_base) + (1 - self.nabta) * (Tc_arr / T_base) + P_arr
        
        return np.mean(score)

    def simulated_annealing(self, W_demand, current_ws):
        # Init
        a_curr = 0.5
        T = self.T_start
        
        # Baseline
        # Expected metrics at a=0.8
        tc, cc, _ = self.calculate_mc_vectorized(0.8, W_demand, 10)
        c_base = np.mean(cc)
        t_base = np.mean(tc)
        if t_base < 1e-6: t_base = 1.0
        
        E_curr = self.energy_function(a_curr, W_demand, current_ws, c_base, t_base)
        best_a = a_curr
        
        while T > self.T_end:
            # Vectorized Inner Loop? No, SA is sequential.
            # Reduced L loop
            for _ in range(5):
                sigma = 0.1 if T > 50 else 0.01
                a_new = np.clip(a_curr + np.random.normal(0, sigma), 0.0, 1.0)
                
                E_new = self.energy_function(a_new, W_demand, current_ws, c_base, t_base)
                
                dE = E_new - E_curr
                if dE < 0 or np.random.rand() < np.exp(-dE / T):
                    a_curr = a_new
                    E_curr = E_new
                    if E_curr < 1e8: # Only track valid solutions
                        best_a = a_new
            
            T *= self.gamma
            
        return best_a

    def run_simulation(self, start_year=2050, duration=50):
        print(f"Starting Q3 Fast Optimization {start_year}-{start_year+duration}...")
        years = np.arange(start_year, start_year + duration)
        results = []
        
        curr_ws = self.WS
        
        for yr in years:
            W_dem = self.predict_demand(yr)
            
            # REFILL STRATEGY:
            # If WS is low, try to transport MORE than demand.
            # target = demand + (Target_WS - Current_WS) * rate
            target_fill = self.WS_max * 0.8
            if curr_ws < target_fill:
                # Try to refill 20% of the gap
                gap = target_fill - curr_ws
                W_target = W_dem + gap * 0.2
            else:
                W_target = W_dem
                
            # Optimize a for W_target
            best_a = self.simulated_annealing(W_target, curr_ws)
            
            # Execute (Realization)
            noise_se = np.random.uniform(0.8, 1.0)
            noise_rk = np.random.uniform(0.8, 1.0)
            
            Qeff = (self.Qse_base * noise_se) + (self.Qrocket_base * noise_rk)
            
            # Actual Delivered
            # Tc check
            t_se = (best_a * W_target) / (self.Qse_base * noise_se) if best_a > 0 else 0
            t_rk = ((1-best_a)*W_target)/(self.Qrocket_base*noise_rk) if best_a < 1 else 0
            T_actual = max(t_se, t_rk)
            
            if T_actual <= 1.0:
                # Delivered full target
                W_delivered = W_target
            else:
                # Delivered capacity only
                W_delivered = Qeff * 1.0
            
            # Inventory Update
            # WS_t = WS_{t-1} + W_in - W_out
            ws_next = curr_ws + W_delivered - W_dem
            ws_next = max(0, min(self.WS_max, ws_next))
            
            # Cost
            # Deterministic cost func
            term1 = best_a * W_target * self.Ce
            term2 = self.Ce * (1 - self.b_factor) * W_target
            term3 = (1 - best_a) * W_target * self.C_RM
            term4 = (1 - best_a) * W_target * self.C_RB / self.MI
            Cost_real = term1 + term2 + term3 + term4
            
            results.append({
                'Year': yr,
                'Ws': ws_next,
                'a': best_a,
                'Cost': Cost_real,
                'Demand': W_dem
            })
            curr_ws = ws_next
            
            if (yr - start_year) % 10 == 0:
                print(f"Year {yr} | Dem: {W_dem/1e6:.1f}M | Target: {W_target/1e6:.1f}M | a: {best_a:.2f} | WS: {ws_next/1e6:.1f}M")
                
        return pd.DataFrame(results)

def logistic_growth(t, K, P0, r):
    return K / (1 + (K / P0 - 1) * np.exp(-r * t))

def get_parameters(draw_plots=False):
    # Load datasets
    _, df_water, df_pop, _ = load_data()
    
    # --- 1. Population Model (Logistic Growth) ---
    pop_data = df_pop.dropna(subset=['Year', 'Population'])
    years_pop = pop_data['Year'].values
    population = pop_data['Population'].values
    
    # Normalize time (t=0 at the start of the dataset)
    t_pop = years_pop - years_pop[0]
    
    # Initial Parameter Guesses
    p0_guess = [1.2e10, population[0], 0.02]
    bounds = ([max(population), 0, 0], [np.inf, np.inf, np.inf])
    
    try:
        popt_pop, _ = curve_fit(logistic_growth, t_pop, population, p0=p0_guess, bounds=bounds)
        K_fit, P0_fit, r_fit = popt_pop
        
        # Predict 2050 Population
        target_year_pop = 2050
        t_target_pop = target_year_pop - years_pop[0]
        pop_2050 = logistic_growth(t_target_pop, *popt_pop)
        
    except Exception as e:
        print(f"Error fitting population model: {e}")
        pop_2050 = None

    # --- 2. Water Usage Model (Logistic Growth) ---
    water_data = df_water.dropna(subset=['Year', 'Freshwater use'])
    years_water = water_data['Year'].values
    water_usage = water_data['Freshwater use'].values
    t_water = years_water - years_water[0]

    if len(water_data) > 0:
        w0_guess = [max(water_usage) * 1.5, water_usage[0], 0.02]
        w_bounds = ([max(water_usage), 0, 0], [np.inf, np.inf, np.inf])

        try:
            popt_water, _ = curve_fit(logistic_growth, t_water, water_usage, p0=w0_guess, bounds=w_bounds)
            water_2050 = logistic_growth(2050 - years_water[0], *popt_water)
        except Exception as e:
            print(f"Error fitting water model: {e}")
            water_2050 = None
    else:
        water_2050 = None

    water_per_cap = water_2050 / pop_2050 if (water_2050 and pop_2050) else 541.87
    return water_per_cap

if __name__ == "__main__":
    water_per_cap = get_parameters(draw_plots=False)
    
    solver = Q3(water_per_cap=water_per_cap)
    
    print("\nStarting Q3 Simulation (2050-2100)...")
    # Run
    df_res = solver.run_simulation(start_year=2050, duration=50)
    
    print("\n--- Final Results Head ---")
    print(df_res.head())
    print("\n--- Final Results Tail ---")
    print(df_res.tail())
    print("\nDone.")

