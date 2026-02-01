import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Q1.Q1 import Q1

# --- 1. AHP Class for Weight Calculation ---
class AHP:
    def __init__(self, criteria):
        self.criteria = criteria
        self.n = len(criteria)
        self.matrix = np.eye(self.n)
        
    def set_judgment_matrix(self, matrix):
        self.matrix = np.array(matrix)
        
    def calculate_weights(self):
        eigvals, eigvecs = np.linalg.eig(self.matrix)
        max_eigval = np.max(eigvals)
        max_eigvec = eigvecs[:, np.argmax(eigvals)]
        
        # Normalize to real and sum to 1
        weights = np.real(max_eigvec)
        weights = weights / np.sum(weights)
        
        # Consistency Check
        CI = (max_eigval - self.n) / (self.n - 1)
        RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12}
        RI = RI_dict.get(self.n, 1.41)
        CR = CI / RI if RI != 0 else 0
        
        return weights, CR

# --- 2. Q4 Class embedding LCA and Scenario Analysis ---
class Q4(Q1):
    def __init__(self, FGI=593, C_RB=21464577.32):
        super().__init__(FGI, C_RB)
        
        # --- LCA Parameters ---
        # Atmospheric
        self.Eco2 = 1.0  # Normalized CO2 impact per launch
        self.Eox = 0.5   # Normalized Ozone impact per launch
        
        # Orbital
        self.D = 0.25     # Prob of debris/risk per year/unit activity
        
        # Surface
        self.L = 0.1     # Noise/Local impact per launch
        
        # Initialize Weights (will be set by AHP)
        self.weights = {'A': 0.33, 'B': 0.33, 'C': 0.33}

    def run_ahp(self):
        print("\n--- Step 1: AHP Weight Calculation ---")
        # Criteria: A (Atmosphere), B (Orbit), C (Surface)
        # Judgment Matrix: 
        # B is slightly more important than A (Orbit debris is irreversible)
        # B is much more important than C (Local noise is manageable)
        # A is more important than C (Global vs Local)
        
        #      A    B    C
        # A    1   1/2   3
        # B    2    1    5
        # C   1/3  1/5   1
        
        ahp = AHP(criteria=['Atmosphere', 'Orbital Safety', 'Surface Ecology'])
        matrix = np.array([
            [1,   0.5, 3],
            [2,   1,   5],
            [1/3, 0.2, 1]
        ])
        ahp.set_judgment_matrix(matrix)
        w, CR = ahp.calculate_weights()
        
        self.weights['A'] = w[0]
        self.weights['B'] = w[1]
        self.weights['C'] = w[2]
        
        print(f"Weights: Atmosphere (wA)={w[0]:.3f}, Orbit (wB)={w[1]:.3f}, Surface (wC)={w[2]:.3f}")
        print(f"Consistency Ratio (CR): {CR:.4f} ({'Pass' if CR < 0.1 else 'Fail'})")

    def calculate_environmental_impact(self, alpha):
        # Calculate Physical Quantities
        mass_rockets = (1 - alpha) * self.M
        Nr = mass_rockets / self.MI
        Nmax = np.divide(self.M, self.MI)
        
        # SE Operation Time Tm depends on alpha * M / rate
        # But SE runs continuously? Let's assume Tm ~ Years of operation needed for SE part
        if self.Qse > 0:
            Tm = (alpha * self.M) / self.Qse
        else:
            Tm = 0
            
        # --- LCA Calculation ---
        # fA = [ Nr * (Eco2 + Eox) ] ^2
        # fA = Nr * (self.Eco2 + self.Eox)
        # fA = (Nr * (self.Eco2 + self.Eox)) ** 2
        fA = (np.exp(np.divide(Nr, Nmax) - 1) * (self.Eco2 + self.Eox))**2

        # gB = e ^ (Tm * D) - 1
        # gB = np.exp(Tm * self.D) - 1
        gB = Tm * self.D
        
        # 3. Surface Ecology (C)
        # hC = Nr * L
        # hC = Nr * self.L
        hC = np.exp(np.divide(Nr, Nmax) - 1) * self.L

        
        return fA, gB, hC

    def optimize_environment(self):
        print("\n--- Step 2 & 3: Environmental Optimization ---")
        
        alphas = np.linspace(0, 1, 101)
        results = []
        
        # 1. Collect Raw Data for Normalization
        raw_metrics = []
        for a in alphas:
            fA, gB, hC = self.calculate_environmental_impact(a)
            T, C = self.calculate_scenario_metrics(a)
            raw_metrics.append([fA, gB, hC, T, C])
            
        raw_df = pd.DataFrame(raw_metrics, columns=['fA', 'gB', 'hC', 'T', 'C'])

        # 2. Normalize (Min-Max)
        # Avoid div by zero
        for col in ['fA', 'gB', 'hC', 'T', 'C']:
            mn, mx = raw_df[col].min(), raw_df[col].max()
            if mx > mn:
                raw_df[col + '_norm'] = (raw_df[col] - mn) / (mx - mn)
            else:
                raw_df[col + '_norm'] = 1.0

        # 3. Calculate E_total and Objective Z
        # E_total = wA * fA_norm + wB * gB_norm + wC * hC_norm
        wA, wB, wC = self.weights['A'], self.weights['B'], self.weights['C']

        raw_df['E_total'] = wA * raw_df['fA_norm'] + wB * raw_df['gB_norm'] + wC * raw_df['hC_norm']
        
        # Min Z = lambda1 * Ccost + lambda2 * Ttime + lambda3 * E_total
        # Scenario: Balanced
        l1, l2, l3 = 0.4, 0.3, 0.3
        raw_df['Z'] = l1 * raw_df['C_norm'] + l2 * raw_df['T_norm'] + l3 * raw_df['E_total']
        
        best_idx = raw_df['Z'].idxmin()
        best_a = alphas[best_idx]
        
        print(f"Optimal Alpha (Balanced): {best_a:.2f}")
        print(f"  Cost (Norm): {raw_df.loc[best_idx, 'C_norm']:.4f}")
        print(f"  Time (Norm): {raw_df.loc[best_idx, 'T_norm']:.4f}")
        print(f"  Env (Norm):  {raw_df.loc[best_idx, 'E_total']:.4f}")
        
        # --- Visualization ---
        self.plot_sensitivity(alphas, raw_df)
        
    def plot_sensitivity(self, alphas, df):
        # 1. Impact Components vs Alpha
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, df['fA_norm'], label='Atmosphere (A) ~ Rockets', linestyle='--')
        plt.plot(alphas, df['gB_norm'], label='Orbit Safety (B) ~ Mixed', linestyle='-.')
        plt.plot(alphas, df['hC_norm'], label='Surface (C) ~ Rockets', linestyle=':')
        plt.plot(alphas, df['E_total'], label='Total Env Impact', linewidth=2, color='black')
        
        plt.xlabel('SE Allocation Ratio (a)')
        plt.ylabel('Normalized Impact Score')
        plt.title('Environmental Impact Components vs SE Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('Q4_Env_Components.png')
        print("Saved Q4_Env_Components.png")
        plt.show() # Optional
        
        # 2. Sensitivity Analysis - 3 Scenarios
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # (1) Lambda1 = Lambda2, Lambda3 varies 0->1
        # L1 + L2 + L3 = 1 => 2*L1 = 1 - L3 => L1 = L2 = (1 - L3)/2
        l3_range1 = np.linspace(0, 1, 100)
        opt_a_1 = []
        for l3 in l3_range1:
            l1 = (1 - l3) / 2
            l2 = l1
            Z = l1 * df['C_norm'] + l2 * df['T_norm'] + l3 * df['E_total']
            opt_a_1.append(alphas[Z.idxmin()])
        
        axes[0].plot(l3_range1, opt_a_1, 'b-', linewidth=2)
        axes[0].set_title(r'$\lambda_1 = \lambda_2$ (Equal Cost/Time)')
        axes[0].set_xlabel(r'$\lambda_3$ (Environment)')
        axes[0].set_ylabel(r'Optimal SE Ratio $a^*$')
        axes[0].grid(True, alpha=0.3)

        # (2) Lambda3 = Lambda1, Lambda3 varies 0->0.5
        # L1 + L2 + L3 = 1 => L3 + L2 + L3 = 1 => L2 = 1 - 2*L3
        l3_range2 = np.linspace(0, 0.5, 100)
        opt_a_2 = []
        for l3 in l3_range2:
            l1 = l3
            l2 = 1 - 2 * l3
            Z = l1 * df['C_norm'] + l2 * df['T_norm'] + l3 * df['E_total']
            opt_a_2.append(alphas[Z.idxmin()])
            
        axes[1].plot(l3_range2, opt_a_2, 'r-', linewidth=2)
        axes[1].set_title(r'$\lambda_3 = \lambda_1$ (Env = Cost)')
        axes[1].set_xlabel(r'$\lambda_3$ (Environment)')
        axes[1].grid(True, alpha=0.3)

        # (3) Lambda3 = Lambda2, Lambda3 varies 0->0.5
        # L1 + L2 + L3 = 1 => L1 + L3 + L3 = 1 => L1 = 1 - 2*L3
        l3_range3 = np.linspace(0, 0.5, 100)
        opt_a_3 = []
        for l3 in l3_range3:
            l2 = l3
            l1 = 1 - 2 * l3
            Z = l1 * df['C_norm'] + l2 * df['T_norm'] + l3 * df['E_total']
            opt_a_3.append(alphas[Z.idxmin()])
            
        axes[2].plot(l3_range3, opt_a_3, 'g-', linewidth=2)
        axes[2].set_title(r'$\lambda_3 = \lambda_2$ (Env = Time)')
        axes[2].set_xlabel(r'$\lambda_3$ (Environment)')
        axes[2].grid(True, alpha=0.3)

        plt.suptitle('Sensitivity Analysis: Optimal Strategy $a^*$ under Constraint Relationships', fontsize=16)
        plt.tight_layout()
        plt.savefig('Q4_Sensitivity_3Scenarios.png')
        print("Saved Q4_Sensitivity_3Scenarios.png")
        plt.show() # Optional

if __name__ == "__main__":
    solver = Q4()
    solver.run_ahp()
    solver.optimize_environment()
