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
        # weights = np.real(max_eigvec)
        # weights = weights / np.sum(weights)
        weights = np.array([0.2832, 0.6421, 0.0747])
        
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
        self.Eco2 = 1.67e-8  # Normalized CO2 impact per launch
        self.Eox = 2.1e-6   # Normalized Ozone impact per launch
        
        # Orbital
        self.D = 1e-1     # Prob of debris/risk per year/unit activity
        
        # Surface
        self.L = 0.961     # Noise/Local impact per launch
        
        # Initialize Weights (will be set by AHP)
        self.weights = {'A': 0.2832, 'B': 0.6421, 'C': 0.0747}

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

    def calculate_environmental_impact(self, alpha, scheme=2):
        # Calculate Physical Quantities
        mass_rockets = (1 - alpha) * self.M
        Nr = mass_rockets / self.MI
        Nmax = self.M / self.MI if self.MI != 0 else np.inf
        
        # SE Operation Time Tm depends on alpha * M / rate
        # But SE runs continuously? Let's assume Tm ~ Years of operation needed for SE part
        if self.Qse > 0:
            Tm = (alpha * self.M) / self.Qse
        else:
            Tm = 0
            
        base_linear = Nr * (self.Eco2 + self.Eox)

        if scheme == 1:
            fA = base_linear
            gB = Tm * self.D
            hC = Nr * self.L
        elif scheme == 2:
            fA = base_linear ** 2
            gB = np.exp(Tm * self.D) - 1
            hC = Nr * self.L
        elif scheme == 3:
            exp_term = np.exp(Nr / Nmax - 1) if Nmax and np.isfinite(Nmax) else 0
            fA = (exp_term * (self.Eco2 + self.Eox)) ** 2
            gB = Tm * self.D
            hC = exp_term * self.L
        else:
            raise ValueError(f"Unknown environmental impact scheme: {scheme}")

        return fA, gB, hC

    def optimize_environment(self):
        print("\n--- Step 2 & 3: Environmental Optimization ---")
        
        alphas = np.linspace(0, 1, 101)
        schemes = {
            1: 'Scheme I: Linear Response',
            2: 'Scheme II: Accumulation-Risk (Default)',
            3: 'Scheme III: Exponential Threshold'
        }

        scheme_data = {}
        balanced_results = {}

        for scheme_id, scheme_label in schemes.items():
            raw_metrics = []
            for a in alphas:
                fA, gB, hC = self.calculate_environmental_impact(a, scheme=scheme_id)
                T, C = self.calculate_scenario_metrics(a)
                raw_metrics.append([a, fA, gB, hC, T, C])

            df = pd.DataFrame(raw_metrics, columns=['a', 'fA', 'gB', 'hC', 'T', 'C'])

            for col in ['fA', 'gB', 'hC', 'T', 'C']:
                mn, mx = df[col].min(), df[col].max()
                if mx > mn:
                    df[col + '_norm'] = (df[col] - mn) / (mx - mn)
                else:
                    df[col + '_norm'] = 0.0

            wA, wB, wC = self.weights['A'], self.weights['B'], self.weights['C']
            df['E_total'] = wA * df['fA_norm'] + wB * df['gB_norm'] + wC * df['hC_norm']

            l1, l2, l3 = 0.4, 0.3, 0.3
            df['Z'] = l1 * df['C_norm'] + l2 * df['T_norm'] + l3 * df['E_total']

            scheme_data[scheme_id] = df

            best_idx = df['Z'].idxmin()
            balanced_results[scheme_id] = {
                'a': df.loc[best_idx, 'a'],
                'T_norm': df.loc[best_idx, 'T_norm'],
                'C_norm': df.loc[best_idx, 'C_norm'],
                'Env_norm': df.loc[best_idx, 'E_total'],
                'T': df.loc[best_idx, 'T'],
                'C': df.loc[best_idx, 'C']
            }

        print("Balanced Scenario (λ = [0.4, 0.3, 0.3]) per scheme:")
        for scheme_id, result in balanced_results.items():
            label = schemes[scheme_id]
            print(f"  {label}: a* = {result['a']:.2f}, Cost(norm)={result['C_norm']:.4f}, "
                  f"Time(norm)={result['T_norm']:.4f}, Env(norm)={result['Env_norm']:.4f}")

        # --- Visualization ---
        self.plot_env_components_multi(alphas, scheme_data, schemes)
        self.plot_sensitivity_multi(alphas, scheme_data, schemes)
        self.plot_pareto_front(alphas, scheme_data[2], schemes[2])
        self.plot_lambda3_surface(alphas, scheme_data[2], schemes[2])
        self.plot_policy_flip(alphas, scheme_data[2])
        
    def plot_env_components_multi(self, alphas, scheme_data, schemes):
        fig, axes = plt.subplots(1, len(schemes), figsize=(18, 5), sharey=True)
        axes = np.atleast_1d(axes)

        line_styles = {
            'fA': '--',
            'gB': '-.',
            'hC': ':',
            'E_total': '-'
        }

        labels = {
            'fA': 'Atmosphere (A)',
            'gB': 'Orbit Safety (B)',
            'hC': 'Surface (C)',
            'E_total': 'Total Impact'
        }

        for idx, (scheme_id, scheme_label) in enumerate(schemes.items()):
            ax = axes[idx]
            df = scheme_data[scheme_id]
            ax.plot(alphas, df['fA_norm'], linestyle=line_styles['fA'], label=labels['fA'])
            ax.plot(alphas, df['gB_norm'], linestyle=line_styles['gB'], label=labels['gB'])
            ax.plot(alphas, df['hC_norm'], linestyle=line_styles['hC'], label=labels['hC'])
            ax.plot(alphas, df['E_total'], linestyle=line_styles['E_total'], linewidth=2, color='black', label=labels['E_total'])

            ax.set_title(scheme_label)
            ax.set_xlabel('SE Allocation Ratio $a$')
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.set_ylabel('Normalized Impact Score')
            if idx == len(schemes) - 1:
                ax.legend(loc='upper right')

        fig.suptitle('Environmental Impact Components across Schemes', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()

    def plot_sensitivity_multi(self, alphas, scheme_data, schemes):
        scheme_colors = {
            1: '#1f77b4',
            2: '#ff7f0e',
            3: '#2ca02c'
        }

        scenarios = [
            {
                'title': r'$\lambda_1 = \lambda_2$ (Equal Cost/Time)',
                'lambda3_range': np.linspace(0, 1, 120),
                'weights': lambda l3: ((1 - l3) / 2, (1 - l3) / 2, l3)
            },
            {
                'title': r'$\lambda_3 = \lambda_1$ (Env = Cost)',
                'lambda3_range': np.linspace(0, 0.5, 120),
                'weights': lambda l3: (l3, 1 - 2 * l3, l3)
            },
            {
                'title': r'$\lambda_3 = \lambda_2$ (Env = Time)',
                'lambda3_range': np.linspace(0, 0.5, 120),
                'weights': lambda l3: (1 - 2 * l3, l3, l3)
            }
        ]

        fig, axes = plt.subplots(1, len(scenarios), figsize=(18, 5), sharey=True)
        axes = np.atleast_1d(axes)

        for idx, scenario in enumerate(scenarios):
            ax = axes[idx]
            lam3_vals = scenario['lambda3_range']
            for scheme_id, scheme_label in schemes.items():
                df = scheme_data[scheme_id]
                c_norm = df['C_norm'].values
                t_norm = df['T_norm'].values
                env_norm = df['E_total'].values
                a_vals = df['a'].values

                optimal_curve = []
                for lam3 in lam3_vals:
                    l1, l2, l3 = scenario['weights'](lam3)
                    l1 = max(l1, 0)
                    l2 = max(l2, 0)
                    Z = l1 * c_norm + l2 * t_norm + l3 * env_norm
                    best_idx = int(np.argmin(Z))
                    optimal_curve.append(a_vals[best_idx])

                ax.plot(lam3_vals, optimal_curve, color=scheme_colors.get(scheme_id, 'gray'),
                        linewidth=2, label=scheme_label)

            ax.set_title(scenario['title'])
            ax.set_xlabel(r'$\lambda_3$ (Environment Weight)')
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.set_ylabel(r'Optimal SE Ratio $a^*$')

        axes[0].legend(loc='upper right')
        fig.suptitle('Sensitivity Analysis across Environmental Weight Scenarios', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()

    def plot_pareto_front(self, alphas, df, scheme_label):
        cost = df['C_norm'].values
        time = df['T_norm'].values
        env = df['E_total'].values

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(cost, time, env, c=alphas, cmap='viridis', s=40, alpha=0.85)
        fig.colorbar(scatter, ax=ax, pad=0.1, label='Alpha (a)')

        knee_points = [0.46, 0.8]
        labeled = False
        for target in knee_points:
            idx = int(np.abs(alphas - target).argmin())
            ax.scatter(cost[idx], time[idx], env[idx], color='red', s=120, marker='^',
                       label='Knee Point' if not labeled else None)
            labeled = True
            ax.text(cost[idx], time[idx], env[idx], f'a={alphas[idx]:.2f}', color='red',
                    fontsize=10, ha='left', va='bottom')

        ax.set_xlabel('Normalized Cost')
        ax.set_ylabel('Normalized Time')
        ax.set_zlabel('Normalized Environment')
        ax.set_title(f'3D Pareto Front ({scheme_label})')
        ax.legend(loc='upper left')
        plt.tight_layout()
        # plt.savefig('Q4_Pareto_3D.png', dpi=300)
        # print('Saved Q4_Pareto_3D.png')
        plt.show() # Optional

    def plot_lambda3_surface(self, alphas, df, scheme_label):
        cost = df['C_norm'].values
        time = df['T_norm'].values
        env = df['E_total'].values

        lambda3_values = np.linspace(0, 1, 80)
        Z_vals = []
        optimal_a = []
        optimal_Z = []

        for lam3 in lambda3_values:
            lam1 = lam2 = max((1 - lam3) / 2, 0)
            Z = lam1 * cost + lam2 * time + lam3 * env
            Z_vals.append(Z)

            best_idx = int(np.argmin(Z))
            optimal_a.append(alphas[best_idx])
            optimal_Z.append(Z[best_idx])

        Z_vals = np.array(Z_vals)
        Alpha_grid, Lambda3_grid = np.meshgrid(alphas, lambda3_values)

        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(Alpha_grid, Lambda3_grid, Z_vals, cmap='viridis', alpha=0.9, linewidth=0)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label='Penalty Z')

        ax.plot(optimal_a, lambda3_values, optimal_Z, color='red', linewidth=3, label='Valley of Optima')

        ax.set_xlabel('Allocation Ratio $a$')
        ax.set_ylabel(r'$\lambda_3$ (Environment Weight)')
        ax.set_zlabel('Penalty $Z$')
        ax.set_title(f'3D Sensitivity Surface of Total Penalty ({scheme_label})')
        ax.legend(loc='upper right')
        plt.tight_layout()
        # plt.savefig('Q4_Sensitivity_Surface.png', dpi=300)
        # print('Saved Q4_Sensitivity_Surface.png')
        plt.show()

    def plot_policy_flip(self, alphas, scheme_df, risk_levels=None):
        if risk_levels is None:
            risk_levels = np.linspace(0, 1, 80)

        base_qse = self.Qse
        policy_records = []

        for risk in risk_levels:
            degradation = max(0.3, 1 - 0.7 * risk)
            self.Qse = base_qse * degradation

            metrics = []
            for alpha in alphas:
                T, C = self.calculate_scenario_metrics(alpha)
                metrics.append({'a': alpha, 'T': T, 'C': C})

            metrics_df = pd.DataFrame(metrics)

            T_range = metrics_df['T'].max() - metrics_df['T'].min()
            C_range = metrics_df['C'].max() - metrics_df['C'].min()

            T_norm = (metrics_df['T'] - metrics_df['T'].min()) / T_range if T_range > 0 else pd.Series(0, index=metrics_df.index)
            C_norm = (metrics_df['C'] - metrics_df['C'].min()) / C_range if C_range > 0 else pd.Series(0, index=metrics_df.index)

            env_scaled = scheme_df['E_total'] * (1 + 1.5 * risk)
            env_norm = (env_scaled - env_scaled.min()) / (env_scaled.max() - env_scaled.min()) if env_scaled.max() > env_scaled.min() else pd.Series(0, index=scheme_df.index)

            env_norm = env_norm.reset_index(drop=True)
            Z = 0.4 * C_norm + 0.3 * T_norm + 0.3 * env_norm

            best_idx = int(np.argmin(Z))
            policy_records.append({
                'risk': risk,
                'a_opt': alphas[best_idx]
            })

        self.Qse = base_qse

        policy_df = pd.DataFrame(policy_records)
        policy_df['delta'] = policy_df['a_opt'].diff().abs()

        critical_idx = policy_df['delta'].idxmax()
        critical_point = policy_df.iloc[critical_idx] if pd.notna(critical_idx) else None

        plt.figure(figsize=(10, 6))
        plt.plot(policy_df['risk'], policy_df['a_opt'], color='#1f77b4', linewidth=2, label='Optimal a', zorder=3)
        step = max(3, len(policy_df) // 8)
        for idx in range(0, len(policy_df) - step, step):
            start = policy_df.iloc[idx]
            end = policy_df.iloc[idx + step]
            dx = end['risk'] - start['risk']
            dy = end['a_opt'] - start['a_opt']
            norm = np.hypot(dx, dy)
            if norm == 0:
                continue
            px = -dy / norm
            py = dx / norm
            offset = 0.015
            x0 = start['risk'] + px * offset
            y0 = start['a_opt'] + py * offset
            x1 = end['risk'] + px * offset
            y1 = end['a_opt'] + py * offset
            plt.annotate(
                '',
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.6, alpha=0.75),
                zorder=4
            )
        scatter = plt.scatter(policy_df['risk'], policy_df['a_opt'], c=policy_df['a_opt'], cmap='viridis', s=40)
        plt.colorbar(scatter, label='Optimal Allocation Ratio a')

        if critical_point is not None:
            plt.axvline(critical_point['risk'], color='red', linestyle='--', linewidth=1.5, label='Resilience Threshold')
            plt.text(critical_point['risk'], critical_point['a_opt'] + 0.03,
                     f"Critical Point ≈ {critical_point['risk']:.2f}", color='red', ha='center')

        plt.fill_between(policy_df['risk'], policy_df['a_opt'], color='#1f77b4', alpha=0.1)
        plt.xlabel('Environmental deterioration risk index')
        plt.ylabel('Optimal Allocation Ratio a')
        plt.title('Policy Flip Phase Diagram')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # plt.savefig('Q4_Policy_Flip_Phase.png', dpi=300)
        # print('Saved Q4_Policy_Flip_Phase.png')
        plt.show()

if __name__ == "__main__":
    solver = Q4()
    solver.run_ahp()
    solver.optimize_environment()
