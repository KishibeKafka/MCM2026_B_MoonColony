import numpy as np
import pandas as pd

from Q1.Q1 import Q1
from Q2.Q2 import Q2
from Q4.Q4 import Q4


def prepare_q1_solver():
	launch_capacity = 593  # derived from Q1 forecast
	rocket_repair_cost = 21464577.32  # derived from Q1/Q2 inputs
	return Q1(FGI=launch_capacity, C_RB=rocket_repair_cost)


def prepare_q2_solver():
	launch_capacity = 593
	rocket_repair_cost = 21464577.32
	return Q2(FGI=launch_capacity, C_RB=rocket_repair_cost)


def prepare_q4_solver():
	solver = Q4()
	solver.run_ahp()
	return solver


def evaluate_environment_curve(q4_solver, scheme=2, steps=101):
	alphas = np.linspace(0, 1, steps)
	records = []
	for alpha in alphas:
		f_a, g_b, h_c = q4_solver.calculate_environmental_impact(alpha, scheme=scheme)
		records.append({'a': alpha, 'fA': f_a, 'gB': g_b, 'hC': h_c})

	df = pd.DataFrame(records)
	for col in ['fA', 'gB', 'hC']:
		min_val = df[col].min()
		max_val = df[col].max()
		if max_val > min_val:
			df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
		else:
			df[f'{col}_norm'] = 0.0

	weights = q4_solver.weights
	df['E_total_norm'] = (
		weights['A'] * df['fA_norm'] +
		weights['B'] * df['gB_norm'] +
		weights['C'] * df['hC_norm']
	)
	return df


def find_alpha_for_objective(q1_solver, metric='cost', steps=1001):
	alphas = np.linspace(0, 1, steps)
	best_alpha = 0.0
	best_value = None
	for alpha in alphas:
		time_val, cost_val = q1_solver.calculate_scenario_metrics(alpha)
		value = cost_val if metric == 'cost' else time_val
		if best_value is None or value < best_value:
			best_value = value
			best_alpha = alpha
	return best_alpha, best_value


def lookup_environment_score(env_df, alpha):
	idx = (env_df['a'] - alpha).abs().idxmin()
	return env_df.loc[idx, 'E_total_norm']


def evaluate_plan_metrics(plan_id, alpha, reliability, q1_solver, q2_solver, env_df):
	base_time, base_cost = q1_solver.calculate_scenario_metrics(alpha)
	rel_time, rel_cost = q2_solver.calculate_scenario_metrics(
		alpha,
		reliability['Pes'],
		reliability['Pr'],
		reliability['weather_months']
	)

	delay = max(rel_time - base_time, 0)
	robustness = max(0.0, min(1.0, 1 - delay / base_time if base_time > 0 else 0))
	env_score = lookup_environment_score(env_df, alpha)

	return {
		'plan': plan_id,
		'alpha': alpha,
		'T_base': base_time,
		'C_base': base_cost,
		'T_c': rel_time,
		'C_tot': rel_cost,
		'E_total': env_score,
		'R': robustness
	}


def construct_decision_matrix(q1_solver, q2_solver, env_df):
	alpha_cost, _ = find_alpha_for_objective(q1_solver, metric='cost')
	alpha_time, _ = find_alpha_for_objective(q1_solver, metric='time')
	alpha_env = env_df.loc[env_df['E_total_norm'].idxmin(), 'a']
	alpha_balanced = 0.46

	plans = {
		'P1': {
			'label': 'Cost-Lean Elevator Build',
			'alpha': alpha_cost,
			'reliability': {'Pes': 0.04, 'Pr': 0.12, 'weather_months': 11}
		},
		'P2': {
			'label': 'Rapid Rocket Surge',
			'alpha': alpha_time,
			'reliability': {'Pes': 0.08, 'Pr': 0.05, 'weather_months': 10}
		},
		'P3': {
			'label': 'Low-Carbon Orbital Safety',
			'alpha': alpha_env,
			'reliability': {'Pes': 0.02, 'Pr': 0.08, 'weather_months': 11}
		},
		'P4': {
			'label': 'Resilient Hybrid Knee',
			'alpha': alpha_balanced,
			'reliability': {'Pes': 0.05, 'Pr': 0.07, 'weather_months': 11}
		}
	}

	rows = []
	for plan_id, config in plans.items():
		rows.append(evaluate_plan_metrics(
			plan_id,
			config['alpha'],
			config['reliability'],
			q1_solver,
			q2_solver,
			env_df
		))

	df = pd.DataFrame(rows)
	df.set_index('plan', inplace=True)
	df['label'] = [plans[idx]['label'] for idx in df.index]
	return df


def topsis(decision_df):
	benefit_df = decision_df.copy()

	for column in ['T_c', 'C_tot', 'E_total']:
		max_val = benefit_df[column].max()
		benefit_df[column] = max_val - benefit_df[column]

	norm_df = benefit_df[['T_c', 'C_tot', 'E_total', 'R']].copy()
	for column in norm_df.columns:
		denom = np.sqrt((norm_df[column] ** 2).sum())
		norm_df[column] = norm_df[column] / denom if denom > 0 else 0

	weights = np.array([0.088, 0.161, 0.284, 0.467])
	weighted = norm_df.mul(weights, axis=1)

	ideal_best = weighted.max(axis=0)
	ideal_worst = weighted.min(axis=0)

	dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
	dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

	closeness = dist_worst / (dist_best + dist_worst)

	result = decision_df.copy()
	result['D_plus'] = dist_best
	result['D_minus'] = dist_worst
	result['w'] = weighted.sum(axis=1)
	result['C'] = closeness
	result['rank'] = result['C'].rank(ascending=False, method='dense').astype(int)
	return result.sort_values('rank'), weights


def format_metric(value, unit):
	if unit == 'years':
		return f"{value:,.1f} years"
	if unit == 'usd':
		return f"${value/1e9:,.2f} B"
	if unit == 'score':
		return f"{value:.3f}"
	return f"{value:.2f}"


def generate_letter(ranked_df):
	best_plan = ranked_df.iloc[0]
	plan_titles = {
		'P1': 'Cost-Lean Elevator Build',
		'P2': 'Rapid Rocket Surge',
		'P3': 'Low-Carbon Orbital Safety',
		'P4': 'Resilient Hybrid Knee'
	}

	narrative = []
	narrative.append("MCM Lunar Systems Directorate\nAttn: Strategic Infrastructure Review Board\n\n")
	narrative.append("Subject: Integrated Deployment Blueprint for a 100,000-Resident Lunar Settlement\n\n")
	narrative.append("Directors,\n\n")
	narrative.append("We completed a TOPSIS multi-criteria appraisal spanning cost, schedule, sustainability, and robustness across four canonical build sequences. The weighted normalization employed the board-approved AHP vector w = [0.088, 0.161, 0.284, 0.467], and it favors designs that stay on schedule under Monte Carlo reliability stress tests.\n\n")

	narrative.append(f"Recommendation - adopt plan {best_plan.name} ({plan_titles[best_plan.name]}). It posts the highest proximity to the ideal solution (C = {best_plan['closeness']:.3f}) by combining a "
					 f"{format_metric(best_plan['T_c'], 'years')} critical path, lifecycle spending of {format_metric(best_plan['C_tot'], 'usd')}, an environmental layer score of {best_plan['E_total']:.3f}, "
					 f"and robustness R = {best_plan['R']:.3f}.\n\n")

	runner_ups = ranked_df.iloc[1:]
	narrative.append("Comparative insights:\n")
	for _, row in runner_ups.iterrows():
		gap = best_plan['closeness'] - row['closeness']
		narrative.append(f"• {row.name} trails by {gap:.3f} on the TOPSIS scale; {plan_titles[row.name]} is constrained primarily by "
						 f"{'cost overruns' if row['C_tot'] > best_plan['C_tot'] else 'schedule slippage' if row['T_c'] > best_plan['T_c'] else 'environmental load' if row['E_total'] > best_plan['E_total'] else 'reliability exposure'}.\n")
	narrative.append("\nProgrammatic phasing:\n")
	narrative.append("1. Near term (1-5 years): Lean on the ten launch complexes for rapid depot activation while hardening elevator cables to the plan's reliability settings.\n")
	narrative.append("2. Mid term (5-50 years): Ramp the Galactic Port elevator throughput to alpha >= 0.6, locking in the environmental dividend while executing orbital debris mitigation from Q4.\n")
	narrative.append("3. Steady state: Apply the simulated-annealing dispatch from Q3 to keep water and bulk cargo loops balanced; reserve rocket bandwidth for contingencies only.\n\n")
	narrative.append("Mitigation notes: continue quarterly Monte Carlo reliability audits to keep R above 0.9, dedicate 3% of the capex line to corrosion and micrometeoroid shielding, and retain surge launch contracts to absorb the residual lambda3-driven risk valley mapped in Q4.\n\n")

	return ''.join(narrative)


def main():
	q1_solver = prepare_q1_solver()
	q2_solver = prepare_q2_solver()
	q4_solver = prepare_q4_solver()

	env_df = evaluate_environment_curve(q4_solver, scheme=2)
	decision_df = construct_decision_matrix(q1_solver, q2_solver, env_df)
	ranked, weights = topsis(decision_df)

	print("Top-level decision matrix (raw metrics):")
	print(decision_df[['label', 'alpha', 'T_c', 'C_tot', 'E_total', 'R']])

	print("\nTOPSIS weights (w):", weights)
	display_cols = ['label', 'D_plus', 'D_minus', 'w', 'C', 'rank']
	print("\nTOPSIS summary (per plan):")
	print(ranked[display_cols])


if __name__ == '__main__':
	main()
