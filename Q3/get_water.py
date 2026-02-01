import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

from Q1.load_data import load_data

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

    if draw_plots:
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Population
        axes[0].scatter(years_pop, population, color='royalblue', label='Historical Data', alpha=0.7)
        if pop_2050 is not None:
            # Generate smooth curve from start year to 2050
            t_range = np.linspace(0, 2050 - years_pop[0], 100)
            years_smooth = years_pop[0] + t_range
            pop_smooth = logistic_growth(t_range, *popt_pop)

            axes[0].plot(years_smooth, pop_smooth, color='darkred', linestyle='--', label='Logistic Fit')
            axes[0].scatter([2050], [pop_2050], color='green', s=100, marker='*', zorder=5,
                            label=f'2050 Pred: {pop_2050 / 1e9:.2f}B')

        axes[0].set_title('US Population Projection (Logistic Model)')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Population')
        axes[0].legend()
        axes[0].grid(True, linestyle=':', alpha=0.6)

        # Plot 2: Water Usage
        if len(water_data) > 0:
            axes[1].scatter(years_water, water_usage, color='royalblue', label='Historical Data', alpha=0.7)
            if water_2050 is not None:
                t_range_w = np.linspace(0, 2050 - years_water[0], 100)
                years_smooth_w = years_water[0] + t_range_w
                water_smooth = logistic_growth(t_range_w, *popt_water)

                axes[1].plot(years_smooth_w, water_smooth, color='darkred', linestyle='--', label='Logistic Fit')
                axes[1].scatter([2050], [water_2050], color='green', s=100, marker='*', zorder=5,
                                label=f'2050 Pred: {water_2050 / 1e9:.0f} B m3')

        axes[1].set_title('US Freshwater Use Projection (Logistic Model)')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Freshwater Use (L)')
        axes[1].legend()
        axes[1].grid(True, linestyle=':', alpha=0.6)

        plt.suptitle(f'Parameter Estimation for 2050 (Per Capita Water: {water_per_cap:.2f} m3/y)', fontsize=14)
        plt.tight_layout()
        plt.show()

    return water_per_cap

if __name__ == "__main__":
    water_per_cap = get_parameters(draw_plots=True)
    print(water_per_cap)