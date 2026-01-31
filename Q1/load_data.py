import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    print('Loading datasets...')
    code_path = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.abspath(os.path.join(code_path, '..'))
    data_path = os.path.join(root_path, 'data')
    water_file = os.path.join(data_path, 'global-freshwater-use-over-the-long-run.csv')
    co2_file = os.path.join(data_path, 'co-emissions-per-capita.csv')
    population_file = os.path.join(data_path, 'population.csv')
    launch_file = os.path.join(data_path, 'rocket_launches.csv')

    df_water = pd.read_csv(water_file)
    df_co2 = pd.read_csv(co2_file)
    df_pop = pd.read_csv(population_file)
    df_launch = pd.read_csv(launch_file)

    # rename
    # Entity maybe country, continent, organization or even world...
    df_co2 = df_co2.rename(columns={'Annual CO₂ emissions (per capita)':'CO₂ emissions'})

    # get water usage per capita
    water_world = df_water[
        (df_water['Entity'] == 'World') &
        (df_water['Year'] >= 1950) &
        (df_water['Year'] <= 2014)
        ][['Entity', 'Year', 'Freshwater use']].copy()

    pop_col = 'Population - Sex: all - Age: all - Variant: estimates'
    pop_world = df_pop[
        (df_pop['Entity'] == 'World') &
        (df_pop['Year'] >= 1950) &
        (df_pop['Year'] <= 2014)
        ][['Entity', 'Year', pop_col]].copy()

    merged = pd.merge(
        water_world,
        pop_world,
        on=['Entity', 'Year'],
        how='inner'
    )

    merged['Per Capita Water Use'] = merged['Freshwater use'] / merged[pop_col]

    df_water_per_cap = merged[['Entity', 'Year', 'Per Capita Water Use']].copy()
    df_water_per_cap = df_water_per_cap.rename(columns={
        'Per Capita Water Use': 'Freshwater use per capita'
    })

    # launch data from 1992 to 2025
    df_launch = df_launch[pd.to_datetime(df_launch['Date']).dt.year < 2026]

    return df_launch, df_water_per_cap, df_co2

df_launch, df_water,df_co2 = load_data()
# print(df_launch.tail())
# sns.set_theme(style = "darkgrid")
# sns.lineplot(x = 'Year', y = 'Freshwater use per capita', data = df_water)
# plt.show()

