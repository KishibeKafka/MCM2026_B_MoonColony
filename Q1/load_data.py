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

    water_world = df_water[
        (df_water['Entity'] == 'World')][['Entity', 'Year', 'Freshwater use']].copy()
    water_world['Freshwater use'] = water_world['Freshwater use'] * 0.043

    pop_col = 'Population - Sex: all - Age: all - Variant: estimates'
    pop_world = df_pop[(df_pop['Entity'] == 'World')][['Entity', 'Year', pop_col]].copy()
    pop_world = pop_world.rename(columns={pop_col: 'Population'})
    pop_world['Population'] = pop_world['Population'] * 0.043

    # rename
    # Entity maybe country, continent, organization or even world...
    df_co2 = df_co2.rename(columns={'Annual CO₂ emissions (per capita)':'CO₂ emissions'})

    # launch data from 1992 to 2025
    df_launch['Date'] = pd.to_datetime(df_launch['Date'], utc=True, errors='coerce')
    df_launch['Year'] = df_launch['Date'].dt.year
    df_launch = df_launch[df_launch['Year'] < 2026]

    return df_launch, water_world, pop_world, df_co2

def water_plot():
    code_path = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.abspath(os.path.join(code_path, '..'))
    data_path = os.path.join(root_path, 'data')
    water_detail_file = os.path.join(data_path, 'cleaned_global_water_consumption.csv')
    df_wd = pd.read_csv(water_detail_file)
    
    # 绘制美国2024(或最近一年)年各类用水占比的饼图
    # 筛选美国数据
    df_usa = df_wd[df_wd['Country'].str.contains('United States|USA', case=False, na=False)]
    
    if df_usa.empty:
        print("未找到美国相关数据")
        return

    # 获取最2023年的数据
    latest_row = df_usa[df_usa['Year'] == 2000].iloc[0]
    year = int(latest_row['Year'])
    
    # 获取各类用水比例
    ag_use = latest_row['Agricultural Water Use (%)']
    ind_use = latest_row['Industrial Water Use (%)']
    house_use = latest_row['Household Water Use (%)']
    other_use = 100 - (ag_use + ind_use + house_use)
    print(f"agricultural: {ag_use}%, industrial: {ind_use}%, household: {house_use}%, other: {other_use}%")
    
    # 如果数据有微小误差导致other为负（例如总和略超100），修正为0
    if other_use < 0:
        other_use = 0

    # 绘图数据
    labels = ['Agricultural', 'Industrial', 'Household', 'Other']
    sizes = [ag_use, ind_use, house_use, other_use]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    explode = (0.05, 0.05, 0.05, 0.05)  # 稍微分离各部分

    plt.figure(figsize=(10, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140, textprops={'fontsize': 12})
    
    plt.title(f'Water Consumption Distribution in USA (2050)', fontsize=16)
    plt.axis('equal')  # 保证饼图是正圆
    plt.show()



if __name__ == '__main__':
    df_launch, df_water, df_pop, df_co2 = load_data()
    water_plot()
    # print(df_launch.tail())
    # sns.set_theme(style = "darkgrid")
    # sns.lineplot(x = 'Year', y = 'Freshwater use', data = df_water)
    # plt.show()

#     sns.lineplot(x = 'Year', y = 'Population', data = df_pop)
#     plt.show()

