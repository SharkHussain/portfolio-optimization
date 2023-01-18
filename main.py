#Complexity is a metric that looks at the total number and interdependencies between the products in a portfolio
# I need costs, I need running times



import pandas as pd



def get_data(fileName, plant_id):
    df = pd.read_csv(fileName)
    df = df.loc[df['Plant'] == plant_id]

    dt = pd.DataFrame()
    dt['ID'] = df['Setup Grp. From'].unique()

    for item in df['Setup Grp. From'].unique():
        average = df.loc[df['Setup Grp. From'] == item, 'Duration'].mean()
        median = df.loc[df['Setup Grp. From'] == item, 'Duration'].median()

        dt.loc[dt['ID'] == item, 'Plant Name'] = df.loc[df['Setup Grp. From'] == item, 'Plant Name'].unique()[0]
        dt.loc[dt['ID'] == item, 'Name'] = df.loc[df['Setup Grp. From'] == item, 'SKU_from'].unique()[0]
        dt.loc[dt['ID'] == item, 'volume'] = df.loc[df['Setup Grp. From'] == item, 'volume_SKU_from'].mean()
        dt.loc[dt['ID'] == item, 'mean'] = average
        dt.loc[dt['ID'] == item, 'median'] = median
        dt.loc[dt['ID'] == item, 'setupMin'] = df.loc[df['Setup Grp. From'] == item, 'SetupMin'].median()
        dt.loc[dt['ID'] == item, 'setupMax'] = df.loc[df['Setup Grp. From'] == item, 'SetupMax'].median()
        dt.loc[dt['ID'] == item, 'syrup_components'] = df.loc[
        df['Setup Grp. From'] == item, 'Syrup #Components'].median()
        dt.loc[dt['ID'] == item, 'Name'] = df.loc[df['Setup Grp. From'] == item, 'SKU_from'].unique()[0]

        print(item)
    return(dt)


import random
import numpy as np

data = get_data('forPythonv2.csv', plant_id=298)
data.to_csv('data_for_optimization.csv')


population_size = 1000
portfolio_size = 15
population=[]

for i in np.arange(population_size):
    indices=random.sample(range(0, data.shape[0]), portfolio_size)
    dm=data.iloc[indices]
    population.append(dm)


fitness_volume = []
fitness_time = []

for i in np.arange(population_size):
    volume=population[i]['volume'].sum()
    time=population[i]['mean'].sum()

    fitness_volume.append(volume)
    fitness_time.append(time)



import matplotlib.pyplot as plt

plt.plot(fitness_time, fitness_volume, 'o')
plt.show()



pd.concat([pd.Series(fitness_volume), pd.Series(fitness_time)], axis = 1)


#Try HHI index (Herfindahl-Hirschman Index) and Simpson's Index for complexity and join it with model complexity formula.

