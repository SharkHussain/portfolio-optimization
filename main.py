#Complexity is a metric that looks at the total number and interdependencies between the products in a portfolio
# I need costs, I need running times



import pandas as pd
import random
import numpy as np



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






def run_optimization(data, population_size, portfolio_size):

    population = []

    for i in np.arange(population_size):
        indices = random.sample(range(0, data.shape[0]), portfolio_size)
        dm = data.iloc[indices]
        population.append(dm)

    fitness_volume = []
    fitness_time = []
    fitness_complexity = []

    for i in np.arange(population_size):
        volume = population[i]['volume'].sum()
        time = population[i]['mean'].sum()
        complexity = 10000 / ((100 * (population[i]['volume'] / population[i]['volume'].sum())) ** 2).sum()

        fitness_volume.append(volume)
        fitness_time.append(time)
        fitness_complexity.append(complexity)

    import matplotlib.pyplot as plt

    plt.figure("Welcome to figure 1")
    plt.title('Plant Name')
    plt.plot(fitness_complexity, fitness_volume, 'o')
    plt.show()


    # plt.figure("Welcome to figure 2")
    # plt.plot(fitness_complexity, fitness_time, 'o')
    # plt.show()

    dm = round(pd.concat([pd.Series(fitness_volume), pd.Series(fitness_time), pd.Series(fitness_complexity)], axis=1),
               2)
    dm.columns = ['volume', 'time', 'complexity']
    dm['complexity'] = dm['complexity'].round(1)

    complexity_numbers = dm['complexity'].unique()
    pareto_volume = []
    pareto_time = []
    pareto_complexity = []

    for complexity_no in complexity_numbers:
        da = dm.loc[dm['complexity'] == complexity_no]
        da = da.sort_values('volume', ascending=False)
        v = da['volume'].iloc[0]
        t = da['time'].iloc[0]
        c = da['complexity'].iloc[0]

        pareto_volume.append(v)
        pareto_complexity.append(c)
        pareto_time.append(t)

        print(c)

    pareto=pd.concat([pd.Series(pareto_complexity), pd.Series(pareto_volume), pd.Series(pareto_time)],axis=1)
    plt.plot(pareto_complexity, pareto_volume, 'o')

    return(pareto)




data = get_data('forPythonv2.csv', plant_id=298)
data.to_csv('data_for_optimization.csv')

population_size = 10000
portfolio_size = 20

pareto=run_optimization(data, population_size, portfolio_size)





###################################################################

population_size = 10000
portfolio_size = 25
population=[]

for i in np.arange(population_size):
    indices=random.sample(range(0, data.shape[0]), portfolio_size)
    dm=data.iloc[indices]
    population.append(dm)


fitness_volume=[]
fitness_time=[]
fitness_complexity=[]

for i in np.arange(population_size):
    volume=population[i]['volume'].sum()
    time=population[i]['mean'].sum()
    complexity=10000/((100*(population[i]['volume']/population[i]['volume'].sum()))**2).sum()


    fitness_volume.append(volume)
    fitness_time.append(time)
    fitness_complexity.append(complexity)



import matplotlib.pyplot as plt

plt.figure("Welcome to figure 1")
plt.plot(fitness_complexity, fitness_volume, 'o')
plt.show()


#Try HHI index (Herfindahl-Hirschman Index) and Simpson's Index for complexity and join it with model complexity formula.
#plt.figure("Welcome to figure 2")
#plt.plot(fitness_complexity, fitness_time, 'o')
#plt.show()



dm = round(pd.concat([pd.Series(fitness_volume), pd.Series(fitness_time), pd.Series(fitness_complexity)], axis = 1),2)
dm.columns = ['volume', 'time', 'complexity']
dm['complexity']=dm['complexity'].round(1)


complexity_numbers = dm['complexity'].unique()
pareto_volume=[]
pareto_time=[]
pareto_complexity=[]

for complexity_no in complexity_numbers:

    da=dm.loc[dm['complexity']==complexity_no]
    da=da.sort_values('volume', ascending=False)
    v=da['volume'].iloc[0]
    t=da['time'].iloc[0]
    c=da['complexity'].iloc[0]

    pareto_volume.append(v)
    pareto_complexity.append(c)
    pareto_time.append(t)

    print(c)



plt.plot(pareto_complexity, pareto_volume, 'o')