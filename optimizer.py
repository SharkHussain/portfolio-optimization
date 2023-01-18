import pandas as pd
import random
import numpy as np

class optimizer:
    population_size = 0
    portfolio_size=0
    population=[]
    population_fitness=[]
    pareto=[]
    data=[]


    def __init__(self, data, population_size, portfolio_size):
        self.data=data
        self.population_size=population_size
        self.portfolio_size=portfolio_size

    def population_generation(self):
        population = self.population
        population_size=self.population_size
        data=self.data
        portfolio_size=self.portfolio_size

        for i in np.arange(population_size):
            indices = random.sample(range(0, data.shape[0]), portfolio_size)
            dm = data.iloc[indices]
            population.append(dm)
        self.population=population


    def population_fitness_calculation(self):

        population=self.population
        population_size=self.population_size
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


        dm = round(pd.concat([pd.Series(fitness_volume), pd.Series(fitness_time), pd.Series(fitness_complexity)], axis=1),2)
        dm.columns = ['volume', 'time', 'complexity']
        dm['complexity'] = dm['complexity'].round(1)
        self.population_fitness=dm

    def get_pareto_front(self):
        dm=self.population_fitness
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

        pareto = pd.concat([pd.Series(pareto_complexity), pd.Series(pareto_volume), pd.Series(pareto_time)], axis=1)
        self.pareto=pareto