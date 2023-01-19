#Complexity is a metric that looks at the total number and interdependencies between the products in a portfolio
# I need costs, I need running times
#The time variable at present is a basic sum, it needs to be improved by using the exact change over time



import pandas as pd
import random
import numpy as np
from matplotlib import pyplot as plt

from importlib import reload
import optimizer
reload(optimizer)
from optimizer import optimizer




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

data = get_data('forPythonv2.csv', plant_id=298)
data.to_csv('data_for_optimization.csv')

population_size = 10000
portfolio_size = 25

opt=optimizer(data, population_size, portfolio_size)
pareto=opt.optimize()

#pareto=pareto[pareto['complexity']<8]

pareto=pareto.sort_values('complexity')
plt.figure("Portfolio - Pareto Front")
plt.title('Plant ID = 298, Portfolio Size = 25')
plt.plot(opt.population_fitness['complexity'],opt.population_fitness['volume'], 'o' )
plt.plot(pareto['complexity'],pareto['volume'], 'o' )
plt.xlabel('Complexity')
plt.ylabel('Volume')
plt.show()


pareto=pareto.sort_values('complexity')
plt.figure("Welcome to figure 2")
plt.title('Plant ID = 298')
plt.plot(pareto['complexity'],pareto['time'])
plt.show()


a=opt.population_fitness
a=a.sort_values('complexity')

plt.figure("Welcome to figure 3")
plt.title('Plant ID = 298')
plt.plot(a['time'],a['volume'] , 'o')
plt.show()

plt.figure("Welcome to figure 4")
plt.plot(a['complexity'], a['time'])
plt.show()




from sklearn.linear_model import LinearRegression

X=np.array(a['complexity'])
Y=np.array(a['time'])

X=X.reshape(-1,1)
Y=Y.reshape(-1,1)

model=LinearRegression()
model.fit(X,Y)
print(model.coef_)

y_=model.predict(X)


plt.figure("Complexity vs. Time")
plt.title('Complexity vs. Time, 30min on average')
plt.plot(X,Y,'o')
plt.plot(X,y_)
plt.xlabel('Complexity')
plt.ylabel('Volume')
plt.show()

