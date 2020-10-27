# Carmen Pemsler
# Code for finding the Linear Regression of the Postretrofit data
# Postret data is from Yanke dataset, Outside Air Temperature (F) vs Hourly Electricity Use (kWh/h)
# Findng the daily averages of the OAT and the daily summation of Hourly Electricity Use
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg') #matplotlib backends - solves issue of SIGSEGV

# import numpy as np

# DAILY AVERAGES - Outside Air Temperature
df = pd.read_csv('/Users/carmenpemsler/Desktop/LEAF/Data/postret.csv', index_col=[0], parse_dates=[0], usecols=[0, 1])

OAT_dailyAvg = (df.resample('d').mean())

print(OAT_dailyAvg)

# DAILY SUMMATION - Hourly Electricity Use
df2 = pd.read_csv('/Users/carmenpemsler/Desktop/LEAF/Data/postret.csv', index_col=[0], parse_dates=[0], usecols=[0, 2])

ElecUse_dailyTotals = df2.groupby(pd.Grouper(freq='1D')).sum()

print(ElecUse_dailyTotals)

# __________2P MODEL____________________________________________
# ATTEMPT TO SEPARATE WEEKENDS AND HOLIDAYS

import workdays as wd


def drop_non_busdays(df, holidays=None):
    if holidays is None:
        holidays = []
    start_date = df.index.to_list()[0].date()
    end_date = df.index.to_list()[-1].date()

    start_wd = wd.workday(wd.workday(start_date, -1, holidays), 1, holidays)
    end_wd = wd.workday(wd.workday(end_date, 1, holidays), -1, holidays)

    b_days = [start_wd]
    while b_days[-1] < end_wd:
        b_days.append(wd.workday(b_days[-1], 1, holidays))

    valid = [i in b_days for i in df.index]
    return df[valid]


# FOR OAT VALUES
OAT_busdays = (drop_non_busdays(OAT_dailyAvg, holidays=None))
#print(OAT_busdays)
# FOR ELEC USE VALUES
ElecUse_busdays = (drop_non_busdays(ElecUse_dailyTotals, holidays=None))
#print(ElecUse_busdays)

# LOOKING FOR A CHANGE POINT THAT RESULTS IN OPTIMAL RMSE
df00 = OAT_busdays.reset_index()
df01 = ElecUse_busdays.reset_index()

df_for_temps = pd.concat([df00, df01], axis=1)
print('Data frame for FOR LOOP: ')
print(df_for_temps)

# max and min of reevaluated data
#min_temp = int(round(min(df_for_temps['OAT (F)'])))
#max_temp = int(round(max(df_for_temps['OAT (F)'])))

#index_of_maxTemp = df_for_temps['OAT (F)'].idxmax()
index_max= df_for_temps.loc[df_for_temps['OAT (F)'].idxmax()]
index_min= df_for_temps.loc[df_for_temps['OAT (F)'].idxmin()]



#Stepping through every increment of temperature
#def range_with_floats(start, stop, step):
#    while stop > start:
#        yield start
#        start += step


def frange(start, stop=None, step=None):
    # if stop and step argument is None set start=0.0 and step = 1.0
    start = float(start)
    if stop == None:
        stop = start + 0.0
        start = 0.0
    if step == None:
        step = 1.0

    count = 0
    while True:
        x = float(start + count * step)
        if step > 0 and x >= stop:
            break
        elif step < 0 and x <= stop:
            break
        yield x
        count += 1

#results = np.empty(len(temps)


import numpy
#from sklearn import linear_model

for temps in frange(index_min['OAT (F)'], index_max['OAT (F)'], 1.0):
    temperatures=round(temps)
    print(temperatures)


#rounded_temp_df = round(df_for_temps['OAT (F)'])

#for i in len((temperatures)):
#    subset = rounded_temp_df[rounded_temp_df["OAT (F)"] > temperatures[i]]
#    lin_reg = linear_model.LinearRegression()
#    lin_reg.fit(subset["OAT (F)"], subset["Elec. Use (kWh)"])





#temps = range(index_min, index_max, 1))
#print('temperature range: ', temps)

#for temps, df00 in temps:
#    print(temps)
#    print(df00)


#from sklearn import linear_model

#WORKING ON CODE HERE:_____________________________________________________________
#results = np.empty(len(str(temps)))
#from sklearn import linear_model

#for i in range(len(temps))
#    temperatures = temps[i]
#    dftemps_float = df_for_temps.astype(int)
#    subset = dftemps_float[dftemps_float["OAT (F)"] > temps(i)]
#    lin_reg = linear_model.LinearRegression()
#    lin_reg.fit(subset["OAT (F)"], subset["Elec. Use (kWh)"])

#results[temps[i]] = temps[i]

#results[rsme] = rsme from linear model




#    Subset = data[temps > temps[i]]
# busdays_data_subset = i[(i > 45)]

# Create linear model using that data
# Save temp and associated rsme in an array
# results$temp[i] = temp [i]
# results$rsme = rsms from linear model

# Best_bp = results$temp[max(results$rsme)]
# Return results


# ______________________________________________________________________

# PLOTTING LINEAR REGRESSION
fig = plt.figure(figsize=(10, 7))
sns.regplot(x=OAT_dailyAvg, y=ElecUse_dailyTotals, color='red', marker='+')

plt.xlabel('Outside Air Temperature (F)', size=18)
plt.ylabel('Daily Elec. Use (kWh)', size=18)


# printing the equation of the line

from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(OAT_dailyAvg, ElecUse_dailyTotals)
print('Slope:', regr.coef_[0])
print('Intercept:', regr.intercept_)

# SUBSETTING DATA FRAME

# first joining the dataframes and resetting the index
df3 = OAT_dailyAvg.reset_index()
df4 = ElecUse_dailyTotals.reset_index()
OAT_F = df3['OAT (F)']
Elecuse_kWh = df4['Elec. Use (kWh)']
dates = df3['Date']
result = pd.concat([dates, OAT_F, Elecuse_kWh], axis=1)
print(result)

# finding data that is above 45 deg. F and below 20000 kWh
# with their respective dates
data_subset = result[(result["OAT (F)"] > 45) & (result["Elec. Use (kWh)"] < 2000)]
print('Subset of data: ')
print(data_subset)

#plt.show()
