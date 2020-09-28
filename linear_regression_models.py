# Carmen Pemsler
# Code for finding the Linear Regression of the Postretrofit data
# Postret data is from Yanke dataset, Outside Air Temperature (F) vs Hourly Electricity Use (kWh/h)
# Findng the daily averages of the OAT and the daily summation of Hourly Electricity Use
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
print(OAT_busdays)
# FOR ELEC USE VALUES
ElecUse_busdays = (drop_non_busdays(ElecUse_dailyTotals, holidays=None))
print(ElecUse_busdays)

# LOOKING FOR A CHANGE POINT THAT RESULTS IN OPTIMAL RMSE
df00 = OAT_busdays.reset_index()
df01 = ElecUse_busdays.reset_index()

# max and min of reevaluated data
min_temp = min(df00['OAT (F)'])
max_temp = max(df00['OAT (F)'])


# Stepping through every increment of temperature
def range_with_floats(start, stop, step):
    while stop > start:
        yield start
        start += step


for i in range_with_floats(min_temp, max_temp, 1):
    print(i)

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

# plt.show()


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


