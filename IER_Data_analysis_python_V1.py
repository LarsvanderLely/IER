# -*- coding: utf-8 -*-
"""
Created on Sat May  8 10:50:45 2021

@author: larsv
"""
import tableone
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import pandas as pd
from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.compat import lzip
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.power import tt_ind_solve_power
import seaborn as sns


#&& Load data
data = np.zeros((193,103))
with open('IER_data.csv', mode='r') as infile:
    reader = csv.reader(infile)
    data = np.array(list(reader))



#%% BMI count 2019
for i in range(len(data)):
    if data[i,3] == 'NA':
        data[i,3] = math.nan
      
count_bmi_low_2019 = 0
for i in range(1,len(data)):
    data[i,3] = float(data[i,3])
    if (float(data[i,3]) <= 18.5 and float(data[i,3]) > 0 and data[i,1] == '2019'):
        count_bmi_low_2019 += 1
count_bmi_norm_2019 = 0
for i in range(1,len(data)):
    data[i,3] = float(data[i,3])
    if (float(data[i,3]) > 18.5 and float(data[i,3]) < 25 and data[i,1] == '2019'):
        count_bmi_norm_2019 += 1

count_bmi_high_2019 = 0
for i in range(1,len(data)):
    data[i,3] = float(data[i,3])
    if (float(data[i,3]) >= 25 and data[i,1] == '2019'):
        count_bmi_high_2019 += 1
        
count_bmi_NA_2019 = 0
for i in range(1,len(data)):
    data[i,3] = float(data[i,3])
    if (data[i,3] == 'nan'  and data[i,1] == '2019'):
        count_bmi_NA_2019 += 1
count_bmi_total_2019 = count_bmi_low_2019 + count_bmi_norm_2019 + count_bmi_high_2019 + count_bmi_NA_2019

perc_bmi_low_2019 = count_bmi_low_2019/count_bmi_total_2019 * 100
perc_bmi_norm_2019 = count_bmi_norm_2019/count_bmi_total_2019 * 100
perc_bmi_high_2019 = count_bmi_high_2019/count_bmi_total_2019 * 100
perc_bmi_NA_2019 = count_bmi_NA_2019/count_bmi_total_2019 * 100

bmi_array_2019 = np.zeros(len(data[1:,3]))
for i in range(len(data[1:,3])):
    if data[i+1,3] == 'nan'  and data[i,1] == '2019':
        bmi_array_2019[i] = math.nan
    elif data[i,1] == '2019':
        bmi_array_2019[i] = float(data[i+1,3])

bmi_array_2019[bmi_array_2019 == 0] = np.nan
std_bmi_2019 = np.nanstd(bmi_array_2019)
mean_bmi_2019 = np.nanmean(bmi_array_2019)

#%% BMI count 2020
for i in range(len(data)):
    if data[i,3] == 'NA':
        data[i,3] = math.nan
      
count_bmi_low_2020 = 0
for i in range(1,len(data)):
    data[i,3] = float(data[i,3])
    if (float(data[i,3]) <= 18.5 and float(data[i,3]) > 0 and data[i,1] == '2020'):
        count_bmi_low_2020 += 1
count_bmi_norm_2020 = 0
for i in range(1,len(data)):
    data[i,3] = float(data[i,3])
    if (float(data[i,3]) > 18.5 and float(data[i,3]) < 25 and data[i,1] == '2020'):
        count_bmi_norm_2020 += 1

count_bmi_high_2020 = 0
for i in range(1,len(data)):
    data[i,3] = float(data[i,3])
    if (float(data[i,3]) >= 25 and data[i,1] == '2020'):
        count_bmi_high_2020 += 1
        
count_bmi_NA_2020 = 0
for i in range(1,len(data)):
    data[i,3] = float(data[i,3])
    if (data[i,3] == 'nan'  and data[i,1] == '2020'):
        count_bmi_NA_2020 += 1
count_bmi_total_2020 = count_bmi_low_2020 + count_bmi_norm_2020 + count_bmi_high_2020 + count_bmi_NA_2020

perc_bmi_low_2020 = count_bmi_low_2020/count_bmi_total_2020 * 100
perc_bmi_norm_2020 = count_bmi_norm_2020/count_bmi_total_2020 * 100
perc_bmi_high_2020 = count_bmi_high_2020/count_bmi_total_2020 * 100
perc_bmi_NA_2020 = count_bmi_NA_2020/count_bmi_total_2020 * 100

bmi_array_2020 = np.zeros(len(data[1:,3]))
for i in range(len(data[1:,3])):
    if data[i+1,3] == 'nan'  and data[i,1] == '2020':
        bmi_array_2020[i] = math.nan
    elif data[i,1] == '2020':
        bmi_array_2020[i] = float(data[i+1,3])
    
bmi_array_2020[bmi_array_2020 == 0] = np.nan
std_bmi_2020 = np.nanstd(bmi_array_2020)
mean_bmi_2020 = np.nanmean(bmi_array_2020)

#%% Gender count 2019
count_gen_female_2019 = 0
for i in range(1,len(data)):
    if (data[i,2] == 'Female' and data[i,1] == '2019'):
        count_gen_female_2019 += 1

count_gen_male_2019 = 0
for i in range(1,len(data)):
    if (data[i,2] == 'Male' and data[i,1] == '2019'):
        count_gen_male_2019 += 1

count_gen_NA_2019 = 0
for i in range(1,len(data)):
    if (data[i,2] == 'NA' and data[i,1] == '2019'):
        count_gen_NA_2019 += 1
        
count_gen_total_2019 = count_gen_NA_2019 + count_gen_female_2019 + count_gen_male_2019

perc_gen_female_2019 = count_gen_female_2019/count_gen_total_2019 * 100
perc_gen_male_2019 = count_gen_male_2019/count_gen_total_2019 * 100
perc_gen_NA_2019 = count_gen_NA_2019/count_gen_total_2019 * 100

#%% Gender count 2020
count_gen_female_2020 = 0
for i in range(1,len(data)):
    if (data[i,2] == 'Female' and data[i,1] == '2020'):
        count_gen_female_2020 += 1

count_gen_male_2020 = 0
for i in range(1,len(data)):
    if (data[i,2] == 'Male' and data[i,1] == '2020'):
        count_gen_male_2020 += 1

count_gen_NA_2020 = 0
for i in range(1,len(data)):
    if (data[i,2] == 'NA' and data[i,1] == '2020'):
        count_gen_NA_2020 += 1
        
count_gen_total_2020 = count_gen_NA_2020 + count_gen_female_2020 + count_gen_male_2020

perc_gen_female_2020 = count_gen_female_2020/count_gen_total_2020 * 100
perc_gen_male_2020 = count_gen_male_2020/count_gen_total_2020 * 100
perc_gen_NA_2020 = count_gen_NA_2020/count_gen_total_2020 * 100


#%% Living count 2019
count_liv_move_2019 = 0
for i in range(1,len(data)):
    if (data[i,4] == 'Moved_out' and data[i,1] == '2019'):
        count_liv_move_2019 += 1

count_liv_par_2019 = 0
for i in range(1,len(data)):
    if (data[i,4] == 'Living_with_parents' and data[i,1] == '2019'):
        count_liv_par_2019 += 1

        
count_liv_total_2019 = count_liv_par_2019 + count_liv_move_2019

perc_liv_par_2019 = count_liv_par_2019/count_liv_total_2019 * 100
perc_liv_move_2019 = count_liv_move_2019/count_liv_total_2019 * 100

#%% Living count 2020
count_liv_move_2020 = 0
for i in range(1,len(data)):
    if (data[i,4] == 'Moved_out' and data[i,1] == '2020'):
        count_liv_move_2020 += 1

count_liv_par_2020 = 0
for i in range(1,len(data)):
    if (data[i,4] == 'Living_with_parents' and data[i,1] == '2020'):
        count_liv_par_2020 += 1

        
count_liv_total_2020 = count_liv_par_2020 + count_liv_move_2020

perc_liv_par_2020 = count_liv_par_2020/count_liv_total_2020 * 100
perc_liv_move_2020 = count_liv_move_2020/count_liv_total_2020 * 100

#%% Data analysis
#Calculate average calories burned per day per participants
avg_cal_day = np.zeros((len(data)-1))
data[data == 'NA'] = math.nan #Convert all the 'NA' to zero

for i in range(len(avg_cal_day)):
    average_list = []
    for j in range(7):
        if data[i+1, 96+j] == 'Yes' and float(data[i+1, 45+j*4]) < 2000: # Only include the days on which the person carried the device and whose calories burned are below 1500 
        #if data[i+1, 96+j] == 'Yes': # Only include the days on which the person carried the device and whose calories burned are below 1500 
            average_list.append(float(data[i+1, 45+j*4]))
        #average_list = [float(data[i+1, 45]), float(data[i+1, 49]), float(data[i+1, 53]), float(data[i+1, 57]), float(data[i+1, 61]), float(data[i+1, 65]), float(data[i+1, 69])]  
    avg_cal_day[i]  = np.nanmean(average_list) #average calories burned per day
#avg_cal_day[avg_cal_day > 2000]  = math.nan#Remove >2000 calories (which is impossible)


bmi_groups_list = []
for i in range(len(data)-1):
    if float(data[i+1,3]) <= 18.5:
        bmi_groups_list.append("Underweight")
    elif float(data[i+1,3]) >18.5 and float(data[i+1,3]) < 25:
        bmi_groups_list.append("Normal")
    elif float(data[i+1,3]) >= 25:
        bmi_groups_list.append("Overweight")
    else:
        bmi_groups_list.append(math.nan)

d = {'ID': data[1:,0].astype(np.float), 'Year': data[1:,1], 'BMI_Group': bmi_groups_list, 'Avg_cal_day': avg_cal_day} #Make dataframe of the relevant data
df = pd.DataFrame(data=d)

df.head()
model = ols("Avg_cal_day ~ BMI_Group + Year", data=df)
results = model.fit()
df_model2 = ols("Avg_cal_day ~ BMI_Group*Year", data=df).fit()

print(sm.stats.anova_lm(results, df_model2))
print('-----------')
print(results.summary())

fig, ax = plt.subplots(figsize=(6, 6))
fig = interaction_plot(x=df['Year'], trace=df['BMI_Group'], response=df['Avg_cal_day'],colors=['red', 'blue', 'green'], markers=['D', '^', 's'], ms=10, ax=ax)

#fig = sm.graphics.plot_partregress_grid(df_model)
#fig.tight_layout(pad=1.0)

#Plot checker
fig2, ax2 = plt.subplots(figsize=(6, 6))
#fig2 = 
def LinearRegModel(model, year = 0, Overweight = 0, Underweight = 0):
    intercept = model.params[0]
    over_coef = model.params[1]
    under_coef = model.params[2]
    year_coef = model.params[3]
    return (intercept + over_coef * Overweight + under_coef * Underweight + year_coef * year)

plot_data = np.zeros((len(data)-1, 5)) #ID, year, bmi overweight, bmi underweight, predicted avg_cal_day
for i in range(len(plot_data)):
    plot_data[i,0] = d['ID'][i]
    if d['Year'][i] == '2020':
        plot_data[i,1] = 1
        
    if d['BMI_Group'][i] == 'Overweight':
        plot_data[i,2] = 1
    elif d['BMI_Group'][i] == 'Underweight':
        plot_data[i,3] = 1
    elif d['BMI_Group'][i] == 'Normal':
        plot_data[i,2] = 0
        plot_data[i,3] = 0
    else:
        plot_data[i,2] = math.nan
        plot_data[i,3] = math.nan
    plot_data[i,4] = LinearRegModel(results, year = plot_data[i,1], Overweight =  plot_data[i,2], Underweight =  plot_data[i,3])


avg_cal_day_2019_normal = LinearRegModel(results, year = 0, Overweight =  0, Underweight =  0)
avg_cal_day_2020_normal = LinearRegModel(results, year = 1, Overweight =  0, Underweight =  0)

avg_cal_day_2019_overweight = LinearRegModel(results, year = 0, Overweight =  1, Underweight =  0)
avg_cal_day_2020_overweight = LinearRegModel(results, year = 1, Overweight =  1, Underweight =  0)

avg_cal_day_2019_underweight = LinearRegModel(results, year = 0, Overweight =  0, Underweight =  1)
avg_cal_day_2020_underweight = LinearRegModel(results, year = 1, Overweight =  0, Underweight =  1)

fig2 = plt.plot(['2019', '2020'], [avg_cal_day_2019_normal, avg_cal_day_2020_normal], marker = 'o', label = 'Normal')    
fig2 = plt.plot(['2019', '2020'], [avg_cal_day_2019_overweight, avg_cal_day_2020_overweight], marker = 'o', label = 'Overweight')    
fig2 = plt.plot(['2019', '2020'], [avg_cal_day_2019_underweight, avg_cal_day_2020_underweight], marker = 'o', label = 'Underweight')
plt.legend(title = 'BMI Groups')
plt.xlabel('Year')
plt.ylabel('mean of avg_cal_day (calories)')     

#%% Power analysis
effect_size= tt_ind_solve_power(effect_size=0.3, nobs1 = None, alpha=0.05, power=0.8, ratio=1.0, alternative='two-sided')
print(effect_size)

#%% Analysis of the model
from scipy import stats

def abline(slope, intercept):
     """Plot a line from slope and intercept, borrowed from https://stackoverflow.com/questions/7941226/how-to-add-line-based-on-slope-and-intercept-in-matplotlib"""
     axes = plt.gca()
     x_vals = np.array(axes.get_xlim())
     y_vals = intercept + slope * x_vals
     plt.plot(x_vals, y_vals, '--')


df = df.dropna()
y = df['Avg_cal_day']


y_hat = model.predict(results.params)
plt.plot(y_hat,y,'o')
plt.xlabel('Predicted')#,color='white')
plt.ylabel('Actual')#,color='white')
plt.title('Predicted vs. Actual: Visual Linearity Test')#,color='white')
plt.tick_params(axis='x', colors='white')
plt.tick_params(axis='y', colors='white')
abline(1,0)
plt.show()


plt.plot(y_hat,y-y_hat,'o')
plt.xlabel(r'Predicted y^hat')
plt.ylabel(r'Residuals y-y^hat')
plt.title('Predicted vs Residuals')
plt.tick_params(axis='x')
plt.tick_params(axis='y')
plt.show()

plt.plot(df['Year'],y-y_hat,'o')
plt.xlabel(r'Year')
plt.ylabel(r'Residuals y-y^hat')
plt.title('Predicted vs Residuals')
plt.tick_params(axis='x')
plt.tick_params(axis='y')
plt.show()

plt.plot(df['BMI_Group'],y-y_hat,'o')
plt.xlabel(r'BMI_Group')
plt.ylabel(r'Residuals y-y^hat')
plt.title('Predicted vs Residuals')
plt.tick_params(axis='x')
plt.tick_params(axis='y')
plt.show()
