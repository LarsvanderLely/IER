# The effects of the Covid-19 pandemic and BMI on the average daily burned calories of students: Data analysis
The goal of this project is to analyse the data captured by a fitness tracker. The following question can be answered with this data analysis:
_What is the difference in change in amount of physical activity, measured by average daily burned calories, due to the Covid-19 pandemic between students of different BMI?_
This project is performed for the course ME41125: Introduction To Engineering Research at the TU Delft.

## Getting Started

### Prerequisites
To run the data analysis, python is required with the following libraries:
1. Numpy
2. Statsmodel
3. pandas
4. tableone
5. matplotlib
6. csv
7. math
8. seaborn
9. Scipy

### Data
The python file only contains an data analysis script, and the user has to provide the data (in csv format) himself/herself.
The most important data columns that must be present in the data are: 
- ID
- Year
- BMI
- Calories burned daily
- Tracker worn 'yes' or 'no'.

### Used tests
- sm.stats.anova_lm(model1, model2), to compare to linear regression models. This helps selecting an appropriate model by concluding whether there exists a significant difference between the two models (model1 and model2).
- model.summary(), to find the coefficients and the statistical significants of the linear regression model.

### Plots
The following plots are generated to be analysed±
- Effect plot of the avg_cal_day, BMI groups and Year, to answer the research question
- Predicted versus Actual avg_cal_day, to see if the linearity assumption for the model holds
- Residuals versus predicted avg_cal_day, to see if the homogenity assumption for the model holds
- Residuals versus Year avg_cal_day, to see if the independence of the residuals assumption for the mode holds
- Residuals versus BMI avg_cal_day, to see if the independence of the residuals assumption for the mode holds

## Authors
Lars van der Lely, TU Delft
