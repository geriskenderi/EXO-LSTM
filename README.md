# Statistical forecasting with exogenous factors
#### A data-driven approach to sales forecasting, that uses exogenous temporal and weather encodings.

## Table of contents
- [Statistical forecasting with exogenous factors](#statistical-forecasting-with-exogenous-factors)
      - [A data-driven approach to sales forecasting, that uses exogenous temporal and weather encodings.](#a-data-driven-approach-to-sales-forecasting-that-uses-exogenous-temporal-and-weather-encodings)
  - [Table of contents](#table-of-contents)
  - [Getting started](#getting-started)
  - [Project structure](#project-structure)
  - [Results](#results)
    - [Weekly forecasting (long term)](#weekly-forecasting-long-term)
    - [Daily forecasting (short term)](#daily-forecasting-short-term)
  - [Future Work](#future-work)

## Getting started
You can find all the required packages in the requirements.txt file. First you will need to install those packages or make sure your environment has them in order to be able to run the code. A simple way to install all the packages is by running: `pip -r requirements.txt`

P.S: I recommend using Anacoda as an all-in-one scientific environment and just adding the missing packages.

## Project structure
There are 2 main directories:

1. **datasets**: As the name suggests, in this directory you can find all the preprocessed data of the sales, along with the datasets which are augmented with the exogenous data. You can find the univariate data by looking at the daily/ or weekly/ folders, or the multivariate data by looking at the daily_aug/ or weekly_aug/ folders.
2. **code**: This is the main directory which contains the code for forecasting, models and data analysis. You can see it is further divided into three subfolders:
   1. data_analysis: Inside you can find the analysis performed on the data of a single city (Milan). That analysis was then extended to all shops in malls and in streetshops and was used to derive the usefulness of using the weather and the time as valid exogenous data.
   2. forecasting: In this folder you will find the two notebooks which contain the daily and weekly forecasting
   3. helpers: In this folder you will find the code for some auxiliary functions, as well as the code for the LSTM neural network.

The code should be well documented and hopefully understandable.

## Results 

### Weekly forecasting (long term)
| Dataset | SARIMA RMSE | SARIMAX RMSE | MONO-LSTM RMSE	| EXO1-LSTM RMSE | EXO2-LSTM RMSE | EXO3-LSTM RMSE | SARIMA SMAPE | SARIMAX SMAPE | MONO-LSTM SMAPE | EXO1-LSTM SMAPE | EXO2-LSTM SMAPE	| EXO3-LSTM SMAPE|
|:--------|:------------|:-------------|:---------------|:---------------|:---------------|:---------------|:-------------|:--------------|:----------------|:----------------|:----------------|:---------------|
| Milan	| 299.299334 | 320.983730 | 262.790403 | 256.180035 | 350.757568 | 265.236377 | 6.099036 | 8.041129	| 6.881762 | 7.314275 | 9.065586 | 7.720466 |
| Rome	| 107.928035 | 103.974567 |	138.778368 | 95.183854	| 118.321224 | 113.994414 | 12.846811 | 12.232047 | 14.847312 | 10.074291 | 11.108825 | 11.626161 |
| Turin	| 137.123418 | 142.099703 |	153.850326 | 136.374610	| 178.712156 | 163.467810 | 11.014910 | 12.382435 | 13.518847 | 12.112468 | 14.149336 | 13.880192 |

### Daily forecasting (short term)
| Dataset |MONO-LSTM RMSE | EXO1-LSTM RMSE | EXO2-LSTM RMSE | EXO3-LSTM RMSE | MONO-LSTM SMAPE | EXO1-LSTM SMAPE | EXO2-LSTM SMAPE	| EXO3-LSTM SMAPE|
|:--------|:---------------|:---------------|:---------------|:---------------|:----------------|:----------------|:----------------|:---------------|
| Milan | 93.290671  | 99.372115 | 90.790401 | 87.846658 | 17.157380 | 19.367775 | 19.068441 | 16.592346
| Rome	| 105.050243 | 100.71988 | 99.258985 | 98.375294 | 23.446176 | 22.205415 | 22.901995 | 22.768316
| Turin	| 40.409859	 | 34.707807 | 38.578458 | 36.324167 | 20.168257 | 18.985997 | 19.076133 | 17.761742


## Future Work
1. **Different exogenous data**: Interesting applications would be the use of visual features or textual features deriving from social media.
2. **Other model architectures**: While this work focuses on singular approaches and mostly on the LSTM, it can be interesting to see how ensemble methods or different neural network architectures would perform.