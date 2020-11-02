# Statistical forecasting with exogenous factors
#### A data-driven approach to sales forecasting, that uses exogenous temporal and weather encodings.


## Table of contents
  - [General explanation](#general-explanation)
  - [Getting started](#getting-started)
  - [Project structure](#project-structure)
  - [Results](#results)
    - [Weekly forecasting (long term)](#weekly-forecasting-long-term)
      - [Weekly average over all datasets](#weekly-average-over-all-datasets)
    - [Daily forecasting (short term)](#daily-forecasting-short-term)
      - [Daily average over all datasets](#daily-average-over-all-datasets)
  - [Future Work](#future-work)


## General explanation
As mentioned in the title, this repository contains the code for a data-driven approach to sales forecasting, where a preliminary data analysis was done in order to statistically check for the validity of the hypothesis that the exogenous factors, time and weather, do affect the total sold amount of products. Based on this analysis, different forecasting models were used, with a particular focus on the LSTM RNN. That is because of the ability of neural networks to learn non-linear relationships in the data, which can make a difference when dealing with multivariate time series forecasting (as in our case). You can see the results [here](#results). In general, it seems to be the case then when using valid exogenous data, we can help a forecasting model understand not just a vertical,sequential flow but also a horizontal, spatial one. In simpler words, *adding valid exogenous data* can help the forecasting model better understand the phenomenon we are trying to predict. This approach was used for both long-term forecasting (weeks over seasons) and short-term forecasting (days over weeks), giving satisfactory results in both cases.


## Getting started
You can find all the python packages that haven been used in this project inside the requirements.txt file. First you will need to install those packages or make sure your current environment has them in order to be able to run the code. A simple way to install all the packages is by running: `pip -r requirements.txt`

P.S: I recommend using Anacoda as an all-in-one scientific environment and just adding the missing packages.


## Project structure
There are 2 main directories:
1. **datasets**: As the name suggests, in this directory you can find all the preprocessed data of the sales, along with the datasets which are augmented with the exogenous data. You can find the univariate datasets by looking at the daily/ or weekly/ folders, or the multivariate datasets by looking at the daily_aug/ or weekly_aug/ folders.
2. **code**: This is the main directory which contains the code for forecasting, models and data analysis. It is further divided into three subfolders:
   1. data_analysis: Inside you can find the analysis performed on the data of a single city (Milan). That analysis was then extended to all shops inside malls and streetshops. By checking both visually and statistically (hypothesis testing) it was derived that  the weather and the time are usefulness and valid exogenous data.
   2. forecasting: In this folder you will find the two notebooks which contain the code for both daily and weekly forecasting.
   3. helpers: In this folder you will find the code for some auxiliary functions, as well as the code for the LSTM neural network.

The code itself is well documented and should hopefully be understandable.

## Results 

### Weekly forecasting (long term)
| Dataset | SARIMA RMSE | SARIMAX RMSE | MONO-LSTM RMSE	| EXO1-LSTM RMSE | EXO2-LSTM RMSE | EXO3-LSTM RMSE | SARIMA SMAPE | SARIMAX SMAPE | MONO-LSTM SMAPE | EXO1-LSTM SMAPE | EXO2-LSTM SMAPE	| EXO3-LSTM SMAPE|
|:--------|:------------|:-------------|:---------------|:---------------|:---------------|:---------------|:-------------|:--------------|:----------------|:----------------|:----------------|:---------------|
| Milan | 299.299334 | 320.983730 | 269.000266 | **259.300154** |	294.289433 | 276.278388 | **6.099036** | 8.041129 | 7.412993 | 7.515069 | 8.366195 | 8.250909 |
| Rome | 107.928035 | 103.974567 | 149.296986 | **95.339082** | 127.586807 | 117.149285 | 12.846811 | 12.232047 | 16.026479 | **10.029953** | 12.068225 | 12.251298 |
| Turin | **137.123418** | 142.099703 | 146.110333 | 137.757753 | 174.867309 | 163.300861 | **11.014910** | 12.382435 | 13.085100 | 12.106654 | 13.937898 | 13.659325 |

#### Weekly average over all datasets
| SARIMA RMSE | SARIMAX RMSE | MONO-LSTM RMSE	| EXO1-LSTM RMSE | EXO2-LSTM RMSE | EXO3-LSTM RMSE | SARIMA SMAPE | SARIMAX SMAPE | MONO-LSTM SMAPE | EXO1-LSTM SMAPE | EXO2-LSTM SMAPE	| EXO3-LSTM SMAPE|
|:------------|:-------------|:---------------|:---------------|:---------------|:---------------|:-------------|:--------------|:----------------|:----------------|:----------------|:---------------|
| 181.450262 | 189.019333 | 188.135862 | **164.132330** | 198.914516 | 185.576178 | 9.986919 | 10.885204 | 12.174857 | **9.883892** | 11.457439 | 11.387177

### Daily forecasting (short term)
| Dataset |MONO-LSTM RMSE | EXO1-LSTM RMSE | EXO2-LSTM RMSE | EXO3-LSTM RMSE | MONO-LSTM SMAPE | EXO1-LSTM SMAPE | EXO2-LSTM SMAPE	| EXO3-LSTM SMAPE|
|:--------|:---------------|:---------------|:---------------|:---------------|:----------------|:----------------|:----------------|:---------------|
| Milan | 92.681814  | 97.419602 | 92.756764  | **87.843425**  | 17.082440 | 19.892359 | 20.611555 | **16.801701** |
| Rome  | 103.320297 | 101.16842 | 107.342307 | **99.044623**  | 23.468583 | **21.877664** | 22.764811 | 22.043665 |
| Turin | 40.815775  | **34.641631** | 35.821523  | 36.963587	 | 20.751887 | 18.803312 | 18.672907 | **17.939686** |

#### Daily average over all datasets
|MONO-LSTM RMSE | EXO1-LSTM RMSE | EXO2-LSTM RMSE | EXO3-LSTM RMSE | MONO-LSTM SMAPE | EXO1-LSTM SMAPE | EXO2-LSTM SMAPE	| EXO3-LSTM SMAPE|
|:---------------|:---------------|:---------------|:---------------|:----------------|:----------------|:----------------|:---------------|
| 78.939295 | 77.743218 | 78.640198 | **74.617212** | 20.434304 | 20.191111 | 20.683091 | **18.928351** |

## Future Work
1. **Different exogenous data**: Very interesting ones would be the use of visual features or textual features deriving from social media. This would be particularly useful for e-commerce.
2. **Other model architectures**: While this work focuses on "singular" approaches and mostly on the LSTM, it can be interesting to see how ensemble methods or different neural network architectures would perform. 
3. **Automatic feature extraction**: I believe that another great thing to consider would be automating this process by having a way to extract meaningful features from the multivariate time series. This way the whole data analysis could be bypassed in a certain sense. Multi-modal forecasting is a big trend right now in forecasting research.
