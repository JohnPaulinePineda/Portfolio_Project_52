***
# Supervised Learning : Implementing Shapley Additive Explanations for Interpreting Feature Contributions in Penalized Cox Regression

***
### John Pauline Pineda <br> <br> *July 28, 2024*
***

* [**1. Table of Contents**](#TOC)
    * [1.1 Data Background](#1.1)
    * [1.2 Data Description](#1.2)
    * [1.3 Data Quality Assessment](#1.3)
    * [1.4 Data Preprocessing](#1.4)
        * [1.4.1 Data Cleaning](#1.4.1)
        * [1.4.2 Missing Data Imputation](#1.4.2)
        * [1.4.3 Outlier Treatment](#1.4.3)
        * [1.4.4 Collinearity](#1.4.4)
        * [1.4.5 Shape Transformation](#1.4.5)
        * [1.4.6 Centering and Scaling](#1.4.6)
        * [1.4.7 Data Encoding](#1.4.7)
        * [1.4.8 Preprocessed Data Description](#1.4.8)
    * [1.5 Data Exploration](#1.5)
        * [1.5.1 Exploratory Data Analysis](#1.5.1)
        * [1.5.2 Hypothesis Testing](#1.5.2)
    * [1.6 Predictive Model Development](#1.6)
        * [1.6.1 Premodelling Data Description](#1.6.1)
        * [1.6.2 Cox Regression with No Penalty](#1.6.2)
        * [1.6.3 Cox Regression With Full L2 Penalty](#1.6.3)
        * [1.6.4 Cox Regression With Full L1 Penalty](#1.6.4)
        * [1.6.5 Cox Regression With Equal L1|L2 Penalty](#1.6.5)
        * [1.6.6 Cox Regression With Predominantly L1-Weighted|L2 Penalty](#1.6.6)
        * [1.6.7 Cox Regression With Predominantly L2-Weighted|L1 Penalty](#1.6.7)
    * [1.7 Consolidated Findings](#1.7)
* [**2. Summary**](#Summary)   
* [**3. References**](#References)

***

# 1. Table of Contents <a class="anchor" id="TOC"></a>

This project implements the **Cox Proportional Hazards Regression** algorithm with **No Penalty**, **Full L1 Penalty**, **Full L2 Penalty**, **Equal L1|L2 Penalty** and **Weighted L1|L2 Penalty** using various helpful packages in <mark style="background-color: #CCECFF"><b>Python</b></mark> to estimate the survival probabilities of right-censored survival time and status responses. The resulting predictions derived from the candidate models were evaluated in terms of their discrimination power using the Harrel's concordance index metric. Additionally, feature contributions were estimated using **Shapley Additive Explanations**. All results were consolidated in a [<span style="color: #FF0000"><b>Summary</b></span>](#Summary) presented at the end of the document. 

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Survival Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) refer to statistical methods used to analyze survival data, accounting for censored observations. These models aim to describe the relationship between survival time and one or more predictor variables, and to estimate the survival function and hazard function. Survival models are essential for understanding the factors that influence time-to-event data, allowing for predictions and comparisons between different groups or treatment effects. They are widely used in clinical trials, reliability engineering, and other areas where time-to-event data is prevalent.

[Shapley Additive Explanations](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) are based on Shapley values developed in the cooperative game theory. The process involves explaining a prediction by assuming that each explanatory variable for an instance is a player in a game where the prediction is the payout. The game is the prediction task for a single instance of the data set. The gain is the actual prediction for this instance minus the average prediction for all instances. The players are the explanatory variable values of the instance that collaborate to receive the gain (predict a certain value). The determined value is the average marginal contribution of an explanatory variable across all possible coalitions.

## 1.1. Data Background <a class="anchor" id="1.1"></a>

An open [Liver Cirrhosis Dataset](https://www.kaggle.com/code/arjunbhaybhang/liver-cirrhosis-prediction-with-xgboost-eda) from [Kaggle](https://www.kaggle.com/) (with all credits attributed to [Arjun Bhaybhang](https://www.kaggle.com/arjunbhaybhang)) was used for the analysis as consolidated from the following primary sources: 
1. Reference Book entitled **Counting Processes and Survival Analysis** from [Wiley](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118150672)
2. Research Paper entitled **Efficacy of Liver Transplantation in Patients with Primary Biliary Cirrhosis** from the [New England Journal of Medicine](https://www.nejm.org/doi/abs/10.1056/NEJM198906293202602)
3. Research Paper entitled **Prognosis in Primary Biliary Cirrhosis: Model for Decision Making** from the [Hepatology](https://aasldpubs.onlinelibrary.wiley.com/doi/10.1002/hep.1840100102)

This study hypothesized that the evaluated drug, liver profile test biomarkers and various clinicopathological characteristics influence liver cirrhosis survival between patients.

The event status and survival duration variables for the study are:
* <span style="color: #FF0000">Status</span> - Status of the patient (C, censored | CL, censored due to liver transplant | D, death)
* <span style="color: #FF0000">N_Days</span> - Number of days between registration and the earlier of death, transplantation, or study analysis time (1986)

The predictor variables for the study are:
* <span style="color: #FF0000">Drug</span> - Type of administered drug to the patient (D-Penicillamine | Placebo)
* <span style="color: #FF0000">Age</span> - Patient's age (Days)
* <span style="color: #FF0000">Sex</span> - Patient's sex (Male | Female)
* <span style="color: #FF0000">Ascites</span> - Presence of ascites (Yes | No)
* <span style="color: #FF0000">Hepatomegaly</span> - Presence of hepatomegaly (Yes | No)
* <span style="color: #FF0000">Spiders</span> - Presence of spiders (Yes | No)
* <span style="color: #FF0000">Edema</span> - Presence of edema ( N, No edema and no diuretic therapy for edema | S, Edema present without diuretics or edema resolved by diuretics) | Y, Edema despite diuretic therapy)
* <span style="color: #FF0000">Bilirubin</span> - Serum bilirubin (mg/dl)
* <span style="color: #FF0000">Cholesterol</span> - Serum cholesterol (mg/dl)
* <span style="color: #FF0000">Albumin</span> - Albumin (gm/dl)
* <span style="color: #FF0000">Copper</span> - Urine copper (ug/day)
* <span style="color: #FF0000">Alk_Phos</span> - Alkaline phosphatase (U/liter)
* <span style="color: #FF0000">SGOT</span> - Serum glutamic-oxaloacetic transaminase (U/ml)
* <span style="color: #FF0000">Triglycerides</span> - Triglicerides (mg/dl)
* <span style="color: #FF0000">Platelets</span> - Platelets (cubic ml/1000)
* <span style="color: #FF0000">Prothrombin</span> - Prothrombin time (seconds)
* <span style="color: #FF0000">Stage</span> - Histologic stage of disease (Stage I | Stage II | Stage III | Stage IV)


## 1.2. Data Description <a class="anchor" id="1.2"></a>

1. The dataset is comprised of:
    * **418 rows** (observations)
    * **20 columns** (variables)
        * **1/20 metadata** (object)
            * <span style="color: #FF0000">ID</span>
        * **2/20 event | duration** (object | numeric)
             * <span style="color: #FF0000">Status</span>
             * <span style="color: #FF0000">N_Days</span>
        * **10/20 predictor** (numeric)
             * <span style="color: #FF0000">Age</span>
             * <span style="color: #FF0000">Bilirubin</span>
             * <span style="color: #FF0000">Cholesterol</span>
             * <span style="color: #FF0000">Albumin</span>
             * <span style="color: #FF0000">Copper</span>
             * <span style="color: #FF0000">Alk_Phos</span>
             * <span style="color: #FF0000">SGOT</span>
             * <span style="color: #FF0000">Triglycerides</span>
             * <span style="color: #FF0000">Platelets</span>
             * <span style="color: #FF0000">Prothrombin</span>
        * **7/20 predictor** (object)
             * <span style="color: #FF0000">Drug</span>
             * <span style="color: #FF0000">Sex</span>
             * <span style="color: #FF0000">Ascites</span>
             * <span style="color: #FF0000">Hepatomegaly</span>
             * <span style="color: #FF0000">Spiders</span>
             * <span style="color: #FF0000">Edema</span>
             * <span style="color: #FF0000">Stage</span>


```python
##################################
# Loading Python Libraries
##################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
%matplotlib inline

from operator import add,mul,truediv
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance

from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency

from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.tree import SurvivalTree
from sksurv.metrics import concordance_index_censored
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

import warnings
warnings.filterwarnings('ignore')
```


```python
##################################
# Loading the dataset
##################################
cirrhosis_survival = pd.read_csv('Cirrhosis_Survival.csv')
```


```python
##################################
# Performing a general exploration of the dataset
##################################
print('Dataset Dimensions: ')
display(cirrhosis_survival.shape)
```

    Dataset Dimensions: 
    


    (418, 20)



```python
##################################
# Listing the column names and data types
##################################
print('Column Names and Data Types:')
display(cirrhosis_survival.dtypes)
```

    Column Names and Data Types:
    


    ID                 int64
    N_Days             int64
    Status            object
    Drug              object
    Age                int64
    Sex               object
    Ascites           object
    Hepatomegaly      object
    Spiders           object
    Edema             object
    Bilirubin        float64
    Cholesterol      float64
    Albumin          float64
    Copper           float64
    Alk_Phos         float64
    SGOT             float64
    Tryglicerides    float64
    Platelets        float64
    Prothrombin      float64
    Stage            float64
    dtype: object



```python
##################################
# Taking a snapshot of the dataset
##################################
cirrhosis_survival.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>N_Days</th>
      <th>Status</th>
      <th>Drug</th>
      <th>Age</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
      <th>Stage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>400</td>
      <td>D</td>
      <td>D-penicillamine</td>
      <td>21464</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>14.5</td>
      <td>261.0</td>
      <td>2.60</td>
      <td>156.0</td>
      <td>1718.0</td>
      <td>137.95</td>
      <td>172.0</td>
      <td>190.0</td>
      <td>12.2</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4500</td>
      <td>C</td>
      <td>D-penicillamine</td>
      <td>20617</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>Y</td>
      <td>N</td>
      <td>1.1</td>
      <td>302.0</td>
      <td>4.14</td>
      <td>54.0</td>
      <td>7394.8</td>
      <td>113.52</td>
      <td>88.0</td>
      <td>221.0</td>
      <td>10.6</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1012</td>
      <td>D</td>
      <td>D-penicillamine</td>
      <td>25594</td>
      <td>M</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>S</td>
      <td>1.4</td>
      <td>176.0</td>
      <td>3.48</td>
      <td>210.0</td>
      <td>516.0</td>
      <td>96.10</td>
      <td>55.0</td>
      <td>151.0</td>
      <td>12.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1925</td>
      <td>D</td>
      <td>D-penicillamine</td>
      <td>19994</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>Y</td>
      <td>S</td>
      <td>1.8</td>
      <td>244.0</td>
      <td>2.54</td>
      <td>64.0</td>
      <td>6121.8</td>
      <td>60.63</td>
      <td>92.0</td>
      <td>183.0</td>
      <td>10.3</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1504</td>
      <td>CL</td>
      <td>Placebo</td>
      <td>13918</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>Y</td>
      <td>N</td>
      <td>3.4</td>
      <td>279.0</td>
      <td>3.53</td>
      <td>143.0</td>
      <td>671.0</td>
      <td>113.15</td>
      <td>72.0</td>
      <td>136.0</td>
      <td>10.9</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Taking the ID column as the index
##################################
cirrhosis_survival.set_index(['ID'], inplace=True)
```


```python
##################################
# Changing the data type for Stage
##################################
cirrhosis_survival['Stage'] = cirrhosis_survival['Stage'].astype('object')
```


```python
##################################
# Changing the data type for Status
##################################
cirrhosis_survival['Status'] = cirrhosis_survival['Status'].replace({'C':False, 'CL':False, 'D':True}) 
```


```python
##################################
# Performing a general exploration of the numeric variables
##################################
print('Numeric Variable Summary:')
display(cirrhosis_survival.describe(include='number').transpose())
```

    Numeric Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>N_Days</th>
      <td>418.0</td>
      <td>1917.782297</td>
      <td>1104.672992</td>
      <td>41.00</td>
      <td>1092.7500</td>
      <td>1730.00</td>
      <td>2613.50</td>
      <td>4795.00</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>418.0</td>
      <td>18533.351675</td>
      <td>3815.845055</td>
      <td>9598.00</td>
      <td>15644.5000</td>
      <td>18628.00</td>
      <td>21272.50</td>
      <td>28650.00</td>
    </tr>
    <tr>
      <th>Bilirubin</th>
      <td>418.0</td>
      <td>3.220813</td>
      <td>4.407506</td>
      <td>0.30</td>
      <td>0.8000</td>
      <td>1.40</td>
      <td>3.40</td>
      <td>28.00</td>
    </tr>
    <tr>
      <th>Cholesterol</th>
      <td>284.0</td>
      <td>369.510563</td>
      <td>231.944545</td>
      <td>120.00</td>
      <td>249.5000</td>
      <td>309.50</td>
      <td>400.00</td>
      <td>1775.00</td>
    </tr>
    <tr>
      <th>Albumin</th>
      <td>418.0</td>
      <td>3.497440</td>
      <td>0.424972</td>
      <td>1.96</td>
      <td>3.2425</td>
      <td>3.53</td>
      <td>3.77</td>
      <td>4.64</td>
    </tr>
    <tr>
      <th>Copper</th>
      <td>310.0</td>
      <td>97.648387</td>
      <td>85.613920</td>
      <td>4.00</td>
      <td>41.2500</td>
      <td>73.00</td>
      <td>123.00</td>
      <td>588.00</td>
    </tr>
    <tr>
      <th>Alk_Phos</th>
      <td>312.0</td>
      <td>1982.655769</td>
      <td>2140.388824</td>
      <td>289.00</td>
      <td>871.5000</td>
      <td>1259.00</td>
      <td>1980.00</td>
      <td>13862.40</td>
    </tr>
    <tr>
      <th>SGOT</th>
      <td>312.0</td>
      <td>122.556346</td>
      <td>56.699525</td>
      <td>26.35</td>
      <td>80.6000</td>
      <td>114.70</td>
      <td>151.90</td>
      <td>457.25</td>
    </tr>
    <tr>
      <th>Tryglicerides</th>
      <td>282.0</td>
      <td>124.702128</td>
      <td>65.148639</td>
      <td>33.00</td>
      <td>84.2500</td>
      <td>108.00</td>
      <td>151.00</td>
      <td>598.00</td>
    </tr>
    <tr>
      <th>Platelets</th>
      <td>407.0</td>
      <td>257.024570</td>
      <td>98.325585</td>
      <td>62.00</td>
      <td>188.5000</td>
      <td>251.00</td>
      <td>318.00</td>
      <td>721.00</td>
    </tr>
    <tr>
      <th>Prothrombin</th>
      <td>416.0</td>
      <td>10.731731</td>
      <td>1.022000</td>
      <td>9.00</td>
      <td>10.0000</td>
      <td>10.60</td>
      <td>11.10</td>
      <td>18.00</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration of the object variables
##################################
print('object Variable Summary:')
display(cirrhosis_survival.describe(include='object').transpose())
```

    object Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Drug</th>
      <td>312</td>
      <td>2</td>
      <td>D-penicillamine</td>
      <td>158</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>418</td>
      <td>2</td>
      <td>F</td>
      <td>374</td>
    </tr>
    <tr>
      <th>Ascites</th>
      <td>312</td>
      <td>2</td>
      <td>N</td>
      <td>288</td>
    </tr>
    <tr>
      <th>Hepatomegaly</th>
      <td>312</td>
      <td>2</td>
      <td>Y</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Spiders</th>
      <td>312</td>
      <td>2</td>
      <td>N</td>
      <td>222</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>418</td>
      <td>3</td>
      <td>N</td>
      <td>354</td>
    </tr>
    <tr>
      <th>Stage</th>
      <td>412.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>155.0</td>
    </tr>
  </tbody>
</table>
</div>


## 1.3. Data Quality Assessment <a class="anchor" id="1.3"></a>

Data quality findings based on assessment are as follows:
1. No duplicated rows observed.
2. Missing data noted for 12 variables with Null.Count>0 and Fill.Rate<1.0.
    * <span style="color: #FF0000">Tryglicerides</span>: Null.Count = 136, Fill.Rate = 0.675
    * <span style="color: #FF0000">Cholesterol</span>: Null.Count = 134, Fill.Rate = 0.679
    * <span style="color: #FF0000">Copper</span>: Null.Count = 108, Fill.Rate = 0.741
    * <span style="color: #FF0000">Drug</span>: Null.Count = 106, Fill.Rate = 0.746
    * <span style="color: #FF0000">Ascites</span>: Null.Count = 106, Fill.Rate = 0.746
    * <span style="color: #FF0000">Hepatomegaly</span>: Null.Count = 106, Fill.Rate = 0.746
    * <span style="color: #FF0000">Spiders</span>: Null.Count = 106, Fill.Rate = 0.746
    * <span style="color: #FF0000">Alk_Phos</span>: Null.Count = 106, Fill.Rate = 0.746
    * <span style="color: #FF0000">SGOT</span>: Null.Count = 106, Fill.Rate = 0.746
    * <span style="color: #FF0000">Platelets</span>: Null.Count = 11, Fill.Rate = 0.974
    * <span style="color: #FF0000">Stage</span>: Null.Count = 6, Fill.Rate = 0.986
    * <span style="color: #FF0000">Prothrombin</span>: Null.Count = 2, Fill.Rate = 0.995
3. 142 observations noted with at least 1 missing data. From this number, 106 observations reported high Missing.Rate>0.4.
    * 91 Observations: Missing.Rate = 0.450 (9 columns)
    * 15 Observations: Missing.Rate = 0.500 (10 columns)
    * 28 Observations: Missing.Rate = 0.100 (2 columns)
    * 8 Observations: Missing.Rate = 0.050 (1 column)
4. Low variance observed for 3 variables with First.Second.Mode.Ratio>5.
    * <span style="color: #FF0000">Ascites</span>: First.Second.Mode.Ratio = 12.000
    * <span style="color: #FF0000">Sex</span>: First.Second.Mode.Ratio = 8.500
    * <span style="color: #FF0000">Edema</span>: First.Second.Mode.Ratio = 8.045
5. No low variance observed for any variable with Unique.Count.Ratio>10.
6. High and marginally high skewness observed for 2 variables with Skewness>3 or Skewness<(-3).
    * <span style="color: #FF0000">Cholesterol</span>: Skewness = +3.409
    * <span style="color: #FF0000">Alk_Phos</span>: Skewness = +2.993


```python
##################################
# Counting the number of duplicated rows
##################################
cirrhosis_survival.duplicated().sum()
```




    0




```python
##################################
# Gathering the data types for each column
##################################
data_type_list = list(cirrhosis_survival.dtypes)
```


```python
##################################
# Gathering the variable names for each column
##################################
variable_name_list = list(cirrhosis_survival.columns)
```


```python
##################################
# Gathering the number of observations for each column
##################################
row_count_list = list([len(cirrhosis_survival)] * len(cirrhosis_survival.columns))
```


```python
##################################
# Gathering the number of missing data for each column
##################################
null_count_list = list(cirrhosis_survival.isna().sum(axis=0))
```


```python
##################################
# Gathering the number of non-missing data for each column
##################################
non_null_count_list = list(cirrhosis_survival.count())
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
fill_rate_list = map(truediv, non_null_count_list, row_count_list)
```


```python
##################################
# Formulating the summary
# for all columns
##################################
all_column_quality_summary = pd.DataFrame(zip(variable_name_list,
                                              data_type_list,
                                              row_count_list,
                                              non_null_count_list,
                                              null_count_list,
                                              fill_rate_list), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count',                                                 
                                                 'Fill.Rate'])
display(all_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>N_Days</td>
      <td>int64</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Status</td>
      <td>bool</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Drug</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Age</td>
      <td>int64</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sex</td>
      <td>object</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ascites</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hepatomegaly</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Spiders</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Edema</td>
      <td>object</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bilirubin</td>
      <td>float64</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cholesterol</td>
      <td>float64</td>
      <td>418</td>
      <td>284</td>
      <td>134</td>
      <td>0.679426</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Albumin</td>
      <td>float64</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Copper</td>
      <td>float64</td>
      <td>418</td>
      <td>310</td>
      <td>108</td>
      <td>0.741627</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Alk_Phos</td>
      <td>float64</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SGOT</td>
      <td>float64</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Tryglicerides</td>
      <td>float64</td>
      <td>418</td>
      <td>282</td>
      <td>136</td>
      <td>0.674641</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Platelets</td>
      <td>float64</td>
      <td>418</td>
      <td>407</td>
      <td>11</td>
      <td>0.973684</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Prothrombin</td>
      <td>float64</td>
      <td>418</td>
      <td>416</td>
      <td>2</td>
      <td>0.995215</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Stage</td>
      <td>object</td>
      <td>418</td>
      <td>412</td>
      <td>6</td>
      <td>0.985646</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of columns
# with Fill.Rate < 1.00
##################################
print('Number of Columns with Missing Data:', str(len(all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1)])))
```

    Number of Columns with Missing Data: 12
    


```python
##################################
# Identifying the columns
# with Fill.Rate < 1.00
##################################
print('Columns with Missing Data:')
display(all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1)].sort_values(by=['Fill.Rate'], ascending=True))
```

    Columns with Missing Data:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>Tryglicerides</td>
      <td>float64</td>
      <td>418</td>
      <td>282</td>
      <td>136</td>
      <td>0.674641</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cholesterol</td>
      <td>float64</td>
      <td>418</td>
      <td>284</td>
      <td>134</td>
      <td>0.679426</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Copper</td>
      <td>float64</td>
      <td>418</td>
      <td>310</td>
      <td>108</td>
      <td>0.741627</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Drug</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ascites</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hepatomegaly</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Spiders</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Alk_Phos</td>
      <td>float64</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SGOT</td>
      <td>float64</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Platelets</td>
      <td>float64</td>
      <td>418</td>
      <td>407</td>
      <td>11</td>
      <td>0.973684</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Stage</td>
      <td>object</td>
      <td>418</td>
      <td>412</td>
      <td>6</td>
      <td>0.985646</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Prothrombin</td>
      <td>float64</td>
      <td>418</td>
      <td>416</td>
      <td>2</td>
      <td>0.995215</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Identifying the rows
# with Fill.Rate < 1.00
##################################
column_low_fill_rate = all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1.00)]
```


```python
##################################
# Gathering the metadata labels for each observation
##################################
row_metadata_list = cirrhosis_survival.index.values.tolist()
```


```python
##################################
# Gathering the number of columns for each observation
##################################
column_count_list = list([len(cirrhosis_survival.columns)] * len(cirrhosis_survival))
```


```python
##################################
# Gathering the number of missing data for each row
##################################
null_row_list = list(cirrhosis_survival.isna().sum(axis=1))
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
missing_rate_list = map(truediv, null_row_list, column_count_list)
```


```python
##################################
# Exploring the rows
# for missing data
##################################
all_row_quality_summary = pd.DataFrame(zip(row_metadata_list,
                                           column_count_list,
                                           null_row_list,
                                           missing_rate_list), 
                                        columns=['Row.Name',
                                                 'Column.Count',
                                                 'Null.Count',                                                 
                                                 'Missing.Rate'])
display(all_row_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Row.Name</th>
      <th>Column.Count</th>
      <th>Null.Count</th>
      <th>Missing.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>414</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>414</th>
      <td>415</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>415</th>
      <td>416</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>416</th>
      <td>417</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>417</th>
      <td>418</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
  </tbody>
</table>
<p>418 rows × 4 columns</p>
</div>



```python
##################################
# Counting the number of rows
# with Fill.Rate < 1.00
##################################
print('Number of Rows with Missing Data:',str(len(all_row_quality_summary[all_row_quality_summary['Missing.Rate']>0])))
```

    Number of Rows with Missing Data: 142
    


```python
##################################
# Identifying the rows
# with Fill.Rate < 1.00
##################################
print('Rows with Missing Data:')
display(all_row_quality_summary[all_row_quality_summary['Missing.Rate']>0])
```

    Rows with Missing Data:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Row.Name</th>
      <th>Column.Count</th>
      <th>Null.Count</th>
      <th>Missing.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>19</td>
      <td>1</td>
      <td>0.052632</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>19</td>
      <td>2</td>
      <td>0.105263</td>
    </tr>
    <tr>
      <th>39</th>
      <td>40</td>
      <td>19</td>
      <td>2</td>
      <td>0.105263</td>
    </tr>
    <tr>
      <th>40</th>
      <td>41</td>
      <td>19</td>
      <td>2</td>
      <td>0.105263</td>
    </tr>
    <tr>
      <th>41</th>
      <td>42</td>
      <td>19</td>
      <td>2</td>
      <td>0.105263</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>414</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>414</th>
      <td>415</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>415</th>
      <td>416</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>416</th>
      <td>417</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>417</th>
      <td>418</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
  </tbody>
</table>
<p>142 rows × 4 columns</p>
</div>



```python
##################################
# Counting the number of rows
# based on different Fill.Rate categories
##################################
missing_rate_categories = all_row_quality_summary['Missing.Rate'].value_counts().reset_index()
missing_rate_categories.columns = ['Missing.Rate.Category','Missing.Rate.Count']
display(missing_rate_categories.sort_values(['Missing.Rate.Category'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing.Rate.Category</th>
      <th>Missing.Rate.Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>0.526316</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.473684</td>
      <td>91</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.105263</td>
      <td>28</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.052632</td>
      <td>8</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>276</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Identifying the rows
# with Missing.Rate > 0.40
##################################
row_high_missing_rate = all_row_quality_summary[(all_row_quality_summary['Missing.Rate']>0.40)]
```


```python
##################################
# Formulating the dataset
# with numeric columns only
##################################
cirrhosis_survival_numeric = cirrhosis_survival.select_dtypes(include='number')
```


```python
##################################
# Gathering the variable names for each numeric column
##################################
numeric_variable_name_list = cirrhosis_survival_numeric.columns
```


```python
##################################
# Gathering the minimum value for each numeric column
##################################
numeric_minimum_list = cirrhosis_survival_numeric.min()
```


```python
##################################
# Gathering the mean value for each numeric column
##################################
numeric_mean_list = cirrhosis_survival_numeric.mean()
```


```python
##################################
# Gathering the median value for each numeric column
##################################
numeric_median_list = cirrhosis_survival_numeric.median()
```


```python
##################################
# Gathering the maximum value for each numeric column
##################################
numeric_maximum_list = cirrhosis_survival_numeric.max()
```


```python
##################################
# Gathering the first mode values for each numeric column
##################################
numeric_first_mode_list = [cirrhosis_survival[x].value_counts(dropna=True).index.tolist()[0] for x in cirrhosis_survival_numeric]
```


```python
##################################
# Gathering the second mode values for each numeric column
##################################
numeric_second_mode_list = [cirrhosis_survival[x].value_counts(dropna=True).index.tolist()[1] for x in cirrhosis_survival_numeric]
```


```python
##################################
# Gathering the count of first mode values for each numeric column
##################################
numeric_first_mode_count_list = [cirrhosis_survival_numeric[x].isin([cirrhosis_survival[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in cirrhosis_survival_numeric]
```


```python
##################################
# Gathering the count of second mode values for each numeric column
##################################
numeric_second_mode_count_list = [cirrhosis_survival_numeric[x].isin([cirrhosis_survival[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in cirrhosis_survival_numeric]
```


```python
##################################
# Gathering the first mode to second mode ratio for each numeric column
##################################
numeric_first_second_mode_ratio_list = map(truediv, numeric_first_mode_count_list, numeric_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each numeric column
##################################
numeric_unique_count_list = cirrhosis_survival_numeric.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each numeric column
##################################
numeric_row_count_list = list([len(cirrhosis_survival_numeric)] * len(cirrhosis_survival_numeric.columns))
```


```python
##################################
# Gathering the unique to count ratio for each numeric column
##################################
numeric_unique_count_ratio_list = map(truediv, numeric_unique_count_list, numeric_row_count_list)
```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
numeric_skewness_list = cirrhosis_survival_numeric.skew()
```


```python
##################################
# Gathering the kurtosis value for each numeric column
##################################
numeric_kurtosis_list = cirrhosis_survival_numeric.kurtosis()
```


```python
numeric_column_quality_summary = pd.DataFrame(zip(numeric_variable_name_list,
                                                numeric_minimum_list,
                                                numeric_mean_list,
                                                numeric_median_list,
                                                numeric_maximum_list,
                                                numeric_first_mode_list,
                                                numeric_second_mode_list,
                                                numeric_first_mode_count_list,
                                                numeric_second_mode_count_list,
                                                numeric_first_second_mode_ratio_list,
                                                numeric_unique_count_list,
                                                numeric_row_count_list,
                                                numeric_unique_count_ratio_list,
                                                numeric_skewness_list,
                                                numeric_kurtosis_list), 
                                        columns=['Numeric.Column.Name',
                                                 'Minimum',
                                                 'Mean',
                                                 'Median',
                                                 'Maximum',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio',
                                                 'Skewness',
                                                 'Kurtosis'])
display(numeric_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Minimum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Maximum</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>N_Days</td>
      <td>41.00</td>
      <td>1917.782297</td>
      <td>1730.00</td>
      <td>4795.00</td>
      <td>1434.00</td>
      <td>3445.00</td>
      <td>2</td>
      <td>2</td>
      <td>1.000000</td>
      <td>399</td>
      <td>418</td>
      <td>0.954545</td>
      <td>0.472602</td>
      <td>-0.482139</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Age</td>
      <td>9598.00</td>
      <td>18533.351675</td>
      <td>18628.00</td>
      <td>28650.00</td>
      <td>19724.00</td>
      <td>18993.00</td>
      <td>7</td>
      <td>6</td>
      <td>1.166667</td>
      <td>344</td>
      <td>418</td>
      <td>0.822967</td>
      <td>0.086850</td>
      <td>-0.616730</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bilirubin</td>
      <td>0.30</td>
      <td>3.220813</td>
      <td>1.40</td>
      <td>28.00</td>
      <td>0.70</td>
      <td>0.60</td>
      <td>33</td>
      <td>31</td>
      <td>1.064516</td>
      <td>98</td>
      <td>418</td>
      <td>0.234450</td>
      <td>2.717611</td>
      <td>8.065336</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cholesterol</td>
      <td>120.00</td>
      <td>369.510563</td>
      <td>309.50</td>
      <td>1775.00</td>
      <td>260.00</td>
      <td>316.00</td>
      <td>4</td>
      <td>4</td>
      <td>1.000000</td>
      <td>201</td>
      <td>418</td>
      <td>0.480861</td>
      <td>3.408526</td>
      <td>14.337870</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Albumin</td>
      <td>1.96</td>
      <td>3.497440</td>
      <td>3.53</td>
      <td>4.64</td>
      <td>3.35</td>
      <td>3.50</td>
      <td>11</td>
      <td>8</td>
      <td>1.375000</td>
      <td>154</td>
      <td>418</td>
      <td>0.368421</td>
      <td>-0.467527</td>
      <td>0.566745</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Copper</td>
      <td>4.00</td>
      <td>97.648387</td>
      <td>73.00</td>
      <td>588.00</td>
      <td>52.00</td>
      <td>67.00</td>
      <td>8</td>
      <td>7</td>
      <td>1.142857</td>
      <td>158</td>
      <td>418</td>
      <td>0.377990</td>
      <td>2.303640</td>
      <td>7.624023</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Alk_Phos</td>
      <td>289.00</td>
      <td>1982.655769</td>
      <td>1259.00</td>
      <td>13862.40</td>
      <td>601.00</td>
      <td>794.00</td>
      <td>2</td>
      <td>2</td>
      <td>1.000000</td>
      <td>295</td>
      <td>418</td>
      <td>0.705742</td>
      <td>2.992834</td>
      <td>9.662553</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SGOT</td>
      <td>26.35</td>
      <td>122.556346</td>
      <td>114.70</td>
      <td>457.25</td>
      <td>71.30</td>
      <td>137.95</td>
      <td>6</td>
      <td>5</td>
      <td>1.200000</td>
      <td>179</td>
      <td>418</td>
      <td>0.428230</td>
      <td>1.449197</td>
      <td>4.311976</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Tryglicerides</td>
      <td>33.00</td>
      <td>124.702128</td>
      <td>108.00</td>
      <td>598.00</td>
      <td>118.00</td>
      <td>90.00</td>
      <td>7</td>
      <td>6</td>
      <td>1.166667</td>
      <td>146</td>
      <td>418</td>
      <td>0.349282</td>
      <td>2.523902</td>
      <td>11.802753</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Platelets</td>
      <td>62.00</td>
      <td>257.024570</td>
      <td>251.00</td>
      <td>721.00</td>
      <td>344.00</td>
      <td>269.00</td>
      <td>6</td>
      <td>5</td>
      <td>1.200000</td>
      <td>243</td>
      <td>418</td>
      <td>0.581340</td>
      <td>0.627098</td>
      <td>0.863045</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Prothrombin</td>
      <td>9.00</td>
      <td>10.731731</td>
      <td>10.60</td>
      <td>18.00</td>
      <td>10.60</td>
      <td>11.00</td>
      <td>39</td>
      <td>32</td>
      <td>1.218750</td>
      <td>48</td>
      <td>418</td>
      <td>0.114833</td>
      <td>2.223276</td>
      <td>10.040773</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the dataset
# with object column only
##################################
cirrhosis_survival_object = cirrhosis_survival.select_dtypes(include='object')
```


```python
##################################
# Gathering the variable names for the object column
##################################
object_variable_name_list = cirrhosis_survival_object.columns
```


```python
##################################
# Gathering the first mode values for the object column
##################################
object_first_mode_list = [cirrhosis_survival[x].value_counts().index.tolist()[0] for x in cirrhosis_survival_object]
```


```python
##################################
# Gathering the second mode values for each object column
##################################
object_second_mode_list = [cirrhosis_survival[x].value_counts().index.tolist()[1] for x in cirrhosis_survival_object]
```


```python
##################################
# Gathering the count of first mode values for each object column
##################################
object_first_mode_count_list = [cirrhosis_survival_object[x].isin([cirrhosis_survival[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in cirrhosis_survival_object]
```


```python
##################################
# Gathering the count of second mode values for each object column
##################################
object_second_mode_count_list = [cirrhosis_survival_object[x].isin([cirrhosis_survival[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in cirrhosis_survival_object]
```


```python
##################################
# Gathering the first mode to second mode ratio for each object column
##################################
object_first_second_mode_ratio_list = map(truediv, object_first_mode_count_list, object_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each object column
##################################
object_unique_count_list = cirrhosis_survival_object.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each object column
##################################
object_row_count_list = list([len(cirrhosis_survival_object)] * len(cirrhosis_survival_object.columns))
```


```python
##################################
# Gathering the unique to count ratio for each object column
##################################
object_unique_count_ratio_list = map(truediv, object_unique_count_list, object_row_count_list)
```


```python
object_column_quality_summary = pd.DataFrame(zip(object_variable_name_list,
                                                 object_first_mode_list,
                                                 object_second_mode_list,
                                                 object_first_mode_count_list,
                                                 object_second_mode_count_list,
                                                 object_first_second_mode_ratio_list,
                                                 object_unique_count_list,
                                                 object_row_count_list,
                                                 object_unique_count_ratio_list), 
                                        columns=['Object.Column.Name',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio'])
display(object_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Object.Column.Name</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Drug</td>
      <td>D-penicillamine</td>
      <td>Placebo</td>
      <td>158</td>
      <td>154</td>
      <td>1.025974</td>
      <td>2</td>
      <td>418</td>
      <td>0.004785</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sex</td>
      <td>F</td>
      <td>M</td>
      <td>374</td>
      <td>44</td>
      <td>8.500000</td>
      <td>2</td>
      <td>418</td>
      <td>0.004785</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ascites</td>
      <td>N</td>
      <td>Y</td>
      <td>288</td>
      <td>24</td>
      <td>12.000000</td>
      <td>2</td>
      <td>418</td>
      <td>0.004785</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hepatomegaly</td>
      <td>Y</td>
      <td>N</td>
      <td>160</td>
      <td>152</td>
      <td>1.052632</td>
      <td>2</td>
      <td>418</td>
      <td>0.004785</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Spiders</td>
      <td>N</td>
      <td>Y</td>
      <td>222</td>
      <td>90</td>
      <td>2.466667</td>
      <td>2</td>
      <td>418</td>
      <td>0.004785</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Edema</td>
      <td>N</td>
      <td>S</td>
      <td>354</td>
      <td>44</td>
      <td>8.045455</td>
      <td>3</td>
      <td>418</td>
      <td>0.007177</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Stage</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>155</td>
      <td>144</td>
      <td>1.076389</td>
      <td>4</td>
      <td>418</td>
      <td>0.009569</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of object columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(object_column_quality_summary[(object_column_quality_summary['First.Second.Mode.Ratio']>5)])
```




    3




```python
##################################
# Counting the number of object columns
# with Unique.Count.Ratio > 10.00
##################################
len(object_column_quality_summary[(object_column_quality_summary['Unique.Count.Ratio']>10)])
```




    0



## 1.4. Data Preprocessing <a class="anchor" id="1.4"></a>

### 1.4.1 Data Cleaning <a class="anchor" id="1.4.1"></a>

1. Subsets of rows with high rates of missing data were removed from the dataset:
    * 106 rows with Missing.Rate>0.4 were exluded for subsequent analysis.
2. No variables were removed due to zero or near-zero variance.


```python
##################################
# Performing a general exploration of the original dataset
##################################
print('Dataset Dimensions: ')
display(cirrhosis_survival.shape)
```

    Dataset Dimensions: 
    


    (418, 19)



```python
##################################
# Filtering out the rows with
# with Missing.Rate > 0.40
##################################
cirrhosis_survival_filtered_row = cirrhosis_survival.drop(cirrhosis_survival[cirrhosis_survival.index.isin(row_high_missing_rate['Row.Name'].values.tolist())].index)
```


```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cirrhosis_survival_filtered_row.shape)
```

    Dataset Dimensions: 
    


    (312, 19)



```python
##################################
# Gathering the missing data percentage for each column
# from the filtered data
##################################
data_type_list = list(cirrhosis_survival_filtered_row.dtypes)
variable_name_list = list(cirrhosis_survival_filtered_row.columns)
null_count_list = list(cirrhosis_survival_filtered_row.isna().sum(axis=0))
non_null_count_list = list(cirrhosis_survival_filtered_row.count())
row_count_list = list([len(cirrhosis_survival_filtered_row)] * len(cirrhosis_survival_filtered_row.columns))
fill_rate_list = map(truediv, non_null_count_list, row_count_list)
all_column_quality_summary = pd.DataFrame(zip(variable_name_list,
                                              data_type_list,
                                              row_count_list,
                                              non_null_count_list,
                                              null_count_list,
                                              fill_rate_list), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count',                                                 
                                                 'Fill.Rate'])
display(all_column_quality_summary.sort_values(['Fill.Rate'], ascending=True))

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>Tryglicerides</td>
      <td>float64</td>
      <td>312</td>
      <td>282</td>
      <td>30</td>
      <td>0.903846</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cholesterol</td>
      <td>float64</td>
      <td>312</td>
      <td>284</td>
      <td>28</td>
      <td>0.910256</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Platelets</td>
      <td>float64</td>
      <td>312</td>
      <td>308</td>
      <td>4</td>
      <td>0.987179</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Copper</td>
      <td>float64</td>
      <td>312</td>
      <td>310</td>
      <td>2</td>
      <td>0.993590</td>
    </tr>
    <tr>
      <th>0</th>
      <td>N_Days</td>
      <td>int64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SGOT</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Alk_Phos</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Albumin</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Prothrombin</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bilirubin</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Spiders</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hepatomegaly</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ascites</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sex</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Age</td>
      <td>int64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Drug</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Status</td>
      <td>bool</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Edema</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Stage</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating a new dataset object
# for the cleaned data
##################################
cirrhosis_survival_cleaned = cirrhosis_survival_filtered_row
```

### 1.4.2 Missing Data Imputation <a class="anchor" id="1.4.2"></a>

1. To prevent data leakage, the original dataset was divided into training and testing subsets prior to imputation.
2. Missing data in the training subset for float variables were imputed using the iterative imputer algorithm with a  linear regression estimator.
    * <span style="color: #FF0000">Tryglicerides</span>: Null.Count = 20
    * <span style="color: #FF0000">Cholesterol</span>: Null.Count = 18
    * <span style="color: #FF0000">Platelets</span>: Null.Count = 2
    * <span style="color: #FF0000">Copper</span>: Null.Count = 1
3. Missing data in the testing subset for float variables will be treated with iterative imputing downstream using a pipeline involving the final preprocessing steps.



```python
##################################
# Formulating the summary
# for all cleaned columns
##################################
cleaned_column_quality_summary = pd.DataFrame(zip(list(cirrhosis_survival_cleaned.columns),
                                                  list(cirrhosis_survival_cleaned.dtypes),
                                                  list([len(cirrhosis_survival_cleaned)] * len(cirrhosis_survival_cleaned.columns)),
                                                  list(cirrhosis_survival_cleaned.count()),
                                                  list(cirrhosis_survival_cleaned.isna().sum(axis=0))), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count'])
display(cleaned_column_quality_summary.sort_values(by=['Null.Count'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>Tryglicerides</td>
      <td>float64</td>
      <td>312</td>
      <td>282</td>
      <td>30</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cholesterol</td>
      <td>float64</td>
      <td>312</td>
      <td>284</td>
      <td>28</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Platelets</td>
      <td>float64</td>
      <td>312</td>
      <td>308</td>
      <td>4</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Copper</td>
      <td>float64</td>
      <td>312</td>
      <td>310</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>N_Days</td>
      <td>int64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Prothrombin</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SGOT</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Alk_Phos</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Albumin</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bilirubin</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Status</td>
      <td>bool</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Edema</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Spiders</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hepatomegaly</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ascites</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sex</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Age</td>
      <td>int64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Drug</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Stage</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Creating training and testing data
##################################
cirrhosis_survival_train, cirrhosis_survival_test = train_test_split(cirrhosis_survival_cleaned, 
                                                                     test_size=0.30, 
                                                                     stratify=cirrhosis_survival_cleaned['Status'], 
                                                                     random_state=88888888)
cirrhosis_survival_X_train_cleaned = cirrhosis_survival_train.drop(columns=['Status', 'N_Days'])
cirrhosis_survival_y_train_cleaned = cirrhosis_survival_train[['Status', 'N_Days']]
cirrhosis_survival_X_test_cleaned = cirrhosis_survival_test.drop(columns=['Status', 'N_Days'])
cirrhosis_survival_y_test_cleaned = cirrhosis_survival_test[['Status', 'N_Days']]
```


```python
##################################
# Gathering the training data information
##################################
print(f'Training Dataset Dimensions: Predictors: {cirrhosis_survival_X_train_cleaned.shape}, Event|Duration: {cirrhosis_survival_y_train_cleaned.shape}')
```

    Training Dataset Dimensions: Predictors: (218, 17), Event|Duration: (218, 2)
    


```python
##################################
# Gathering the testing data information
##################################
print(f'Testing Dataset Dimensions: Predictors: {cirrhosis_survival_X_test_cleaned.shape}, Event|Duration: {cirrhosis_survival_y_test_cleaned.shape}')
```

    Testing Dataset Dimensions: Predictors: (94, 17), Event|Duration: (94, 2)
    


```python
##################################
# Formulating the summary
# for all cleaned columns
# from the training data
##################################
X_train_cleaned_column_quality_summary = pd.DataFrame(zip(list(cirrhosis_survival_X_train_cleaned.columns),
                                                  list(cirrhosis_survival_X_train_cleaned.dtypes),
                                                  list([len(cirrhosis_survival_X_train_cleaned)] * len(cirrhosis_survival_X_train_cleaned.columns)),
                                                  list(cirrhosis_survival_X_train_cleaned.count()),
                                                  list(cirrhosis_survival_X_train_cleaned.isna().sum(axis=0))), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count'])
display(X_train_cleaned_column_quality_summary.sort_values(by=['Null.Count'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>Tryglicerides</td>
      <td>float64</td>
      <td>218</td>
      <td>200</td>
      <td>18</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Cholesterol</td>
      <td>float64</td>
      <td>218</td>
      <td>202</td>
      <td>16</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Platelets</td>
      <td>float64</td>
      <td>218</td>
      <td>215</td>
      <td>3</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Copper</td>
      <td>float64</td>
      <td>218</td>
      <td>217</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Albumin</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Prothrombin</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>SGOT</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Alk_Phos</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Drug</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Age</td>
      <td>int64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Bilirubin</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Edema</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Spiders</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hepatomegaly</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ascites</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sex</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Stage</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the summary
# for all cleaned columns
# from the testing data
##################################
X_test_cleaned_column_quality_summary = pd.DataFrame(zip(list(cirrhosis_survival_X_test_cleaned.columns),
                                                  list(cirrhosis_survival_X_test_cleaned.dtypes),
                                                  list([len(cirrhosis_survival_X_test_cleaned)] * len(cirrhosis_survival_X_test_cleaned.columns)),
                                                  list(cirrhosis_survival_X_test_cleaned.count()),
                                                  list(cirrhosis_survival_X_test_cleaned.isna().sum(axis=0))), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count'])
display(X_test_cleaned_column_quality_summary.sort_values(by=['Null.Count'], ascending=False))

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>Cholesterol</td>
      <td>float64</td>
      <td>94</td>
      <td>82</td>
      <td>12</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Tryglicerides</td>
      <td>float64</td>
      <td>94</td>
      <td>82</td>
      <td>12</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Platelets</td>
      <td>float64</td>
      <td>94</td>
      <td>93</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Copper</td>
      <td>float64</td>
      <td>94</td>
      <td>93</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Albumin</td>
      <td>float64</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Prothrombin</td>
      <td>float64</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>SGOT</td>
      <td>float64</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Alk_Phos</td>
      <td>float64</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Drug</td>
      <td>object</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Age</td>
      <td>int64</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Bilirubin</td>
      <td>float64</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Edema</td>
      <td>object</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Spiders</td>
      <td>object</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hepatomegaly</td>
      <td>object</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ascites</td>
      <td>object</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sex</td>
      <td>object</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Stage</td>
      <td>object</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the cleaned training dataset
# with object columns only
##################################
cirrhosis_survival_X_train_cleaned_object = cirrhosis_survival_X_train_cleaned.select_dtypes(include='object')
cirrhosis_survival_X_train_cleaned_object.reset_index(drop=True, inplace=True)
cirrhosis_survival_X_train_cleaned_object.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>D-penicillamine</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>D-penicillamine</td>
      <td>M</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>D-penicillamine</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Placebo</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Placebo</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>N</td>
      <td>N</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Formulating the cleaned training dataset
# with integer columns only
##################################
cirrhosis_survival_X_train_cleaned_int = cirrhosis_survival_X_train_cleaned.select_dtypes(include='int')
cirrhosis_survival_X_train_cleaned_int.reset_index(drop=True, inplace=True)
cirrhosis_survival_X_train_cleaned_int.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13329</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12912</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17180</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17884</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15177</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Formulating the cleaned training dataset
# with float columns only
##################################
cirrhosis_survival_X_train_cleaned_float = cirrhosis_survival_X_train_cleaned.select_dtypes(include='float')
cirrhosis_survival_X_train_cleaned_float.reset_index(drop=True, inplace=True)
cirrhosis_survival_X_train_cleaned_float.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.4</td>
      <td>450.0</td>
      <td>3.37</td>
      <td>32.0</td>
      <td>1408.0</td>
      <td>116.25</td>
      <td>118.0</td>
      <td>313.0</td>
      <td>11.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.4</td>
      <td>646.0</td>
      <td>3.83</td>
      <td>102.0</td>
      <td>855.0</td>
      <td>127.00</td>
      <td>194.0</td>
      <td>306.0</td>
      <td>10.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.9</td>
      <td>346.0</td>
      <td>3.77</td>
      <td>59.0</td>
      <td>794.0</td>
      <td>125.55</td>
      <td>56.0</td>
      <td>336.0</td>
      <td>10.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.5</td>
      <td>188.0</td>
      <td>3.67</td>
      <td>57.0</td>
      <td>1273.0</td>
      <td>119.35</td>
      <td>102.0</td>
      <td>110.0</td>
      <td>11.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.7</td>
      <td>296.0</td>
      <td>3.44</td>
      <td>114.0</td>
      <td>9933.2</td>
      <td>206.40</td>
      <td>101.0</td>
      <td>195.0</td>
      <td>10.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Defining the estimator to be used
# at each step of the round-robin imputation
##################################
lr = LinearRegression()
```


```python
##################################
# Defining the parameter of the
# iterative imputer which will estimate 
# the columns with missing values
# as a function of the other columns
# in a round-robin fashion
##################################
iterative_imputer = IterativeImputer(
    estimator = lr,
    max_iter = 10,
    tol = 1e-10,
    imputation_order = 'ascending',
    random_state=88888888
)
```


```python
##################################
# Implementing the iterative imputer 
##################################
cirrhosis_survival_X_train_imputed_float_array = iterative_imputer.fit_transform(cirrhosis_survival_X_train_cleaned_float)
```


```python
##################################
# Transforming the imputed training data
# from an array to a dataframe
##################################
cirrhosis_survival_X_train_imputed_float = pd.DataFrame(cirrhosis_survival_X_train_imputed_float_array, 
                                                        columns = cirrhosis_survival_X_train_cleaned_float.columns)
cirrhosis_survival_X_train_imputed_float.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.4</td>
      <td>450.0</td>
      <td>3.37</td>
      <td>32.0</td>
      <td>1408.0</td>
      <td>116.25</td>
      <td>118.0</td>
      <td>313.0</td>
      <td>11.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.4</td>
      <td>646.0</td>
      <td>3.83</td>
      <td>102.0</td>
      <td>855.0</td>
      <td>127.00</td>
      <td>194.0</td>
      <td>306.0</td>
      <td>10.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.9</td>
      <td>346.0</td>
      <td>3.77</td>
      <td>59.0</td>
      <td>794.0</td>
      <td>125.55</td>
      <td>56.0</td>
      <td>336.0</td>
      <td>10.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.5</td>
      <td>188.0</td>
      <td>3.67</td>
      <td>57.0</td>
      <td>1273.0</td>
      <td>119.35</td>
      <td>102.0</td>
      <td>110.0</td>
      <td>11.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.7</td>
      <td>296.0</td>
      <td>3.44</td>
      <td>114.0</td>
      <td>9933.2</td>
      <td>206.40</td>
      <td>101.0</td>
      <td>195.0</td>
      <td>10.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Formulating the imputed training dataset
##################################
cirrhosis_survival_X_train_imputed = pd.concat([cirrhosis_survival_X_train_cleaned_int,
                                                cirrhosis_survival_X_train_cleaned_object,
                                                cirrhosis_survival_X_train_imputed_float], 
                                               axis=1, 
                                               join='inner')  
```


```python
##################################
# Formulating the summary
# for all imputed columns
##################################
X_train_imputed_column_quality_summary = pd.DataFrame(zip(list(cirrhosis_survival_X_train_imputed.columns),
                                                         list(cirrhosis_survival_X_train_imputed.dtypes),
                                                         list([len(cirrhosis_survival_X_train_imputed)] * len(cirrhosis_survival_X_train_imputed.columns)),
                                                         list(cirrhosis_survival_X_train_imputed.count()),
                                                         list(cirrhosis_survival_X_train_imputed.isna().sum(axis=0))), 
                                                     columns=['Column.Name',
                                                              'Column.Type',
                                                              'Row.Count',
                                                              'Non.Null.Count',
                                                              'Null.Count'])
display(X_train_imputed_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>int64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Drug</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sex</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ascites</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hepatomegaly</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Spiders</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Edema</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Stage</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Bilirubin</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Cholesterol</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Albumin</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Copper</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Alk_Phos</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>SGOT</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Tryglicerides</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Platelets</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Prothrombin</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


### 1.4.3 Outlier Detection <a class="anchor" id="1.4.3"></a>

1. High number of outliers observed in the training subset for 4 numeric variables with Outlier.Ratio>0.05 and marginal to high Skewness.
    * <span style="color: #FF0000">Alk_Phos</span>: Outlier.Count = 25, Outlier.Ratio = 0.114, Skewness=+3.035
    * <span style="color: #FF0000">Bilirubin</span>: Outlier.Count = 18, Outlier.Ratio = 0.083, Skewness=+3.121
    * <span style="color: #FF0000">Cholesterol</span>: Outlier.Count = 17, Outlier.Ratio = 0.078, Skewness=+3.761
    * <span style="color: #FF0000">Prothrombin</span>: Outlier.Count = 12, Outlier.Ratio = 0.055, Skewness=+1.009
2. Minimal number of outliers observed in the training subset for 5 numeric variables with Outlier.Ratio>0.00 but <0.05 and normal to marginal Skewness.
    * <span style="color: #FF0000">Copper</span>: Outlier.Count = 8, Outlier.Ratio = 0.037, Skewness=+1.485
    * <span style="color: #FF0000">Albumin</span>: Outlier.Count = 6, Outlier.Ratio = 0.027, Skewness=-0.589
    * <span style="color: #FF0000">SGOT</span>: Outlier.Count = 4, Outlier.Ratio = 0.018, Skewness=+0.934
    * <span style="color: #FF0000">Tryglicerides</span>: Outlier.Count = 4, Outlier.Ratio = 0.018, Skewness=+2.817
    * <span style="color: #FF0000">Platelets</span>: Outlier.Count = 4, Outlier.Ratio = 0.018, Skewness=+0.374
    * <span style="color: #FF0000">Age</span>: Outlier.Count = 1, Outlier.Ratio = 0.004, Skewness=+0.223


```python
##################################
# Formulating the imputed dataset
# with numeric columns only
##################################
cirrhosis_survival_X_train_imputed_numeric = cirrhosis_survival_X_train_imputed.select_dtypes(include='number')
```


```python
##################################
# Gathering the variable names for each numeric column
##################################
X_train_numeric_variable_name_list = list(cirrhosis_survival_X_train_imputed_numeric.columns)
```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
X_train_numeric_skewness_list = cirrhosis_survival_X_train_imputed_numeric.skew()
```


```python
##################################
# Computing the interquartile range
# for all columns
##################################
cirrhosis_survival_X_train_imputed_numeric_q1 = cirrhosis_survival_X_train_imputed_numeric.quantile(0.25)
cirrhosis_survival_X_train_imputed_numeric_q3 = cirrhosis_survival_X_train_imputed_numeric.quantile(0.75)
cirrhosis_survival_X_train_imputed_numeric_iqr = cirrhosis_survival_X_train_imputed_numeric_q3 - cirrhosis_survival_X_train_imputed_numeric_q1
```


```python
##################################
# Gathering the outlier count for each numeric column
# based on the interquartile range criterion
##################################
X_train_numeric_outlier_count_list = ((cirrhosis_survival_X_train_imputed_numeric < (cirrhosis_survival_X_train_imputed_numeric_q1 - 1.5 * cirrhosis_survival_X_train_imputed_numeric_iqr)) | (cirrhosis_survival_X_train_imputed_numeric > (cirrhosis_survival_X_train_imputed_numeric_q3 + 1.5 * cirrhosis_survival_X_train_imputed_numeric_iqr))).sum()
```


```python
##################################
# Gathering the number of observations for each column
##################################
X_train_numeric_row_count_list = list([len(cirrhosis_survival_X_train_imputed_numeric)] * len(cirrhosis_survival_X_train_imputed_numeric.columns))
```


```python
##################################
# Gathering the unique to count ratio for each object column
##################################
X_train_numeric_outlier_ratio_list = map(truediv, X_train_numeric_outlier_count_list, X_train_numeric_row_count_list)
```


```python
##################################
# Formulating the outlier summary
# for all numeric columns
##################################
X_train_numeric_column_outlier_summary = pd.DataFrame(zip(X_train_numeric_variable_name_list,
                                                          X_train_numeric_skewness_list,
                                                          X_train_numeric_outlier_count_list,
                                                          X_train_numeric_row_count_list,
                                                          X_train_numeric_outlier_ratio_list), 
                                                      columns=['Numeric.Column.Name',
                                                               'Skewness',
                                                               'Outlier.Count',
                                                               'Row.Count',
                                                               'Outlier.Ratio'])
display(X_train_numeric_column_outlier_summary.sort_values(by=['Outlier.Count'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Skewness</th>
      <th>Outlier.Count</th>
      <th>Row.Count</th>
      <th>Outlier.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Alk_Phos</td>
      <td>3.035777</td>
      <td>25</td>
      <td>218</td>
      <td>0.114679</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bilirubin</td>
      <td>3.121255</td>
      <td>18</td>
      <td>218</td>
      <td>0.082569</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cholesterol</td>
      <td>3.760943</td>
      <td>17</td>
      <td>218</td>
      <td>0.077982</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Prothrombin</td>
      <td>1.009263</td>
      <td>12</td>
      <td>218</td>
      <td>0.055046</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Copper</td>
      <td>1.485547</td>
      <td>8</td>
      <td>218</td>
      <td>0.036697</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albumin</td>
      <td>-0.589651</td>
      <td>6</td>
      <td>218</td>
      <td>0.027523</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SGOT</td>
      <td>0.934535</td>
      <td>4</td>
      <td>218</td>
      <td>0.018349</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Tryglicerides</td>
      <td>2.817187</td>
      <td>4</td>
      <td>218</td>
      <td>0.018349</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Platelets</td>
      <td>0.374251</td>
      <td>4</td>
      <td>218</td>
      <td>0.018349</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>0.223080</td>
      <td>1</td>
      <td>218</td>
      <td>0.004587</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the individual boxplots
# for all numeric columns
##################################
for column in cirrhosis_survival_X_train_imputed_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=cirrhosis_survival_X_train_imputed_numeric, x=column)
```


    
![png](output_99_0.png)
    



    
![png](output_99_1.png)
    



    
![png](output_99_2.png)
    



    
![png](output_99_3.png)
    



    
![png](output_99_4.png)
    



    
![png](output_99_5.png)
    



    
![png](output_99_6.png)
    



    
![png](output_99_7.png)
    



    
![png](output_99_8.png)
    



    
![png](output_99_9.png)
    


### 1.4.4 Collinearity <a class="anchor" id="1.4.4"></a>

[Pearson’s Correlation Coefficient](https://royalsocietypublishing.org/doi/10.1098/rsta.1896.0007) is a parametric measure of the linear correlation for a pair of features by calculating the ratio between their covariance and the product of their standard deviations. The presence of high absolute correlation values indicate the univariate association between the numeric predictors and the numeric response.

1. All numeric variables in the training subset were retained since majority reported sufficiently moderate and statistically significant correlation with no excessive multicollinearity.
2. Among pairwise combinations of numeric variables in the training subset, the highest Pearson.Correlation.Coefficient values were noted for:
    * <span style="color: #FF0000">Birilubin</span> and <span style="color: #FF0000">Copper</span>: Pearson.Correlation.Coefficient = +0.503
    * <span style="color: #FF0000">Birilubin</span> and <span style="color: #FF0000">SGOT</span>: Pearson.Correlation.Coefficient = +0.444
    * <span style="color: #FF0000">Birilubin</span> and <span style="color: #FF0000">Tryglicerides</span>: Pearson.Correlation.Coefficient = +0.389
    * <span style="color: #FF0000">Birilubin</span> and <span style="color: #FF0000">Cholesterol</span>: Pearson.Correlation.Coefficient = +0.348
    * <span style="color: #FF0000">Birilubin</span> and <span style="color: #FF0000">Prothrombin</span>: Pearson.Correlation.Coefficient = +0.344


```python
##################################
# Formulating a function 
# to plot the correlation matrix
# for all pairwise combinations
# of numeric columns
##################################
def plot_correlation_matrix(corr, mask=None):
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, 
                ax=ax,
                mask=mask,
                annot=True, 
                vmin=-1, 
                vmax=1, 
                center=0,
                cmap='coolwarm', 
                linewidths=1, 
                linecolor='gray', 
                cbar_kws={'orientation': 'horizontal'}) 
```


```python
##################################
# Computing the correlation coefficients
# and correlation p-values
# among pairs of numeric columns
##################################
cirrhosis_survival_X_train_imputed_numeric_correlation_pairs = {}
cirrhosis_survival_X_train_imputed_numeric_columns = cirrhosis_survival_X_train_imputed_numeric.columns.tolist()
for numeric_column_a, numeric_column_b in itertools.combinations(cirrhosis_survival_X_train_imputed_numeric_columns, 2):
    cirrhosis_survival_X_train_imputed_numeric_correlation_pairs[numeric_column_a + '_' + numeric_column_b] = stats.pearsonr(
        cirrhosis_survival_X_train_imputed_numeric.loc[:, numeric_column_a], 
        cirrhosis_survival_X_train_imputed_numeric.loc[:, numeric_column_b])
```


```python
##################################
# Formulating the pairwise correlation summary
# for all numeric columns
##################################
cirrhosis_survival_X_train_imputed_numeric_summary = cirrhosis_survival_X_train_imputed_numeric.from_dict(cirrhosis_survival_X_train_imputed_numeric_correlation_pairs, orient='index')
cirrhosis_survival_X_train_imputed_numeric_summary.columns = ['Pearson.Correlation.Coefficient', 'Correlation.PValue']
display(cirrhosis_survival_X_train_imputed_numeric_summary.sort_values(by=['Pearson.Correlation.Coefficient'], ascending=False).head(20))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pearson.Correlation.Coefficient</th>
      <th>Correlation.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bilirubin_SGOT</th>
      <td>0.503007</td>
      <td>2.210899e-15</td>
    </tr>
    <tr>
      <th>Bilirubin_Copper</th>
      <td>0.444366</td>
      <td>5.768566e-12</td>
    </tr>
    <tr>
      <th>Bilirubin_Tryglicerides</th>
      <td>0.389493</td>
      <td>2.607951e-09</td>
    </tr>
    <tr>
      <th>Bilirubin_Cholesterol</th>
      <td>0.348174</td>
      <td>1.311597e-07</td>
    </tr>
    <tr>
      <th>Bilirubin_Prothrombin</th>
      <td>0.344724</td>
      <td>1.775156e-07</td>
    </tr>
    <tr>
      <th>Copper_SGOT</th>
      <td>0.305052</td>
      <td>4.475849e-06</td>
    </tr>
    <tr>
      <th>Cholesterol_SGOT</th>
      <td>0.280530</td>
      <td>2.635566e-05</td>
    </tr>
    <tr>
      <th>Alk_Phos_Tryglicerides</th>
      <td>0.265538</td>
      <td>7.199789e-05</td>
    </tr>
    <tr>
      <th>Cholesterol_Tryglicerides</th>
      <td>0.257973</td>
      <td>1.169491e-04</td>
    </tr>
    <tr>
      <th>Copper_Tryglicerides</th>
      <td>0.256448</td>
      <td>1.287335e-04</td>
    </tr>
    <tr>
      <th>Copper_Prothrombin</th>
      <td>0.232051</td>
      <td>5.528189e-04</td>
    </tr>
    <tr>
      <th>Copper_Alk_Phos</th>
      <td>0.215001</td>
      <td>1.404964e-03</td>
    </tr>
    <tr>
      <th>Alk_Phos_Platelets</th>
      <td>0.182762</td>
      <td>6.814702e-03</td>
    </tr>
    <tr>
      <th>SGOT_Tryglicerides</th>
      <td>0.176605</td>
      <td>8.972028e-03</td>
    </tr>
    <tr>
      <th>SGOT_Prothrombin</th>
      <td>0.170928</td>
      <td>1.147644e-02</td>
    </tr>
    <tr>
      <th>Albumin_Platelets</th>
      <td>0.170836</td>
      <td>1.152154e-02</td>
    </tr>
    <tr>
      <th>Cholesterol_Copper</th>
      <td>0.165834</td>
      <td>1.422873e-02</td>
    </tr>
    <tr>
      <th>Cholesterol_Alk_Phos</th>
      <td>0.165814</td>
      <td>1.424066e-02</td>
    </tr>
    <tr>
      <th>Age_Prothrombin</th>
      <td>0.157493</td>
      <td>1.999022e-02</td>
    </tr>
    <tr>
      <th>Cholesterol_Platelets</th>
      <td>0.152235</td>
      <td>2.458130e-02</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting the correlation matrix
# for all pairwise combinations
# of numeric columns
##################################
cirrhosis_survival_X_train_imputed_numeric_correlation = cirrhosis_survival_X_train_imputed_numeric.corr()
mask = np.triu(cirrhosis_survival_X_train_imputed_numeric_correlation)
plot_correlation_matrix(cirrhosis_survival_X_train_imputed_numeric_correlation,mask)
plt.show()
```


    
![png](output_104_0.png)
    



```python
##################################
# Formulating a function 
# to plot the correlation matrix
# for all pairwise combinations
# of numeric columns
# with significant p-values only
##################################
def correlation_significance(df=None):
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            _ , p = stats.pearsonr(df[col],df[col2])
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
    return p_matrix
```


```python
##################################
# Plotting the correlation matrix
# for all pairwise combinations
# of numeric columns
# with significant p-values only
##################################
cirrhosis_survival_X_train_imputed_numeric_correlation_p_values = correlation_significance(cirrhosis_survival_X_train_imputed_numeric)                     
mask = np.invert(np.tril(cirrhosis_survival_X_train_imputed_numeric_correlation_p_values<0.05)) 
plot_correlation_matrix(cirrhosis_survival_X_train_imputed_numeric_correlation,mask)
```


    
![png](output_106_0.png)
    


### 1.4.5 Shape Transformation <a class="anchor" id="1.4.5"></a>

[Yeo-Johnson Transformation](https://academic.oup.com/biomet/article-abstract/87/4/954/232908?redirectedFrom=fulltext&login=false) applies a new family of distributions that can be used without restrictions, extending many of the good properties of the Box-Cox power family. Similar to the Box-Cox transformation, the method also estimates the optimal value of lambda but has the ability to transform both positive and negative values by inflating low variance data and deflating high variance data to create a more uniform data set. While there are no restrictions in terms of the applicable values, the interpretability of the transformed values is more diminished as compared to the other methods.

1. A Yeo-Johnson transformation was applied to all numeric variables in the training subset to improve distributional shape.
2. Most variables in the training subset achieved symmetrical distributions with minimal outliers after transformation.
    * <span style="color: #FF0000">Cholesterol</span>: Outlier.Count = 9, Outlier.Ratio = 0.041, Skewness=-0.083
    * <span style="color: #FF0000">Albumin</span>: Outlier.Count = 4, Outlier.Ratio = 0.018, Skewness=+0.006
    * <span style="color: #FF0000">Platelets</span>: Outlier.Count = 2, Outlier.Ratio = 0.009, Skewness=-0.019
    * <span style="color: #FF0000">Age</span>: Outlier.Count = 1, Outlier.Ratio = 0.004, Skewness=+0.223
    * <span style="color: #FF0000">Copper</span>: Outlier.Count = 1, Outlier.Ratio = 0.004, Skewness=-0.010
    * <span style="color: #FF0000">Alk_Phos</span>: Outlier.Count = 1, Outlier.Ratio = 0.004, Skewness=+0.027
    * <span style="color: #FF0000">SGOT</span>: Outlier.Count = 1, Outlier.Ratio = 0.004, Skewness=-0.001
    * <span style="color: #FF0000">Tryglicerides</span>: Outlier.Count = 1, Outlier.Ratio = 0.004, Skewness=+0.000
3. Outlier data in the testing subset for numeric variables will be treated with Yeo-Johnson transformation downstream using a pipeline involving the final preprocessing steps.



```python
##################################
# Formulating a data subset containing
# variables with noted outliers
##################################
X_train_predictors_with_outliers = ['Bilirubin','Cholesterol','Albumin','Copper','Alk_Phos','SGOT','Tryglicerides','Platelets','Prothrombin']
cirrhosis_survival_X_train_imputed_numeric_with_outliers = cirrhosis_survival_X_train_imputed_numeric[X_train_predictors_with_outliers]
```


```python
##################################
# Conducting a Yeo-Johnson Transformation
# to address the distributional
# shape of the variables
##################################
yeo_johnson_transformer = PowerTransformer(method='yeo-johnson',
                                          standardize=False)
cirrhosis_survival_X_train_imputed_numeric_with_outliers_array = yeo_johnson_transformer.fit_transform(cirrhosis_survival_X_train_imputed_numeric_with_outliers)
```


```python
##################################
# Formulating a new dataset object
# for the transformed data
##################################
cirrhosis_survival_X_train_transformed_numeric_with_outliers = pd.DataFrame(cirrhosis_survival_X_train_imputed_numeric_with_outliers_array,
                                                                            columns=cirrhosis_survival_X_train_imputed_numeric_with_outliers.columns)
cirrhosis_survival_X_train_transformed_numeric = pd.concat([cirrhosis_survival_X_train_imputed_numeric[['Age']],
                                                            cirrhosis_survival_X_train_transformed_numeric_with_outliers], 
                                                           axis=1)
```


```python
cirrhosis_survival_X_train_transformed_numeric.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13329</td>
      <td>0.830251</td>
      <td>1.528771</td>
      <td>25.311621</td>
      <td>4.367652</td>
      <td>2.066062</td>
      <td>7.115310</td>
      <td>3.357597</td>
      <td>58.787709</td>
      <td>0.236575</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12912</td>
      <td>0.751147</td>
      <td>1.535175</td>
      <td>34.049208</td>
      <td>6.244827</td>
      <td>2.047167</td>
      <td>7.303237</td>
      <td>3.581345</td>
      <td>57.931137</td>
      <td>0.236572</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17180</td>
      <td>0.491099</td>
      <td>1.523097</td>
      <td>32.812930</td>
      <td>5.320861</td>
      <td>2.043970</td>
      <td>7.278682</td>
      <td>2.990077</td>
      <td>61.554228</td>
      <td>0.236573</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17884</td>
      <td>0.760957</td>
      <td>1.505628</td>
      <td>30.818146</td>
      <td>5.264915</td>
      <td>2.062590</td>
      <td>7.170942</td>
      <td>3.288822</td>
      <td>29.648190</td>
      <td>0.236575</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15177</td>
      <td>0.893603</td>
      <td>1.519249</td>
      <td>26.533792</td>
      <td>6.440904</td>
      <td>2.109170</td>
      <td>8.385199</td>
      <td>3.284118</td>
      <td>43.198326</td>
      <td>0.236572</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Formulating the individual boxplots
# for all transformed numeric columns
##################################
for column in cirrhosis_survival_X_train_transformed_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=cirrhosis_survival_X_train_transformed_numeric, x=column)
```


    
![png](output_112_0.png)
    



    
![png](output_112_1.png)
    



    
![png](output_112_2.png)
    



    
![png](output_112_3.png)
    



    
![png](output_112_4.png)
    



    
![png](output_112_5.png)
    



    
![png](output_112_6.png)
    



    
![png](output_112_7.png)
    



    
![png](output_112_8.png)
    



    
![png](output_112_9.png)
    



```python
##################################
# Formulating the outlier summary
# for all numeric columns
##################################
X_train_numeric_variable_name_list = list(cirrhosis_survival_X_train_transformed_numeric.columns)
X_train_numeric_skewness_list = cirrhosis_survival_X_train_transformed_numeric.skew()
cirrhosis_survival_X_train_transformed_numeric_q1 = cirrhosis_survival_X_train_transformed_numeric.quantile(0.25)
cirrhosis_survival_X_train_transformed_numeric_q3 = cirrhosis_survival_X_train_transformed_numeric.quantile(0.75)
cirrhosis_survival_X_train_transformed_numeric_iqr = cirrhosis_survival_X_train_transformed_numeric_q3 - cirrhosis_survival_X_train_transformed_numeric_q1
X_train_numeric_outlier_count_list = ((cirrhosis_survival_X_train_transformed_numeric < (cirrhosis_survival_X_train_transformed_numeric_q1 - 1.5 * cirrhosis_survival_X_train_transformed_numeric_iqr)) | (cirrhosis_survival_X_train_transformed_numeric > (cirrhosis_survival_X_train_transformed_numeric_q3 + 1.5 * cirrhosis_survival_X_train_transformed_numeric_iqr))).sum()
X_train_numeric_row_count_list = list([len(cirrhosis_survival_X_train_transformed_numeric)] * len(cirrhosis_survival_X_train_transformed_numeric.columns))
X_train_numeric_outlier_ratio_list = map(truediv, X_train_numeric_outlier_count_list, X_train_numeric_row_count_list)

X_train_numeric_column_outlier_summary = pd.DataFrame(zip(X_train_numeric_variable_name_list,
                                                          X_train_numeric_skewness_list,
                                                          X_train_numeric_outlier_count_list,
                                                          X_train_numeric_row_count_list,
                                                          X_train_numeric_outlier_ratio_list),                                                      
                                        columns=['Numeric.Column.Name',
                                                 'Skewness',
                                                 'Outlier.Count',
                                                 'Row.Count',
                                                 'Outlier.Ratio'])
display(X_train_numeric_column_outlier_summary.sort_values(by=['Outlier.Count'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Skewness</th>
      <th>Outlier.Count</th>
      <th>Row.Count</th>
      <th>Outlier.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Cholesterol</td>
      <td>-0.083072</td>
      <td>9</td>
      <td>218</td>
      <td>0.041284</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albumin</td>
      <td>0.006523</td>
      <td>4</td>
      <td>218</td>
      <td>0.018349</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Platelets</td>
      <td>-0.019323</td>
      <td>2</td>
      <td>218</td>
      <td>0.009174</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>0.223080</td>
      <td>1</td>
      <td>218</td>
      <td>0.004587</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Copper</td>
      <td>-0.010240</td>
      <td>1</td>
      <td>218</td>
      <td>0.004587</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Alk_Phos</td>
      <td>0.027977</td>
      <td>1</td>
      <td>218</td>
      <td>0.004587</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Tryglicerides</td>
      <td>-0.000881</td>
      <td>1</td>
      <td>218</td>
      <td>0.004587</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Prothrombin</td>
      <td>0.000000</td>
      <td>1</td>
      <td>218</td>
      <td>0.004587</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bilirubin</td>
      <td>0.263101</td>
      <td>0</td>
      <td>218</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SGOT</td>
      <td>-0.008416</td>
      <td>0</td>
      <td>218</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


### 1.4.6 Centering and Scaling <a class="anchor" id="1.4.6"></a>

1. All numeric variables in the training subset were transformed using the standardization method to achieve a comparable scale between values.
2. Original data in the testing subset for numeric variables will be treated with standardization scaling downstream using a pipeline involving the final preprocessing steps.


```python
##################################
# Conducting standardization
# to transform the values of the 
# variables into comparable scale
##################################
standardization_scaler = StandardScaler()
cirrhosis_survival_X_train_transformed_numeric_array = standardization_scaler.fit_transform(cirrhosis_survival_X_train_transformed_numeric)
```


```python
##################################
# Formulating a new dataset object
# for the scaled data
##################################
cirrhosis_survival_X_train_scaled_numeric = pd.DataFrame(cirrhosis_survival_X_train_transformed_numeric_array,
                                                         columns=cirrhosis_survival_X_train_transformed_numeric.columns)
```


```python
##################################
# Formulating the individual boxplots
# for all transformed numeric columns
##################################
for column in cirrhosis_survival_X_train_scaled_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=cirrhosis_survival_X_train_scaled_numeric, x=column)
```


    
![png](output_117_0.png)
    



    
![png](output_117_1.png)
    



    
![png](output_117_2.png)
    



    
![png](output_117_3.png)
    



    
![png](output_117_4.png)
    



    
![png](output_117_5.png)
    



    
![png](output_117_6.png)
    



    
![png](output_117_7.png)
    



    
![png](output_117_8.png)
    



    
![png](output_117_9.png)
    


### 1.4.7 Data Encoding <a class="anchor" id="1.4.7"></a>

1. Binary encoding was applied to the predictor object columns in the training subset:
    * <span style="color: #FF0000">Status</span>
    * <span style="color: #FF0000">Drug</span>
    * <span style="color: #FF0000">Sex</span>
    * <span style="color: #FF0000">Ascites</span>
    * <span style="color: #FF0000">Hepatomegaly</span>
    * <span style="color: #FF0000">Spiders</span>
    * <span style="color: #FF0000">Edema</span>
1. One-hot encoding was applied to the <span style="color: #FF0000">Stage</span> variable resulting to 4 additional columns in the training subset:
    * <span style="color: #FF0000">Stage_1.0</span>
    * <span style="color: #FF0000">Stage_2.0</span>
    * <span style="color: #FF0000">Stage_3.0</span>
    * <span style="color: #FF0000">Stage_4.0</span>
3. Original data in the testing subset for object variables will be treated with binary and one-hot encoding downstream using a pipeline involving the final preprocessing steps.


```python
##################################
# Applying a binary encoding transformation
# for the two-level object columns
##################################
cirrhosis_survival_X_train_cleaned_object['Sex'] = cirrhosis_survival_X_train_cleaned_object['Sex'].replace({'M':0, 'F':1}) 
cirrhosis_survival_X_train_cleaned_object['Ascites'] = cirrhosis_survival_X_train_cleaned_object['Ascites'].replace({'N':0, 'Y':1}) 
cirrhosis_survival_X_train_cleaned_object['Drug'] = cirrhosis_survival_X_train_cleaned_object['Drug'].replace({'Placebo':0, 'D-penicillamine':1}) 
cirrhosis_survival_X_train_cleaned_object['Hepatomegaly'] = cirrhosis_survival_X_train_cleaned_object['Hepatomegaly'].replace({'N':0, 'Y':1}) 
cirrhosis_survival_X_train_cleaned_object['Spiders'] = cirrhosis_survival_X_train_cleaned_object['Spiders'].replace({'N':0, 'Y':1}) 
cirrhosis_survival_X_train_cleaned_object['Edema'] = cirrhosis_survival_X_train_cleaned_object['Edema'].replace({'N':0, 'Y':1, 'S':1}) 
```


```python
##################################
# Formulating the multi-level object column stage
# for encoding transformation
##################################
cirrhosis_survival_X_train_cleaned_object_stage_encoded = pd.DataFrame(cirrhosis_survival_X_train_cleaned_object.loc[:, 'Stage'].to_list(),
                                                                       columns=['Stage'])
```


```python
##################################
# Applying a one-hot encoding transformation
# for the multi-level object column stage
##################################
cirrhosis_survival_X_train_cleaned_object_stage_encoded = pd.get_dummies(cirrhosis_survival_X_train_cleaned_object_stage_encoded, columns=['Stage'])
```


```python
##################################
# Applying a one-hot encoding transformation
# for the multi-level object column stage
##################################
cirrhosis_survival_X_train_cleaned_encoded_object = pd.concat([cirrhosis_survival_X_train_cleaned_object.drop(['Stage'], axis=1), 
                                                               cirrhosis_survival_X_train_cleaned_object_stage_encoded], axis=1)
cirrhosis_survival_X_train_cleaned_encoded_object.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_1.0</th>
      <th>Stage_2.0</th>
      <th>Stage_3.0</th>
      <th>Stage_4.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### 1.4.8 Preprocessed Data Description <a class="anchor" id="1.4.8"></a>

1. A preprocessing pipeline was formulated to standardize the data transformation methods applied to both the training and testing subsets.
2. The preprocessed training subset is comprised of:
    * **218 rows** (observations)
    * **22 columns** (variables)
        * **2/22 event | duration** (boolean | numeric)
             * <span style="color: #FF0000">Status</span>
             * <span style="color: #FF0000">N_Days</span>
        * **10/22 predictor** (numeric)
             * <span style="color: #FF0000">Age</span>
             * <span style="color: #FF0000">Bilirubin</span>
             * <span style="color: #FF0000">Cholesterol</span>
             * <span style="color: #FF0000">Albumin</span>
             * <span style="color: #FF0000">Copper</span>
             * <span style="color: #FF0000">Alk_Phos</span>
             * <span style="color: #FF0000">SGOT</span>
             * <span style="color: #FF0000">Triglycerides</span>
             * <span style="color: #FF0000">Platelets</span>
             * <span style="color: #FF0000">Prothrombin</span>
        * **10/21 predictor** (object)
             * <span style="color: #FF0000">Drug</span>
             * <span style="color: #FF0000">Sex</span>
             * <span style="color: #FF0000">Ascites</span>
             * <span style="color: #FF0000">Hepatomegaly</span>
             * <span style="color: #FF0000">Spiders</span>
             * <span style="color: #FF0000">Edema</span>
             * <span style="color: #FF0000">Stage_1.0</span>
             * <span style="color: #FF0000">Stage_2.0</span>
             * <span style="color: #FF0000">Stage_3.0</span>
             * <span style="color: #FF0000">Stage_4.0</span>
3. The preprocessed testing subset is comprised of:
    * **94 rows** (observations)
    * **22 columns** (variables)
        * **2/22 event | duration** (boolean | numeric)
             * <span style="color: #FF0000">Status</span>
             * <span style="color: #FF0000">N_Days</span>
        * **10/22 predictor** (numeric)
             * <span style="color: #FF0000">Age</span>
             * <span style="color: #FF0000">Bilirubin</span>
             * <span style="color: #FF0000">Cholesterol</span>
             * <span style="color: #FF0000">Albumin</span>
             * <span style="color: #FF0000">Copper</span>
             * <span style="color: #FF0000">Alk_Phos</span>
             * <span style="color: #FF0000">SGOT</span>
             * <span style="color: #FF0000">Triglycerides</span>
             * <span style="color: #FF0000">Platelets</span>
             * <span style="color: #FF0000">Prothrombin</span>
        * **10/21 predictor** (object)
             * <span style="color: #FF0000">Drug</span>
             * <span style="color: #FF0000">Sex</span>
             * <span style="color: #FF0000">Ascites</span>
             * <span style="color: #FF0000">Hepatomegaly</span>
             * <span style="color: #FF0000">Spiders</span>
             * <span style="color: #FF0000">Edema</span>
             * <span style="color: #FF0000">Stage_1.0</span>
             * <span style="color: #FF0000">Stage_2.0</span>
             * <span style="color: #FF0000">Stage_3.0</span>
             * <span style="color: #FF0000">Stage_4.0</span>


```python
##################################
# Consolidating all preprocessed
# numeric and object predictors
# for the training subset
##################################
cirrhosis_survival_X_train_preprocessed = pd.concat([cirrhosis_survival_X_train_scaled_numeric,
                                                     cirrhosis_survival_X_train_cleaned_encoded_object], 
                                                     axis=1)
cirrhosis_survival_X_train_preprocessed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_1.0</th>
      <th>Stage_2.0</th>
      <th>Stage_3.0</th>
      <th>Stage_4.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.296446</td>
      <td>0.863802</td>
      <td>0.885512</td>
      <td>-0.451884</td>
      <td>-0.971563</td>
      <td>0.140990</td>
      <td>0.104609</td>
      <td>0.155256</td>
      <td>0.539120</td>
      <td>0.747580</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.405311</td>
      <td>0.516350</td>
      <td>1.556983</td>
      <td>0.827618</td>
      <td>0.468389</td>
      <td>-0.705337</td>
      <td>0.301441</td>
      <td>1.275281</td>
      <td>0.472266</td>
      <td>-0.315794</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.291081</td>
      <td>-0.625875</td>
      <td>0.290561</td>
      <td>0.646582</td>
      <td>-0.240371</td>
      <td>-0.848544</td>
      <td>0.275723</td>
      <td>-1.684460</td>
      <td>0.755044</td>
      <td>0.087130</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.107291</td>
      <td>0.559437</td>
      <td>-1.541148</td>
      <td>0.354473</td>
      <td>-0.283286</td>
      <td>-0.014525</td>
      <td>0.162878</td>
      <td>-0.189015</td>
      <td>-1.735183</td>
      <td>0.649171</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.813996</td>
      <td>1.142068</td>
      <td>-0.112859</td>
      <td>-0.272913</td>
      <td>0.618797</td>
      <td>2.071847</td>
      <td>1.434674</td>
      <td>-0.212560</td>
      <td>-0.677612</td>
      <td>-0.315794</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Creating a pre-processing pipeline
# for numeric predictors
##################################
cirrhosis_survival_numeric_predictors = ['Age', 'Bilirubin','Cholesterol', 'Albumin','Copper', 'Alk_Phos','SGOT', 'Tryglicerides','Platelets', 'Prothrombin']
numeric_transformer = Pipeline(steps=[
    ('imputer', IterativeImputer(estimator = lr,
                                 max_iter = 10,
                                 tol = 1e-10,
                                 imputation_order = 'ascending',
                                 random_state=88888888)),
    ('yeo_johnson', PowerTransformer(method='yeo-johnson',
                                    standardize=False)),
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, cirrhosis_survival_numeric_predictors)])
```


```python
##################################
# Fitting and transforming 
# training subset numeric predictors
##################################
cirrhosis_survival_X_train_numeric_preprocessed = preprocessor.fit_transform(cirrhosis_survival_X_train_cleaned)
cirrhosis_survival_X_train_numeric_preprocessed = pd.DataFrame(cirrhosis_survival_X_train_numeric_preprocessed,
                                                                columns=cirrhosis_survival_numeric_predictors)
cirrhosis_survival_X_train_numeric_preprocessed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.342097</td>
      <td>0.863802</td>
      <td>0.886087</td>
      <td>-0.451884</td>
      <td>-0.972098</td>
      <td>0.140990</td>
      <td>0.104609</td>
      <td>0.155130</td>
      <td>0.540960</td>
      <td>0.747580</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.470901</td>
      <td>0.516350</td>
      <td>1.554523</td>
      <td>0.827618</td>
      <td>0.467579</td>
      <td>-0.705337</td>
      <td>0.301441</td>
      <td>1.275222</td>
      <td>0.474140</td>
      <td>-0.315794</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.239814</td>
      <td>-0.625875</td>
      <td>0.293280</td>
      <td>0.646582</td>
      <td>-0.241205</td>
      <td>-0.848544</td>
      <td>0.275723</td>
      <td>-1.684460</td>
      <td>0.756741</td>
      <td>0.087130</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.052733</td>
      <td>0.559437</td>
      <td>-1.534283</td>
      <td>0.354473</td>
      <td>-0.284113</td>
      <td>-0.014525</td>
      <td>0.162878</td>
      <td>-0.189139</td>
      <td>-1.735375</td>
      <td>0.649171</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.795010</td>
      <td>1.142068</td>
      <td>-0.108933</td>
      <td>-0.272913</td>
      <td>0.618030</td>
      <td>2.071847</td>
      <td>1.434674</td>
      <td>-0.212684</td>
      <td>-0.675951</td>
      <td>-0.315794</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Performing pre-processing operations
# for object predictors
# in the training subset
##################################
cirrhosis_survival_object_predictors = ['Drug', 'Sex','Ascites', 'Hepatomegaly','Spiders', 'Edema','Stage']
cirrhosis_survival_X_train_object = cirrhosis_survival_X_train_cleaned.copy()
cirrhosis_survival_X_train_object = cirrhosis_survival_X_train_object[cirrhosis_survival_object_predictors]
cirrhosis_survival_X_train_object.reset_index(drop=True, inplace=True)
cirrhosis_survival_X_train_object.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>D-penicillamine</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>D-penicillamine</td>
      <td>M</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>D-penicillamine</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Placebo</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Placebo</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>N</td>
      <td>N</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Applying a binary encoding transformation
# for the two-level object columns
# in the training subset
##################################
cirrhosis_survival_X_train_object['Sex'].replace({'M':0, 'F':1}, inplace=True) 
cirrhosis_survival_X_train_object['Ascites'].replace({'N':0, 'Y':1}, inplace=True) 
cirrhosis_survival_X_train_object['Drug'].replace({'Placebo':0, 'D-penicillamine':1}, inplace=True) 
cirrhosis_survival_X_train_object['Hepatomegaly'].replace({'N':0, 'Y':1}, inplace=True) 
cirrhosis_survival_X_train_object['Spiders'].replace({'N':0, 'Y':1}, inplace=True) 
cirrhosis_survival_X_train_object['Edema'].replace({'N':0, 'Y':1, 'S':1}, inplace=True) 
cirrhosis_survival_X_train_object_stage_encoded = pd.DataFrame(cirrhosis_survival_X_train_object.loc[:, 'Stage'].to_list(),
                                                                       columns=['Stage'])
cirrhosis_survival_X_train_object_stage_encoded = pd.get_dummies(cirrhosis_survival_X_train_object_stage_encoded, columns=['Stage'])
cirrhosis_survival_X_train_object_preprocessed = pd.concat([cirrhosis_survival_X_train_object.drop(['Stage'], axis=1), 
                                                            cirrhosis_survival_X_train_object_stage_encoded], 
                                                           axis=1)
cirrhosis_survival_X_train_object_preprocessed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_1.0</th>
      <th>Stage_2.0</th>
      <th>Stage_3.0</th>
      <th>Stage_4.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Consolidating the preprocessed
# training subset
##################################
cirrhosis_survival_X_train_preprocessed = pd.concat([cirrhosis_survival_X_train_numeric_preprocessed, cirrhosis_survival_X_train_object_preprocessed], 
                                                    axis=1)
cirrhosis_survival_X_train_preprocessed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_1.0</th>
      <th>Stage_2.0</th>
      <th>Stage_3.0</th>
      <th>Stage_4.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.342097</td>
      <td>0.863802</td>
      <td>0.886087</td>
      <td>-0.451884</td>
      <td>-0.972098</td>
      <td>0.140990</td>
      <td>0.104609</td>
      <td>0.155130</td>
      <td>0.540960</td>
      <td>0.747580</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.470901</td>
      <td>0.516350</td>
      <td>1.554523</td>
      <td>0.827618</td>
      <td>0.467579</td>
      <td>-0.705337</td>
      <td>0.301441</td>
      <td>1.275222</td>
      <td>0.474140</td>
      <td>-0.315794</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.239814</td>
      <td>-0.625875</td>
      <td>0.293280</td>
      <td>0.646582</td>
      <td>-0.241205</td>
      <td>-0.848544</td>
      <td>0.275723</td>
      <td>-1.684460</td>
      <td>0.756741</td>
      <td>0.087130</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.052733</td>
      <td>0.559437</td>
      <td>-1.534283</td>
      <td>0.354473</td>
      <td>-0.284113</td>
      <td>-0.014525</td>
      <td>0.162878</td>
      <td>-0.189139</td>
      <td>-1.735375</td>
      <td>0.649171</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.795010</td>
      <td>1.142068</td>
      <td>-0.108933</td>
      <td>-0.272913</td>
      <td>0.618030</td>
      <td>2.071847</td>
      <td>1.434674</td>
      <td>-0.212684</td>
      <td>-0.675951</td>
      <td>-0.315794</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Verifying the dimensions of the
# preprocessed training subset
##################################
cirrhosis_survival_X_train_preprocessed.shape
```




    (218, 20)




```python
##################################
# Fitting and transforming 
# testing subset numeric predictors
##################################
cirrhosis_survival_X_test_numeric_preprocessed = preprocessor.transform(cirrhosis_survival_X_test_cleaned)
cirrhosis_survival_X_test_numeric_preprocessed = pd.DataFrame(cirrhosis_survival_X_test_numeric_preprocessed,
                                                                columns=cirrhosis_survival_numeric_predictors)
cirrhosis_survival_X_test_numeric_preprocessed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.043704</td>
      <td>0.744396</td>
      <td>0.922380</td>
      <td>0.240951</td>
      <td>0.045748</td>
      <td>0.317282</td>
      <td>-0.078335</td>
      <td>2.671950</td>
      <td>1.654815</td>
      <td>-0.948196</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.936476</td>
      <td>-0.764558</td>
      <td>0.160096</td>
      <td>-0.600950</td>
      <td>-0.179138</td>
      <td>-0.245613</td>
      <td>0.472422</td>
      <td>-0.359800</td>
      <td>0.348533</td>
      <td>0.439089</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.749033</td>
      <td>0.371523</td>
      <td>0.558115</td>
      <td>0.646582</td>
      <td>-0.159024</td>
      <td>0.339454</td>
      <td>0.685117</td>
      <td>-3.109146</td>
      <td>-0.790172</td>
      <td>-0.617113</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.485150</td>
      <td>-0.918484</td>
      <td>-0.690904</td>
      <td>1.629765</td>
      <td>0.028262</td>
      <td>1.713791</td>
      <td>-1.387751</td>
      <td>0.155130</td>
      <td>0.679704</td>
      <td>0.087130</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.815655</td>
      <td>1.286438</td>
      <td>2.610501</td>
      <td>-0.722153</td>
      <td>0.210203</td>
      <td>0.602860</td>
      <td>3.494936</td>
      <td>-0.053214</td>
      <td>-0.475634</td>
      <td>-1.714435</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Performing pre-processing operations
# for object predictors
# in the testing subset
##################################
cirrhosis_survival_object_predictors = ['Drug', 'Sex','Ascites', 'Hepatomegaly','Spiders', 'Edema','Stage']
cirrhosis_survival_X_test_object = cirrhosis_survival_X_test_cleaned.copy()
cirrhosis_survival_X_test_object = cirrhosis_survival_X_test_object[cirrhosis_survival_object_predictors]
cirrhosis_survival_X_test_object.reset_index(drop=True, inplace=True)
cirrhosis_survival_X_test_object.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>D-penicillamine</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>Y</td>
      <td>S</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Placebo</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>D-penicillamine</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>N</td>
      <td>N</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D-penicillamine</td>
      <td>M</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Placebo</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>N</td>
      <td>N</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Applying a binary encoding transformation
# for the two-level object columns
# in the testing subset
##################################
cirrhosis_survival_X_test_object['Sex'].replace({'M':0, 'F':1}, inplace=True) 
cirrhosis_survival_X_test_object['Ascites'].replace({'N':0, 'Y':1}, inplace=True) 
cirrhosis_survival_X_test_object['Drug'].replace({'Placebo':0, 'D-penicillamine':1}, inplace=True) 
cirrhosis_survival_X_test_object['Hepatomegaly'].replace({'N':0, 'Y':1}, inplace=True) 
cirrhosis_survival_X_test_object['Spiders'].replace({'N':0, 'Y':1}, inplace=True) 
cirrhosis_survival_X_test_object['Edema'].replace({'N':0, 'Y':1, 'S':1}, inplace=True) 
cirrhosis_survival_X_test_object_stage_encoded = pd.DataFrame(cirrhosis_survival_X_test_object.loc[:, 'Stage'].to_list(),
                                                                       columns=['Stage'])
cirrhosis_survival_X_test_object_stage_encoded = pd.get_dummies(cirrhosis_survival_X_test_object_stage_encoded, columns=['Stage'])
cirrhosis_survival_X_test_object_preprocessed = pd.concat([cirrhosis_survival_X_test_object.drop(['Stage'], axis=1), 
                                                            cirrhosis_survival_X_test_object_stage_encoded], 
                                                           axis=1)
cirrhosis_survival_X_test_object_preprocessed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_1.0</th>
      <th>Stage_2.0</th>
      <th>Stage_3.0</th>
      <th>Stage_4.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Consolidating the preprocessed
# testing subset
##################################
cirrhosis_survival_X_test_preprocessed = pd.concat([cirrhosis_survival_X_test_numeric_preprocessed, cirrhosis_survival_X_test_object_preprocessed], 
                                                    axis=1)
cirrhosis_survival_X_test_preprocessed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_1.0</th>
      <th>Stage_2.0</th>
      <th>Stage_3.0</th>
      <th>Stage_4.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.043704</td>
      <td>0.744396</td>
      <td>0.922380</td>
      <td>0.240951</td>
      <td>0.045748</td>
      <td>0.317282</td>
      <td>-0.078335</td>
      <td>2.671950</td>
      <td>1.654815</td>
      <td>-0.948196</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.936476</td>
      <td>-0.764558</td>
      <td>0.160096</td>
      <td>-0.600950</td>
      <td>-0.179138</td>
      <td>-0.245613</td>
      <td>0.472422</td>
      <td>-0.359800</td>
      <td>0.348533</td>
      <td>0.439089</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.749033</td>
      <td>0.371523</td>
      <td>0.558115</td>
      <td>0.646582</td>
      <td>-0.159024</td>
      <td>0.339454</td>
      <td>0.685117</td>
      <td>-3.109146</td>
      <td>-0.790172</td>
      <td>-0.617113</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.485150</td>
      <td>-0.918484</td>
      <td>-0.690904</td>
      <td>1.629765</td>
      <td>0.028262</td>
      <td>1.713791</td>
      <td>-1.387751</td>
      <td>0.155130</td>
      <td>0.679704</td>
      <td>0.087130</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.815655</td>
      <td>1.286438</td>
      <td>2.610501</td>
      <td>-0.722153</td>
      <td>0.210203</td>
      <td>0.602860</td>
      <td>3.494936</td>
      <td>-0.053214</td>
      <td>-0.475634</td>
      <td>-1.714435</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Verifying the dimensions of the
# preprocessed testing subset
##################################
cirrhosis_survival_X_test_preprocessed.shape
```




    (94, 20)



## 1.5. Data Exploration <a class="anchor" id="1.5"></a>

### 1.5.1 Exploratory Data Analysis <a class="anchor" id="1.5.1"></a>

1. The estimated baseline survival plot indicated a 50% survival rate at <span style="color: #FF0000">N_Days=3358</span>.
2. Bivariate analysis identified individual predictors with potential association to the event status based on visual inspection.
    * Higher values for the following numeric predictors are associated with <span style="color: #FF0000">Status=True</span>: 
        * <span style="color: #FF0000">Age</span>
        * <span style="color: #FF0000">Bilirubin</span>   
        * <span style="color: #FF0000">Copper</span>
        * <span style="color: #FF0000">Alk_Phos</span> 
        * <span style="color: #FF0000">SGOT</span>   
        * <span style="color: #FF0000">Tryglicerides</span> 
        * <span style="color: #FF0000">Prothrombin</span>    
    * Higher counts for the following object predictors are associated with better differentiation between <span style="color: #FF0000">Status=True</span> and <span style="color: #FF0000">Status=False</span>:  
        * <span style="color: #FF0000">Drug</span>
        * <span style="color: #FF0000">Sex</span>
        * <span style="color: #FF0000">Ascites</span>
        * <span style="color: #FF0000">Hepatomegaly</span>
        * <span style="color: #FF0000">Spiders</span>
        * <span style="color: #FF0000">Edema</span>
        * <span style="color: #FF0000">Stage_1.0</span>
        * <span style="color: #FF0000">Stage_2.0</span>
        * <span style="color: #FF0000">Stage_3.0</span>
        * <span style="color: #FF0000">Stage_4.0</span>
2. Bivariate analysis identified individual predictors with potential association to the survival time based on visual inspection.
    * Higher values for the following numeric predictors are positively associated with <span style="color: #FF0000">N_Days</span>: 
        * <span style="color: #FF0000">Albumin</span>        
        * <span style="color: #FF0000">Platelets</span>
    * Levels for the following object predictors are associated with differences in <span style="color: #FF0000">N_Days</span> between <span style="color: #FF0000">Status=True</span> and <span style="color: #FF0000">Status=False</span>:  
        * <span style="color: #FF0000">Drug</span>
        * <span style="color: #FF0000">Sex</span>
        * <span style="color: #FF0000">Ascites</span>
        * <span style="color: #FF0000">Hepatomegaly</span>
        * <span style="color: #FF0000">Spiders</span>
        * <span style="color: #FF0000">Edema</span>
        * <span style="color: #FF0000">Stage_1.0</span>
        * <span style="color: #FF0000">Stage_2.0</span>
        * <span style="color: #FF0000">Stage_3.0</span>
        * <span style="color: #FF0000">Stage_4.0</span>


```python
##################################
# Formulating a complete dataframe
# from the training subset for EDA
##################################
cirrhosis_survival_y_train_cleaned.reset_index(drop=True, inplace=True)
cirrhosis_survival_train_EDA = pd.concat([cirrhosis_survival_y_train_cleaned,
                                          cirrhosis_survival_X_train_preprocessed],
                                         axis=1)
cirrhosis_survival_train_EDA.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Status</th>
      <th>N_Days</th>
      <th>Age</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>...</th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_1.0</th>
      <th>Stage_2.0</th>
      <th>Stage_3.0</th>
      <th>Stage_4.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>2475</td>
      <td>-1.342097</td>
      <td>0.863802</td>
      <td>0.886087</td>
      <td>-0.451884</td>
      <td>-0.972098</td>
      <td>0.140990</td>
      <td>0.104609</td>
      <td>0.155130</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>877</td>
      <td>-1.470901</td>
      <td>0.516350</td>
      <td>1.554523</td>
      <td>0.827618</td>
      <td>0.467579</td>
      <td>-0.705337</td>
      <td>0.301441</td>
      <td>1.275222</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>3050</td>
      <td>-0.239814</td>
      <td>-0.625875</td>
      <td>0.293280</td>
      <td>0.646582</td>
      <td>-0.241205</td>
      <td>-0.848544</td>
      <td>0.275723</td>
      <td>-1.684460</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>True</td>
      <td>110</td>
      <td>-0.052733</td>
      <td>0.559437</td>
      <td>-1.534283</td>
      <td>0.354473</td>
      <td>-0.284113</td>
      <td>-0.014525</td>
      <td>0.162878</td>
      <td>-0.189139</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>True</td>
      <td>3839</td>
      <td>-0.795010</td>
      <td>1.142068</td>
      <td>-0.108933</td>
      <td>-0.272913</td>
      <td>0.618030</td>
      <td>2.071847</td>
      <td>1.434674</td>
      <td>-0.212684</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
##################################
# Plotting the baseline survival curve
# and computing the survival rates
##################################
kmf = KaplanMeierFitter()
kmf.fit(durations=cirrhosis_survival_train_EDA['N_Days'], event_observed=cirrhosis_survival_train_EDA['Status'])
plt.figure(figsize=(17, 8))
kmf.plot_survival_function()
plt.title('Kaplan-Meier Baseline Survival Plot')
plt.ylim(0, 1.05)
plt.xlabel('N_Days')
plt.ylabel('Event Survival Probability')

##################################
# Determing the at-risk numbers
##################################
at_risk_counts = kmf.event_table.at_risk
survival_probabilities = kmf.survival_function_.values.flatten()
time_points = kmf.survival_function_.index
for time, prob, at_risk in zip(time_points, survival_probabilities, at_risk_counts):
    if time % 50 == 0: 
        plt.text(time, prob, f'{prob:.2f} : {at_risk}', ha='left', fontsize=10)
median_survival_time = kmf.median_survival_time_
plt.axvline(x=median_survival_time, color='r', linestyle='--')
plt.axhline(y=0.5, color='r', linestyle='--')
plt.text(3400, 0.52, f'Median: {median_survival_time}', ha='left', va='bottom', color='r', fontsize=10)
plt.show()
```


    
![png](output_140_0.png)
    



```python
##################################
# Computing the median survival time
##################################
median_survival_time = kmf.median_survival_time_
print(f'Median Survival Time: {median_survival_time}')
```

    Median Survival Time: 3358.0
    


```python
##################################
# Exploring the relationships between
# the numeric predictors and event status
##################################
cirrhosis_survival_numeric_predictors = ['Age', 'Bilirubin','Cholesterol', 'Albumin','Copper', 'Alk_Phos','SGOT', 'Tryglicerides','Platelets', 'Prothrombin']
plt.figure(figsize=(17, 12))
for i in range(1, 11):
    plt.subplot(2, 5, i)
    sns.boxplot(x='Status', y=cirrhosis_survival_numeric_predictors[i-1], data=cirrhosis_survival_train_EDA)
    plt.title(f'{cirrhosis_survival_numeric_predictors[i-1]} vs Event Status')
plt.tight_layout()
plt.show()
```


    
![png](output_142_0.png)
    



```python
##################################
# Exploring the relationships between
# the object predictors and event status
##################################
cirrhosis_survival_object_predictors = ['Drug', 'Sex','Ascites', 'Hepatomegaly','Spiders', 'Edema','Stage_1.0','Stage_2.0','Stage_3.0','Stage_4.0']
plt.figure(figsize=(17, 12))
for i in range(1, 11):
    plt.subplot(2, 5, i)
    sns.countplot(x=cirrhosis_survival_object_predictors[i-1], hue='Status', data=cirrhosis_survival_train_EDA)
    plt.title(f'{cirrhosis_survival_object_predictors[i-1]} vs Event Status')
    plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
```


    
![png](output_143_0.png)
    



```python
##################################
# Exploring the relationships between
# the numeric predictors and survival time
##################################
plt.figure(figsize=(17, 12))
for i in range(1, 11):
    plt.subplot(2, 5, i)
    sns.scatterplot(x='N_Days', y=cirrhosis_survival_numeric_predictors[i-1], data=cirrhosis_survival_train_EDA, hue='Status')
    loess_smoothed = lowess(cirrhosis_survival_train_EDA['N_Days'], cirrhosis_survival_train_EDA[cirrhosis_survival_numeric_predictors[i-1]], frac=0.3)
    plt.plot(loess_smoothed[:, 1], loess_smoothed[:, 0], color='red')
    plt.title(f'{cirrhosis_survival_numeric_predictors[i-1]} vs Survival Time')
    plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
```


    
![png](output_144_0.png)
    



```python
##################################
# Exploring the relationships between
# the object predictors and survival time
##################################
plt.figure(figsize=(17, 12))
for i in range(1, 11):
    plt.subplot(2, 5, i)
    sns.boxplot(x=cirrhosis_survival_object_predictors[i-1], y='N_Days', hue='Status', data=cirrhosis_survival_train_EDA)
    plt.title(f'{cirrhosis_survival_object_predictors[i-1]} vs Survival Time')
    plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
```


    
![png](output_145_0.png)
    


### 1.5.2 Hypothesis Testing <a class="anchor" id="1.5.2"></a>

1. The relationship between the numeric predictors to the <span style="color: #FF0000">Status</span> event variable was statistically evaluated using the following hypotheses:
    * **Null**: Difference in the means between groups True and False is equal to zero  
    * **Alternative**: Difference in the means between groups True and False is not equal to zero   
2. There is sufficient evidence to conclude of a statistically significant difference between the means of the numeric measurements obtained from the <span style="color: #FF0000">Status</span> groups in 10 numeric predictors given their high t-test statistic values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">Bilirubin</span>: T.Test.Statistic=-8.031, Correlation.PValue=0.000
    * <span style="color: #FF0000">Prothrombin</span>: T.Test.Statistic=-7.062, Correlation.PValue=0.000 
    * <span style="color: #FF0000">Copper</span>: T.Test.Statistic=-5.699, Correlation.PValue=0.000  
    * <span style="color: #FF0000">Alk_Phos</span>: T.Test.Statistic=-4.638, Correlation.PValue=0.000 
    * <span style="color: #FF0000">SGOT</span>: T.Test.Statistic=-4.207, Correlation.PValue=0.000 
    * <span style="color: #FF0000">Albumin</span>: T.Test.Statistic=+3.871, Correlation.PValue=0.000  
    * <span style="color: #FF0000">Tryglicerides</span>: T.Test.Statistic=-3.575, Correlation.PValue=0.000   
    * <span style="color: #FF0000">Age</span>: T.Test.Statistic=-3.264, Correlation.PValue=0.001
    * <span style="color: #FF0000">Platelets</span>: T.Test.Statistic=+3.261, Correlation.PValue=0.001
    * <span style="color: #FF0000">Cholesterol</span>: T.Test.Statistic=-2.256, Correlation.PValue=0.025
3. The relationship between the object predictors to the <span style="color: #FF0000">Status</span> event variable was statistically evaluated using the following hypotheses:
    * **Null**: The object predictor is independent of the event variable 
    * **Alternative**: The object predictor is dependent on the event variable   
4. There is sufficient evidence to conclude of a statistically significant relationship between the individual categories and the <span style="color: #FF0000">Status</span> groups in 8 object predictors given their high chisquare statistic values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">Ascites</span>: ChiSquare.Test.Statistic=16.854, ChiSquare.Test.PValue=0.000
    * <span style="color: #FF0000">Hepatomegaly</span>: ChiSquare.Test.Statistic=14.206, ChiSquare.Test.PValue=0.000   
    * <span style="color: #FF0000">Edema</span>: ChiSquare.Test.Statistic=12.962, ChiSquare.Test.PValue=0.001 
    * <span style="color: #FF0000">Stage_4.0</span>: ChiSquare.Test.Statistic=11.505, ChiSquare.Test.PValue=0.00
    * <span style="color: #FF0000">Sex</span>: ChiSquare.Test.Statistic=6.837, ChiSquare.Test.PValue=0.008
    * <span style="color: #FF0000">Stage_2.0</span>: ChiSquare.Test.Statistic=4.024, ChiSquare.Test.PValue=0.045   
    * <span style="color: #FF0000">Stage_1.0</span>: ChiSquare.Test.Statistic=3.978, ChiSquare.Test.PValue=0.046 
    * <span style="color: #FF0000">Spiders</span>: ChiSquare.Test.Statistic=3.953, ChiSquare.Test.PValue=0.047
5. The relationship between the object predictors to the <span style="color: #FF0000">Status</span> and <span style="color: #FF0000">N_Days</span> variables was statistically evaluated using the following hypotheses:
    * **Null**: There is no difference in survival probabilities among cases belonging to each category of the object predictor.
    * **Alternative**: There is a difference in survival probabilities among cases belonging to each category of the object predictor.
6. There is sufficient evidence to conclude of a statistically significant difference in survival probabilities between the individual categories and the <span style="color: #FF0000">Status</span> groups with respect to the survival duration <span style="color: #FF0000">N_Days</span> in 8 object predictors given their high log-rank test statistic values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">Ascites</span>: LR.Test.Statistic=37.792, LR.Test.PValue=0.000
    * <span style="color: #FF0000">Edema</span>: LR.Test.Statistic=31.619, LR.Test.PValue=0.000 
    * <span style="color: #FF0000">Stage_4.0</span>: LR.Test.Statistic=26.482, LR.Test.PValue=0.000
    * <span style="color: #FF0000">Hepatomegaly</span>: LR.Test.Statistic=20.350, LR.Test.PValue=0.000   
    * <span style="color: #FF0000">Spiders</span>: LR.Test.Statistic=10.762, LR.Test.PValue=0.001
    * <span style="color: #FF0000">Stage_2.0</span>: LR.Test.Statistic=6.775, LR.Test.PValue=0.009   
    * <span style="color: #FF0000">Sex</span>: LR.Test.Statistic=5.514, LR.Test.PValue=0.018
    * <span style="color: #FF0000">Stage_1.0</span>: LR.Test.Statistic=5.473, LR.Test.PValue=0.019 
7. The relationship between the binned numeric predictors to the <span style="color: #FF0000">Status</span> and <span style="color: #FF0000">N_Days</span> variables was statistically evaluated using the following hypotheses:
    * **Null**: There is no difference in survival probabilities among cases belonging to each category of the binned numeric predictor.
    * **Alternative**: There is a difference in survival probabilities among cases belonging to each category of the binned numeric predictor.
8. There is sufficient evidence to conclude of a statistically significant difference in survival probabilities between the individual categories and the <span style="color: #FF0000">Status</span> groups with respect to the survival duration <span style="color: #FF0000">N_Days</span> in 9 binned numeric predictors given their high log-rank test statistic values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">Binned_Bilirubin</span>: LR.Test.Statistic=62.559, LR.Test.PValue=0.000
    * <span style="color: #FF0000">Binned_Albumin</span>: LR.Test.Statistic=29.444, LR.Test.PValue=0.000 
    * <span style="color: #FF0000">Binned_Copper</span>: LR.Test.Statistic=27.452, LR.Test.PValue=0.000
    * <span style="color: #FF0000">Binned_Prothrombin</span>: LR.Test.Statistic=21.695, LR.Test.PValue=0.000   
    * <span style="color: #FF0000">Binned_SGOT</span>: LR.Test.Statistic=16.178, LR.Test.PValue=0.000
    * <span style="color: #FF0000">Binned_Tryglicerides</span>: LR.Test.Statistic=11.512, LR.Test.PValue=0.000   
    * <span style="color: #FF0000">Binned_Age</span>: LR.Test.Statistic=9.012, LR.Test.PValue=0.002
    * <span style="color: #FF0000">Binned_Platelets</span>: LR.Test.Statistic=6.741, LR.Test.PValue=0.009 
    * <span style="color: #FF0000">Binned_Alk_Phos</span>: LR.Test.Statistic=5.503, LR.Test.PValue=0.018 



```python
##################################
# Computing the t-test 
# statistic and p-values
# between the event variable
# and numeric predictor columns
##################################
cirrhosis_survival_numeric_ttest_event = {}
for numeric_column in cirrhosis_survival_numeric_predictors:
    group_0 = cirrhosis_survival_train_EDA[cirrhosis_survival_train_EDA.loc[:,'Status']==False]
    group_1 = cirrhosis_survival_train_EDA[cirrhosis_survival_train_EDA.loc[:,'Status']==True]
    cirrhosis_survival_numeric_ttest_event['Status_' + numeric_column] = stats.ttest_ind(
        group_0[numeric_column], 
        group_1[numeric_column], 
        equal_var=True)
```


```python
##################################
# Formulating the pairwise ttest summary
# between the event variable
# and numeric predictor columns
##################################
cirrhosis_survival_numeric_ttest_summary = cirrhosis_survival_train_EDA.from_dict(cirrhosis_survival_numeric_ttest_event, orient='index')
cirrhosis_survival_numeric_ttest_summary.columns = ['T.Test.Statistic', 'T.Test.PValue']
display(cirrhosis_survival_numeric_ttest_summary.sort_values(by=['T.Test.PValue'], ascending=True))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>T.Test.Statistic</th>
      <th>T.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Status_Bilirubin</th>
      <td>-8.030789</td>
      <td>6.198797e-14</td>
    </tr>
    <tr>
      <th>Status_Prothrombin</th>
      <td>-7.062933</td>
      <td>2.204961e-11</td>
    </tr>
    <tr>
      <th>Status_Copper</th>
      <td>-5.699409</td>
      <td>3.913575e-08</td>
    </tr>
    <tr>
      <th>Status_Alk_Phos</th>
      <td>-4.638524</td>
      <td>6.077981e-06</td>
    </tr>
    <tr>
      <th>Status_SGOT</th>
      <td>-4.207123</td>
      <td>3.791642e-05</td>
    </tr>
    <tr>
      <th>Status_Albumin</th>
      <td>3.871216</td>
      <td>1.434736e-04</td>
    </tr>
    <tr>
      <th>Status_Tryglicerides</th>
      <td>-3.575779</td>
      <td>4.308371e-04</td>
    </tr>
    <tr>
      <th>Status_Age</th>
      <td>-3.264563</td>
      <td>1.274679e-03</td>
    </tr>
    <tr>
      <th>Status_Platelets</th>
      <td>3.261042</td>
      <td>1.289850e-03</td>
    </tr>
    <tr>
      <th>Status_Cholesterol</th>
      <td>-2.256073</td>
      <td>2.506758e-02</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Computing the chisquare
# statistic and p-values
# between the event variable
# and categorical predictor columns
##################################
cirrhosis_survival_object_chisquare_event = {}
for object_column in cirrhosis_survival_object_predictors:
    contingency_table = pd.crosstab(cirrhosis_survival_train_EDA[object_column], 
                                    cirrhosis_survival_train_EDA['Status'])
    cirrhosis_survival_object_chisquare_event['Status_' + object_column] = stats.chi2_contingency(
        contingency_table)[0:2]
```


```python
##################################
# Formulating the pairwise chisquare summary
# between the event variable
# and categorical predictor columns
##################################
cirrhosis_survival_object_chisquare_event_summary = cirrhosis_survival_train_EDA.from_dict(cirrhosis_survival_object_chisquare_event, orient='index')
cirrhosis_survival_object_chisquare_event_summary.columns = ['ChiSquare.Test.Statistic', 'ChiSquare.Test.PValue']
display(cirrhosis_survival_object_chisquare_event_summary.sort_values(by=['ChiSquare.Test.PValue'], ascending=True))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ChiSquare.Test.Statistic</th>
      <th>ChiSquare.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Status_Ascites</th>
      <td>16.854134</td>
      <td>0.000040</td>
    </tr>
    <tr>
      <th>Status_Hepatomegaly</th>
      <td>14.206045</td>
      <td>0.000164</td>
    </tr>
    <tr>
      <th>Status_Edema</th>
      <td>12.962303</td>
      <td>0.000318</td>
    </tr>
    <tr>
      <th>Status_Stage_4.0</th>
      <td>11.505826</td>
      <td>0.000694</td>
    </tr>
    <tr>
      <th>Status_Sex</th>
      <td>6.837272</td>
      <td>0.008928</td>
    </tr>
    <tr>
      <th>Status_Stage_2.0</th>
      <td>4.024677</td>
      <td>0.044839</td>
    </tr>
    <tr>
      <th>Status_Stage_1.0</th>
      <td>3.977918</td>
      <td>0.046101</td>
    </tr>
    <tr>
      <th>Status_Spiders</th>
      <td>3.953826</td>
      <td>0.046765</td>
    </tr>
    <tr>
      <th>Status_Stage_3.0</th>
      <td>0.082109</td>
      <td>0.774459</td>
    </tr>
    <tr>
      <th>Status_Drug</th>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Exploring the relationships between
# the object predictors with
# survival event and duration
##################################
plt.figure(figsize=(17, 25))
for i in range(0, len(cirrhosis_survival_object_predictors)):
    ax = plt.subplot(5, 2, i+1)
    for group in [0,1]:
        kmf.fit(durations=cirrhosis_survival_train_EDA[cirrhosis_survival_train_EDA[cirrhosis_survival_object_predictors[i]] == group]['N_Days'],
                event_observed=cirrhosis_survival_train_EDA[cirrhosis_survival_train_EDA[cirrhosis_survival_object_predictors[i]] == group]['Status'], label=group)
        kmf.plot_survival_function(ax=ax)
    plt.title(f'Survival Probabilities by {cirrhosis_survival_object_predictors[i]} Categories')
    plt.xlabel('N_Days')
    plt.ylabel('Event Survival Probability')
plt.tight_layout()
plt.show()
```


    
![png](output_151_0.png)
    



```python
##################################
# Computing the log-rank test
# statistic and p-values
# between the event and duration variables
# with the object predictor columns
##################################
cirrhosis_survival_object_lrtest_event = {}
for object_column in cirrhosis_survival_object_predictors:
    groups = [0,1]
    group_0_event = cirrhosis_survival_train_EDA[cirrhosis_survival_train_EDA[object_column] == groups[0]]['Status']
    group_1_event = cirrhosis_survival_train_EDA[cirrhosis_survival_train_EDA[object_column] == groups[1]]['Status']
    group_0_duration = cirrhosis_survival_train_EDA[cirrhosis_survival_train_EDA[object_column] == groups[0]]['N_Days']
    group_1_duration = cirrhosis_survival_train_EDA[cirrhosis_survival_train_EDA[object_column] == groups[1]]['N_Days']
    lr_test = logrank_test(group_0_duration, group_1_duration,event_observed_A=group_0_event, event_observed_B=group_1_event)
    cirrhosis_survival_object_lrtest_event['Status_NDays_' + object_column] = (lr_test.test_statistic, lr_test.p_value)
```


```python
##################################
# Formulating the log-rank test summary
# between the event and duration variables
# with the object predictor columns
##################################
cirrhosis_survival_object_lrtest_summary = cirrhosis_survival_train_EDA.from_dict(cirrhosis_survival_object_lrtest_event, orient='index')
cirrhosis_survival_object_lrtest_summary.columns = ['LR.Test.Statistic', 'LR.Test.PValue']
display(cirrhosis_survival_object_lrtest_summary.sort_values(by=['LR.Test.PValue'], ascending=True))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LR.Test.Statistic</th>
      <th>LR.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Status_NDays_Ascites</th>
      <td>37.792220</td>
      <td>7.869499e-10</td>
    </tr>
    <tr>
      <th>Status_NDays_Edema</th>
      <td>31.619652</td>
      <td>1.875223e-08</td>
    </tr>
    <tr>
      <th>Status_NDays_Stage_4.0</th>
      <td>26.482676</td>
      <td>2.659121e-07</td>
    </tr>
    <tr>
      <th>Status_NDays_Hepatomegaly</th>
      <td>20.360210</td>
      <td>6.414988e-06</td>
    </tr>
    <tr>
      <th>Status_NDays_Spiders</th>
      <td>10.762275</td>
      <td>1.035900e-03</td>
    </tr>
    <tr>
      <th>Status_NDays_Stage_2.0</th>
      <td>6.775033</td>
      <td>9.244176e-03</td>
    </tr>
    <tr>
      <th>Status_NDays_Sex</th>
      <td>5.514094</td>
      <td>1.886385e-02</td>
    </tr>
    <tr>
      <th>Status_NDays_Stage_1.0</th>
      <td>5.473270</td>
      <td>1.930946e-02</td>
    </tr>
    <tr>
      <th>Status_NDays_Stage_3.0</th>
      <td>0.478031</td>
      <td>4.893156e-01</td>
    </tr>
    <tr>
      <th>Status_NDays_Drug</th>
      <td>0.000016</td>
      <td>9.968084e-01</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Creating an alternate copy of the 
# EDA data which will utilize
# binning for numeric predictors
##################################
cirrhosis_survival_train_EDA_binned = cirrhosis_survival_train_EDA.copy()

##################################
# Creating a function to bin
# numeric predictors into two groups
##################################
def bin_numeric_predictor(df, predictor):
    median = df[predictor].median()
    df[f'Binned_{predictor}'] = np.where(df[predictor] <= median, 0, 1)
    return df

##################################
# Binning the numeric predictors
# in the alternate EDA data into two groups
##################################
for numeric_column in cirrhosis_survival_numeric_predictors:
    cirrhosis_survival_train_EDA_binned = bin_numeric_predictor(cirrhosis_survival_train_EDA_binned, numeric_column)
    
##################################
# Formulating the binned numeric predictors
##################################    
cirrhosis_survival_binned_numeric_predictors = ["Binned_" + predictor for predictor in cirrhosis_survival_numeric_predictors]
```


```python
##################################
# Exploring the relationships between
# the binned numeric predictors with
# survival event and duration
##################################
plt.figure(figsize=(17, 25))
for i in range(0, len(cirrhosis_survival_binned_numeric_predictors)):
    ax = plt.subplot(5, 2, i+1)
    for group in [0,1]:
        kmf.fit(durations=cirrhosis_survival_train_EDA_binned[cirrhosis_survival_train_EDA_binned[cirrhosis_survival_binned_numeric_predictors[i]] == group]['N_Days'],
                event_observed=cirrhosis_survival_train_EDA_binned[cirrhosis_survival_train_EDA_binned[cirrhosis_survival_binned_numeric_predictors[i]] == group]['Status'], label=group)
        kmf.plot_survival_function(ax=ax)
    plt.title(f'Survival Probabilities by {cirrhosis_survival_binned_numeric_predictors[i]} Categories')
    plt.xlabel('N_Days')
    plt.ylabel('Event Survival Probability')
plt.tight_layout()
plt.show()
```


    
![png](output_155_0.png)
    



```python
##################################
# Computing the log-rank test
# statistic and p-values
# between the event and duration variables
# with the binned numeric predictor columns
##################################
cirrhosis_survival_binned_numeric_lrtest_event = {}
for binned_numeric_column in cirrhosis_survival_binned_numeric_predictors:
    groups = [0,1]
    group_0_event = cirrhosis_survival_train_EDA_binned[cirrhosis_survival_train_EDA_binned[binned_numeric_column] == groups[0]]['Status']
    group_1_event = cirrhosis_survival_train_EDA_binned[cirrhosis_survival_train_EDA_binned[binned_numeric_column] == groups[1]]['Status']
    group_0_duration = cirrhosis_survival_train_EDA_binned[cirrhosis_survival_train_EDA_binned[binned_numeric_column] == groups[0]]['N_Days']
    group_1_duration = cirrhosis_survival_train_EDA_binned[cirrhosis_survival_train_EDA_binned[binned_numeric_column] == groups[1]]['N_Days']
    lr_test = logrank_test(group_0_duration, group_1_duration,event_observed_A=group_0_event, event_observed_B=group_1_event)
    cirrhosis_survival_binned_numeric_lrtest_event['Status_NDays_' + binned_numeric_column] = (lr_test.test_statistic, lr_test.p_value)
```


```python
##################################
# Formulating the log-rank test summary
# between the event and duration variables
# with the binned numeric predictor columns
##################################
cirrhosis_survival_binned_numeric_lrtest_summary = cirrhosis_survival_train_EDA_binned.from_dict(cirrhosis_survival_binned_numeric_lrtest_event, orient='index')
cirrhosis_survival_binned_numeric_lrtest_summary.columns = ['LR.Test.Statistic', 'LR.Test.PValue']
display(cirrhosis_survival_binned_numeric_lrtest_summary.sort_values(by=['LR.Test.PValue'], ascending=True))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LR.Test.Statistic</th>
      <th>LR.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Status_NDays_Binned_Bilirubin</th>
      <td>62.559303</td>
      <td>2.585412e-15</td>
    </tr>
    <tr>
      <th>Status_NDays_Binned_Albumin</th>
      <td>29.444808</td>
      <td>5.753197e-08</td>
    </tr>
    <tr>
      <th>Status_NDays_Binned_Copper</th>
      <td>27.452421</td>
      <td>1.610072e-07</td>
    </tr>
    <tr>
      <th>Status_NDays_Binned_Prothrombin</th>
      <td>21.695995</td>
      <td>3.194575e-06</td>
    </tr>
    <tr>
      <th>Status_NDays_Binned_SGOT</th>
      <td>16.178483</td>
      <td>5.764520e-05</td>
    </tr>
    <tr>
      <th>Status_NDays_Binned_Tryglicerides</th>
      <td>11.512960</td>
      <td>6.911262e-04</td>
    </tr>
    <tr>
      <th>Status_NDays_Binned_Age</th>
      <td>9.011700</td>
      <td>2.682568e-03</td>
    </tr>
    <tr>
      <th>Status_NDays_Binned_Platelets</th>
      <td>6.741196</td>
      <td>9.421142e-03</td>
    </tr>
    <tr>
      <th>Status_NDays_Binned_Alk_Phos</th>
      <td>5.503850</td>
      <td>1.897465e-02</td>
    </tr>
    <tr>
      <th>Status_NDays_Binned_Cholesterol</th>
      <td>3.773953</td>
      <td>5.205647e-02</td>
    </tr>
  </tbody>
</table>
</div>


### 1.6.1 Premodelling Data Description <a class="anchor" id="1.6.1"></a>

1. To evaluate the feature selection capabilities of the candidate models, all predictors were accounted during the model development process using the training subset:
    * **218 rows** (observations)
    * **22 columns** (variables)
        * **2/22 event | duration** (boolean | numeric)
             * <span style="color: #FF0000">Status</span>
             * <span style="color: #FF0000">N_Days</span>
        * **10/22 predictor** (numeric)
             * <span style="color: #FF0000">Age</span>
             * <span style="color: #FF0000">Bilirubin</span>
             * <span style="color: #FF0000">Cholesterol</span>
             * <span style="color: #FF0000">Albumin</span>
             * <span style="color: #FF0000">Copper</span>
             * <span style="color: #FF0000">Alk_Phos</span>
             * <span style="color: #FF0000">SGOT</span>
             * <span style="color: #FF0000">Triglycerides</span>
             * <span style="color: #FF0000">Platelets</span>
             * <span style="color: #FF0000">Prothrombin</span>
        * **10/21 predictor** (object)
             * <span style="color: #FF0000">Drug</span>
             * <span style="color: #FF0000">Sex</span>
             * <span style="color: #FF0000">Ascites</span>
             * <span style="color: #FF0000">Hepatomegaly</span>
             * <span style="color: #FF0000">Spiders</span>
             * <span style="color: #FF0000">Edema</span>
             * <span style="color: #FF0000">Stage_1.0</span>
             * <span style="color: #FF0000">Stage_2.0</span>
             * <span style="color: #FF0000">Stage_3.0</span>
             * <span style="color: #FF0000">Stage_4.0</span>
2. Similarly, all predictors were accounted during the model evaluation process using the testing subset:
    * **94 rows** (observations)
    * **22 columns** (variables)
        * **2/22 event | duration** (boolean | numeric)
             * <span style="color: #FF0000">Status</span>
             * <span style="color: #FF0000">N_Days</span>
        * **10/22 predictor** (numeric)
             * <span style="color: #FF0000">Age</span>
             * <span style="color: #FF0000">Bilirubin</span>
             * <span style="color: #FF0000">Cholesterol</span>
             * <span style="color: #FF0000">Albumin</span>
             * <span style="color: #FF0000">Copper</span>
             * <span style="color: #FF0000">Alk_Phos</span>
             * <span style="color: #FF0000">SGOT</span>
             * <span style="color: #FF0000">Triglycerides</span>
             * <span style="color: #FF0000">Platelets</span>
             * <span style="color: #FF0000">Prothrombin</span>
        * **10/21 predictor** (object)
             * <span style="color: #FF0000">Drug</span>
             * <span style="color: #FF0000">Sex</span>
             * <span style="color: #FF0000">Ascites</span>
             * <span style="color: #FF0000">Hepatomegaly</span>
             * <span style="color: #FF0000">Spiders</span>
             * <span style="color: #FF0000">Edema</span>
             * <span style="color: #FF0000">Stage_1.0</span>
             * <span style="color: #FF0000">Stage_2.0</span>
             * <span style="color: #FF0000">Stage_3.0</span>
             * <span style="color: #FF0000">Stage_4.0</span>


```python
##################################
# Converting the event and duration variables
# for the training set
# to array as preparation for modeling
##################################
cirrhosis_survival_y_train_array = np.array([(row.Status, row.N_Days) for index, row in cirrhosis_survival_y_train_cleaned.iterrows()], dtype=[('Status', 'bool'), ('N_Days', 'int')])
print(cirrhosis_survival_y_train_array)
```

    [(False, 2475) (False,  877) (False, 3050) ( True,  110) ( True, 3839)
     (False, 2241) (False, 2332) (False, 1666) ( True, 2847) (False, 4500)
     (False, 4256) ( True, 1427) ( True,  943) (False, 2456) (False, 1301)
     (False, 3297) ( True, 1434) ( True, 1297) (False,  839) (False, 2995)
     ( True, 1235) (False,  901) ( True,  264) (False, 1614) ( True, 1360)
     (False, 2835) (False, 3445) ( True,  597) (False, 1250) ( True, 4079)
     ( True, 2055) ( True,  850) ( True, 2105) ( True, 3358) (False, 3707)
     (False, 4032) (False, 2657) (False, 1592) ( True,  400) ( True, 1077)
     (False, 3099) (False, 1951) (False, 2294) (False, 4453) (False, 1978)
     ( True, 2297) ( True,  890) (False, 1979) (False, 1149) (False, 1765)
     ( True, 2689) ( True,  326) (False, 3823) ( True,  191) (False, 4523)
     ( True,  930) (False, 1656) (False, 3149) (False, 1230) ( True, 1012)
     (False, 1831) ( True, 1487) (False, 2563) (False, 1067) ( True, 1741)
     ( True, 2796) ( True, 2386) ( True, 2256) ( True,   77) (False, 3255)
     (False, 3933) (False, 2178) ( True, 1657) (False, 2221) (False, 1677)
     ( True, 1444) ( True, 1786) (False, 1434) (False, 4184) ( True,  321)
     ( True,   71) ( True, 1191) ( True,  786) (False, 1568) ( True, 1037)
     (False, 1769) (False, 2170) (False, 3992) ( True, 1170) (False, 2443)
     (False, 2573) (False, 1615) (False, 1810) ( True, 1000) ( True,  611)
     (False, 1320) ( True, 1217) (False, 2171) ( True, 1152) (False, 1363)
     ( True, 1536) ( True,  797) (False, 1401) (False,  732) (False, 1433)
     (False, 1216) ( True, 2583) (False, 1569) ( True, 3428) ( True, 2466)
     ( True, 3090) (False, 2644) (False, 4365) ( True,  304) (False, 2870)
     (False, 3913) ( True,  552) (False, 1874) (False, 1271) (False,  533)
     (False, 3239) ( True, 1847) (False, 1412) (False, 2195) ( True, 3086)
     (False, 2357) (False, 2713) ( True, 2598) ( True,  694) (False, 1084)
     (False, 2106) (False, 3611) (False, 2555) (False, 3069) ( True,  799)
     ( True,  186) ( True,  769) (False, 3581) ( True, 2503) ( True,  859)
     (False, 1525) (False, 1295) ( True,  999) ( True, 1212) (False, 2350)
     ( True,  980) (False, 2468) (False, 1153) (False, 4196) ( True, 1191)
     (False, 4427) ( True,  348) (False, 2624) (False, 4039) ( True, 2288)
     ( True,  207) ( True,  549) (False, 2504) (False, 3820) ( True, 1356)
     ( True, 3853) (False, 4509) (False, 2216) (False, 1558) ( True, 1576)
     ( True, 2224) (False, 4190) (False, 3059) (False, 2797) (False, 2168)
     (False, 2609) (False, 3150) (False, 2976) (False, 1216) (False, 3672)
     (False, 2157) (False, 1293) ( True,  790) (False, 2891) (False, 1234)
     (False, 2318) (False, 1542) (False, 1790) (False,  939) (False, 2301)
     ( True, 2081) ( True, 2090) (False, 4050) (False, 4127) (False, 2692)
     (False, 1302) ( True, 1080) (False, 1908) ( True, 3222) (False, 1770)
     (False, 2272) (False, 1832) (False, 4025) ( True,   41) ( True,  515)
     ( True,  708) ( True, 1690) (False, 1967) (False, 1945) (False, 1504)
     ( True, 1413) (False, 1702) (False, 3422) (False, 2666) (False, 3092)
     ( True, 4191) (False, 1363) (False, 1932) ( True, 3170) (False, 4232)
     ( True,  460) (False, 1614) ( True,  853)]
    


```python
##################################
# Verifying the predictor variables
# for the training set
# as preparation for modeling
##################################
display(cirrhosis_survival_X_train_preprocessed)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_1.0</th>
      <th>Stage_2.0</th>
      <th>Stage_3.0</th>
      <th>Stage_4.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.342097</td>
      <td>0.863802</td>
      <td>0.886087</td>
      <td>-0.451884</td>
      <td>-0.972098</td>
      <td>0.140990</td>
      <td>0.104609</td>
      <td>0.155130</td>
      <td>0.540960</td>
      <td>0.747580</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.470901</td>
      <td>0.516350</td>
      <td>1.554523</td>
      <td>0.827618</td>
      <td>0.467579</td>
      <td>-0.705337</td>
      <td>0.301441</td>
      <td>1.275222</td>
      <td>0.474140</td>
      <td>-0.315794</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.239814</td>
      <td>-0.625875</td>
      <td>0.293280</td>
      <td>0.646582</td>
      <td>-0.241205</td>
      <td>-0.848544</td>
      <td>0.275723</td>
      <td>-1.684460</td>
      <td>0.756741</td>
      <td>0.087130</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.052733</td>
      <td>0.559437</td>
      <td>-1.534283</td>
      <td>0.354473</td>
      <td>-0.284113</td>
      <td>-0.014525</td>
      <td>0.162878</td>
      <td>-0.189139</td>
      <td>-1.735375</td>
      <td>0.649171</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.795010</td>
      <td>1.142068</td>
      <td>-0.108933</td>
      <td>-0.272913</td>
      <td>0.618030</td>
      <td>2.071847</td>
      <td>1.434674</td>
      <td>-0.212684</td>
      <td>-0.675951</td>
      <td>-0.315794</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>213</th>
      <td>0.167351</td>
      <td>-0.764558</td>
      <td>-1.147913</td>
      <td>-0.887287</td>
      <td>0.178721</td>
      <td>0.320864</td>
      <td>1.574157</td>
      <td>0.053603</td>
      <td>-0.848130</td>
      <td>1.402075</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>214</th>
      <td>0.004420</td>
      <td>-0.918484</td>
      <td>-0.782126</td>
      <td>0.046794</td>
      <td>-0.742780</td>
      <td>0.549222</td>
      <td>-0.379344</td>
      <td>0.251836</td>
      <td>-0.519594</td>
      <td>0.546417</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>215</th>
      <td>-0.381113</td>
      <td>1.190111</td>
      <td>0.136728</td>
      <td>-0.194525</td>
      <td>0.569475</td>
      <td>0.881231</td>
      <td>1.871385</td>
      <td>-1.684460</td>
      <td>1.587388</td>
      <td>1.331561</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>216</th>
      <td>0.800410</td>
      <td>-1.283677</td>
      <td>-0.262095</td>
      <td>2.149157</td>
      <td>-0.836372</td>
      <td>-2.600746</td>
      <td>-1.414105</td>
      <td>0.645045</td>
      <td>-0.324107</td>
      <td>0.087130</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>217</th>
      <td>0.900345</td>
      <td>1.951372</td>
      <td>0.375927</td>
      <td>-0.061605</td>
      <td>1.546419</td>
      <td>0.884998</td>
      <td>1.376469</td>
      <td>1.394391</td>
      <td>-1.196794</td>
      <td>1.018851</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>218 rows × 20 columns</p>
</div>



```python
##################################
# Converting the event and duration variables
# for the test set
# to array as preparation for modeling
##################################
cirrhosis_survival_y_test_array = np.array([(row.Status, row.N_Days) for index, row in cirrhosis_survival_y_test_cleaned.iterrows()], dtype=[('Status', 'bool'), ('N_Days', 'int')])
print(cirrhosis_survival_y_test_array)
```

    [(False, 3336) (False, 1321) (False, 1435) (False, 4459) (False, 2721)
     (False, 2022) (False, 2527) ( True, 2400) (False, 3388) (False, 2944)
     ( True, 1827) (False, 3098) (False, 1418) ( True,  216) (False, 2176)
     ( True, 1690) ( True, 3445) (False, 3850) (False, 2449) (False,  788)
     (False, 1447) ( True,   51) ( True, 3574) ( True,  388) ( True, 1350)
     ( True,  762) (False, 2365) (False,  994) ( True,  131) (False, 3458)
     (False, 2574) ( True,  750) (False, 2224) ( True, 3395) (False, 1349)
     (False, 1882) ( True,  974) ( True, 1165) ( True,  971) (False, 4556)
     ( True, 3762) (False, 2863) (False, 1481) (False, 2615) (False, 2772)
     (False, 1300) ( True, 2769) (False, 1776) (False, 2255) ( True, 3282)
     (False,  837) (False, 1783) (False, 1030) (False, 2990) (False, 2580)
     ( True,  334) ( True,  198) ( True, 1492) ( True, 1925) ( True,  673)
     (False, 2556) (False, 1785) (False, 2050) ( True, 1682) (False, 2033)
     (False, 3577) (False, 1408) ( True, 3584) ( True,  264) ( True,  824)
     (False, 1455) ( True,  223) ( True, 2540) (False, 1882) ( True, 1083)
     (False, 1329) ( True,  904) (False, 1457) (False, 1701) ( True,  179)
     ( True,  140) (False, 2452) (False, 1420) ( True,  130) ( True,  733)
     (False, 1735) (False, 2330) ( True, 2419) (False, 4467) (False, 2363)
     (False, 2576) (False,  737) (False, 2504) ( True, 3244)]
    


```python
##################################
# Verifying the predictor variables
# for the test set
# as preparation for modeling
##################################
display(cirrhosis_survival_X_test_preprocessed)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_1.0</th>
      <th>Stage_2.0</th>
      <th>Stage_3.0</th>
      <th>Stage_4.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.043704</td>
      <td>0.744396</td>
      <td>0.922380</td>
      <td>0.240951</td>
      <td>0.045748</td>
      <td>0.317282</td>
      <td>-0.078335</td>
      <td>2.671950</td>
      <td>1.654815</td>
      <td>-0.948196</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.936476</td>
      <td>-0.764558</td>
      <td>0.160096</td>
      <td>-0.600950</td>
      <td>-0.179138</td>
      <td>-0.245613</td>
      <td>0.472422</td>
      <td>-0.359800</td>
      <td>0.348533</td>
      <td>0.439089</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.749033</td>
      <td>0.371523</td>
      <td>0.558115</td>
      <td>0.646582</td>
      <td>-0.159024</td>
      <td>0.339454</td>
      <td>0.685117</td>
      <td>-3.109146</td>
      <td>-0.790172</td>
      <td>-0.617113</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.485150</td>
      <td>-0.918484</td>
      <td>-0.690904</td>
      <td>1.629765</td>
      <td>0.028262</td>
      <td>1.713791</td>
      <td>-1.387751</td>
      <td>0.155130</td>
      <td>0.679704</td>
      <td>0.087130</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.815655</td>
      <td>1.286438</td>
      <td>2.610501</td>
      <td>-0.722153</td>
      <td>0.210203</td>
      <td>0.602860</td>
      <td>3.494936</td>
      <td>-0.053214</td>
      <td>-0.475634</td>
      <td>-1.714435</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>89</th>
      <td>0.692406</td>
      <td>-0.096645</td>
      <td>-0.906164</td>
      <td>-0.477005</td>
      <td>-1.930422</td>
      <td>-0.809457</td>
      <td>-0.888634</td>
      <td>-1.421640</td>
      <td>-1.638792</td>
      <td>1.101933</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>90</th>
      <td>-0.201495</td>
      <td>-1.283677</td>
      <td>0.064451</td>
      <td>0.297476</td>
      <td>-0.062405</td>
      <td>0.425745</td>
      <td>1.204015</td>
      <td>-1.077370</td>
      <td>0.939991</td>
      <td>-1.125995</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>91</th>
      <td>-0.974200</td>
      <td>0.776293</td>
      <td>-0.891985</td>
      <td>0.587203</td>
      <td>0.699548</td>
      <td>-0.199230</td>
      <td>-0.016923</td>
      <td>-0.463921</td>
      <td>0.060683</td>
      <td>-0.778722</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>92</th>
      <td>0.466763</td>
      <td>0.470819</td>
      <td>0.536326</td>
      <td>1.139126</td>
      <td>-1.293580</td>
      <td>0.511347</td>
      <td>0.410413</td>
      <td>0.059267</td>
      <td>0.672973</td>
      <td>-0.462938</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>93</th>
      <td>-1.327100</td>
      <td>0.835905</td>
      <td>0.534335</td>
      <td>-0.034678</td>
      <td>0.467579</td>
      <td>-0.064029</td>
      <td>0.488117</td>
      <td>-0.573417</td>
      <td>-0.249636</td>
      <td>0.546417</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>94 rows × 20 columns</p>
</div>


### 1.6.1 Premodelling Data Description <a class="anchor" id="1.6.1"></a>


### 1.6.2 Cox Regression with No Penalty <a class="anchor" id="1.6.2"></a>


### 1.6.3 Cox Regression With Full L1 Penalty <a class="anchor" id="1.6.3"></a>


### 1.6.4 Cox Regression With Full L2 Penalty <a class="anchor" id="1.6.4"></a>


### 1.6.5 Cox Regression With Equal L1|L2 Penalty <a class="anchor" id="1.6.5"></a>


### 1.6.6 Cox Regression With Predominantly L1-Weighted|L2 Penalty <a class="anchor" id="1.6.6"></a>


### 1.6.7 Cox Regression With Predominantly L2-Weighted|L1 Penalty <a class="anchor" id="1.6.7"></a>


## 1.7. Consolidated Findings <a class="anchor" id="1.7"></a>



# 2. Summary <a class="anchor" id="Summary"></a>

# 3. References <a class="anchor" id="References"></a>

* **[Book]** [Clinical Prediction Models](http://clinicalpredictionmodels.org/) by Ewout Steyerberg
* **[Book]** [Survival Analysis: A Self-Learning Text](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) by David Kleinbaum and Mitchel Klein
* **[Book]** [Applied Survival Analysis Using R](https://link.springer.com/book/10.1007/978-3-319-31245-3/) by Dirk Moore
* **[Python Library API]** [SciKit-Survival](https://pypi.org/project/scikit-survival/) by SciKit-Survival Team
* **[Python Library API]** [SciKit-Learn](https://scikit-learn.org/stable/index.html) by SciKit-Learn Team
* **[Python Library API]** [StatsModels](https://www.statsmodels.org/stable/index.html) by StatsModels Team
* **[Python Library API]** [SciPy](https://scipy.org/) by SciPy Team
* **[Python Library API]** [Lifelines](https://lifelines.readthedocs.io/en/latest/) by Lifelines Team
* **[Article]** [Exploring Time-to-Event with Survival Analysis](https://towardsdatascience.com/exploring-time-to-event-with-survival-analysis-8b0a7a33a7be) by Olivia Tanuwidjaja (Towards Data Science)
* **[Article]** [The Complete Introduction to Survival Analysis in Python](https://towardsdatascience.com/the-complete-introduction-to-survival-analysis-in-python-7523e17737e6) by Marco Peixeiro (Towards Data Science)
* **[Article]** [Survival Analysis Simplified: Explaining and Applying with Python](https://medium.com/@zynp.atlii/survival-analysis-simplified-explaining-and-applying-with-python-7efacf86ba32) by Zeynep Atli (Towards Data Science)
* **[Article]** [Survival Analysis in Python (KM Estimate, Cox-PH and AFT Model)](https://medium.com/the-researchers-guide/survival-analysis-in-python-km-estimate-cox-ph-and-aft-model-5533843c5d5d) by Rahul Raoniar (Medium)
* **[Article]** [How to Evaluate Survival Analysis Models)](https://towardsdatascience.com/how-to-evaluate-survival-analysis-models-dd67bc10caae) by Nicolo Cosimo Albanese (Towards Data Science)
* **[Article]** [Survival Analysis with Python Tutorial — How, What, When, and Why)](https://pub.towardsai.net/survival-analysis-with-python-tutorial-how-what-when-and-why-19a5cfb3c312) by Towards AI Team (Medium)
* **[Article]** [Survival Analysis: Predict Time-To-Event With Machine Learning)](https://towardsdatascience.com/survival-analysis-predict-time-to-event-with-machine-learning-part-i-ba52f9ab9a46) by Lina Faik (Medium)
* **[Article]** [A Complete Guide To Survival Analysis In Python, Part 1](https://www.kdnuggets.com/2020/07/complete-guide-survival-analysis-python-part1.html) by Pratik Shukla (KDNuggets)
* **[Article]** [A Complete Guide To Survival Analysis In Python, Part 2](https://www.kdnuggets.com/2020/07/guide-survival-analysis-python-part-2.html) by Pratik Shukla (KDNuggets)
* **[Article]** [A Complete Guide To Survival Analysis In Python, Part 3](https://www.kdnuggets.com/2020/07/guide-survival-analysis-python-part-3.html) by Pratik Shukla (KDNuggets)
* **[Article]** [Model Explainability using SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations)](https://medium.com/@anshulgoel991/model-exploitability-using-shap-shapley-additive-explanations-and-lime-local-interpretable-cb4f5594fc1a) by Anshul Goel (Medium)
* **[Article]** [A Comprehensive Guide into SHAP (SHapley Additive exPlanations) Values](https://www.kdnuggets.com/2020/07/guide-survival-analysis-python-part-3.html) by Brain John Aboze (DeepChecks.Com)
* **[Article]** [SHAP - Understanding How This Method for Explainable AI Works](https://safjan.com/how-the-shap-method-for-explainable-ai-works/#google_vignette) by Krystian Safjan (Safjan.Com)
* **[Article]** [SHAP: Shapley Additive Explanations](https://towardsdatascience.com/shap-shapley-additive-explanations-5a2a271ed9c3) by Fernando Lopez (Medium)
* **[Article]** [Explainable Machine Learning, Game Theory, and Shapley Values: A technical review](https://www.kdnuggets.com/2020/07/guide-survival-analysis-python-part-3.html) by Soufiane Fadel (Statistics Canada)
* **[Article]** [SHAP Values Explained Exactly How You Wished Someone Explained to You](https://towardsdatascience.com/shap-explained-the-way-i-wish-someone-explained-it-to-me-ab81cc69ef30) by Samuele Mazzanti (Towards Data Science)
* **[Article]** [Explaining Machine Learning Models: A Non-Technical Guide to Interpreting SHAP Analyses](https://www.aidancooper.co.uk/a-non-technical-guide-to-interpreting-shap-analyses/) by Aidan Cooper (AidanCooper.Co.UK)
* **[Article]** [Shapley Additive Explanations: Unveiling the Black Box of Machine Learning](https://python.plainenglish.io/shapley-additive-explanations-unveiling-the-black-box-of-machine-learning-477ba01ffa07) by Evertone Gomede (Medium)
* **[Article]** [SHAP (SHapley Additive exPlanations)](https://www.nerd-data.com/shap/) by Narut Soontranon (Nerd-Data.Com)
* **[Kaggle Project]** [Survival Analysis with Cox Model Implementation](https://www.kaggle.com/code/bryanb/survival-analysis-with-cox-model-implementation/notebook) by Bryan Boulé (Kaggle)
* **[Kaggle Project]** [Survival Analysis](https://www.kaggle.com/code/gunesevitan/survival-analysis/notebook) by Gunes Evitan (Kaggle)
* **[Kaggle Project]** [Survival Analysis of Lung Cancer Patients](https://www.kaggle.com/code/bryanb/survival-analysis-with-cox-model-implementation/notebook) by Sayan Chakraborty (Kaggle)
* **[Kaggle Project]** [COVID-19 Cox Survival Regression](https://www.kaggle.com/code/bryanb/survival-analysis-with-cox-model-implementation/notebook) by Ilias Katsabalos (Kaggle)
* **[Kaggle Project]** [Liver Cirrhosis Prediction with XGboost & EDA](https://www.kaggle.com/code/arjunbhaybhang/liver-cirrhosis-prediction-with-xgboost-eda) by Arjun Bhaybang (Kaggle)
* **[Kaggle Project]** [Survival Models VS ML Models Benchmark - Churn Tel](https://www.kaggle.com/code/caralosal/survival-models-vs-ml-models-benchmark-churn-tel) by Carlos Alonso Salcedo (Kaggle)
* **[Publication]** [Regression Models and Life Tables](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1972.tb00899.x) by David Cox (Royal Statistical Society)
* **[Publication]** [Covariance Analysis of Censored Survival Data](https://pubmed.ncbi.nlm.nih.gov/4813387/) by Norman Breslow (Biometrics)
* **[Publication]** [The Efficiency of Cox’s Likelihood Function for Censored Data](https://www.jstor.org/stable/2286217) by Bradley Efron (Journal of the American Statistical Association)
* **[Publication]** [Regularization Paths for Cox’s Proportional Hazards Model via Coordinate Descent](https://doi.org/10.18637/jss.v039.i05) by Noah Simon, Jerome Friedman, Trevor Hastie and Rob Tibshirani (Journal of Statistical Software)
* **[Publication]** [Shapley Additive Explanations](https://dl.acm.org/doi/10.5555/1756006.1756007) by Noah Simon, Jerome Friedman, Trevor Hastie and Rob Tibshirani (Journal of Statistical Software) by Erik Strumbelj and Igor Kononenko (The Journal of Machine Learning Research)
* **[Publication]** [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) by Scott Lundberg and Sun-In Lee (Conference on Neural Information Processing Systems)
* **[Publication]** [Survival Analysis Part I: Basic Concepts and First Analyses](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2394262/) by Taane Clark (British Journal of Cancer)
* **[Publication]** [Survival Analysis Part II: Multivariate Data Analysis – An Introduction to Concepts and Methods](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2394368/) by Mike Bradburn (British Journal of Cancer)
* **[Publication]** [Survival Analysis Part III: Multivariate Data Analysis – Choosing a Model and Assessing its Adequacy and Fit](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2376927/) by Mike Bradburn (British Journal of Cancer)
* **[Publication]** [Survival Analysis Part IV: Further Concepts and Methods in Survival Analysis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2394469/) by Taane Clark (British Journal of Cancer)
* **[Course]** [Survival Analysis in Python](https://app.datacamp.com/learn/courses/survival-analysis-in-python) by Shae Wang (DataCamp)


```python
from IPython.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>"))
```


<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>

