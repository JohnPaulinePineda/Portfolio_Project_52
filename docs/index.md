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
        * [1.6.3 Cox Regression With Full L1 Penalty](#1.6.3)
        * [1.6.4 Cox Regression With Full L2 Penalty](#1.6.4)
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

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
import shap

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
# Formulating a complete dataframe
# from the training subset for modelling
##################################
cirrhosis_survival_y_train_cleaned.reset_index(drop=True, inplace=True)
cirrhosis_survival_train_modeling = pd.concat([cirrhosis_survival_y_train_cleaned,
                                               cirrhosis_survival_X_train_preprocessed],
                                              axis=1)
cirrhosis_survival_train_modeling.drop(columns=['Stage_1.0', 'Stage_2.0', 'Stage_3.0'], axis=1, inplace=True)
cirrhosis_survival_train_modeling.head()
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
      <th>Platelets</th>
      <th>Prothrombin</th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
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
      <td>0.540960</td>
      <td>0.747580</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0.474140</td>
      <td>-0.315794</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0.756741</td>
      <td>0.087130</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>-1.735375</td>
      <td>0.649171</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
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
      <td>-0.675951</td>
      <td>-0.315794</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Formulating a complete dataframe
# from the testing subset for modelling
##################################
cirrhosis_survival_y_test_cleaned.reset_index(drop=True, inplace=True)
cirrhosis_survival_test_modeling = pd.concat([cirrhosis_survival_y_test_cleaned,
                                               cirrhosis_survival_X_test_preprocessed],
                                              axis=1)
cirrhosis_survival_test_modeling.drop(columns=['Stage_1.0', 'Stage_2.0', 'Stage_3.0'], axis=1, inplace=True)
cirrhosis_survival_test_modeling.head()
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
      <th>Platelets</th>
      <th>Prothrombin</th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_4.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>3336</td>
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
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>1321</td>
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
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>1435</td>
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
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>4459</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>2721</td>
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
    </tr>
  </tbody>
</table>
</div>



### 1.6.2 Cox Regression with No Penalty <a class="anchor" id="1.6.2"></a>

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Survival Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) refer to statistical methods used to analyze survival data, accounting for censored observations. These models aim to describe the relationship between survival time and one or more predictor variables, and to estimate the survival function and hazard function. Survival models are essential for understanding the factors that influence time-to-event data, allowing for predictions and comparisons between different groups or treatment effects. They are widely used in clinical trials, reliability engineering, and other areas where time-to-event data is prevalent.

[Cox Proportional Hazards Regression](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1972.tb00899.x) is a semiparametric model used to study the relationship between the survival time of subjects and one or more predictor variables. The model assumes that the hazard ratio (the risk of the event occurring at a specific time) is a product of a baseline hazard function and an exponential function of the predictor variables. It also does not require the baseline hazard to be specified, thus making it a semiparametric model. As a method, it is well-established and widely used in survival analysis, can handle time-dependent covariates and provides a relatively straightforward interpretation. However, the process assumes proportional hazards, which may not hold in all datasets, and may be less flexible in capturing complex relationships between variables and survival times compared to some machine learning models. Given a dataset with survival times, event indicators, and predictor variables, the algorithm involves defining the partial likelihood function for the Cox model (which only considers the relative ordering of survival times); using optimization techniques to estimate the regression coefficients by maximizing the log-partial likelihood; estimating the baseline hazard function (although it is not explicitly required for predictions); and calculating the hazard function and survival function for new data using the estimated coefficients and baseline hazard.

[No Penalty](https://lifelines.readthedocs.io/en/latest/), or zero regularization in cox regression, applies no additional constraints to the coefficients. The model estimates the coefficients by maximizing the partial likelihood function without any regularization term. This approach can lead to overfitting, especially when the number of predictors is large or when there is multicollinearity among the predictors.

1. The [cox proportional hazards regression model](https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html) from the <mark style="background-color: #CCECFF"><b>lifelines.CoxPHFitter</b></mark> Python library API was implemented. 
2. The model implementation used 2 hyperparameters:
    * <span style="color: #FF0000">penalizer</span> = penalty to the size of the coefficients during regression fixed at a value = 0.00
    * <span style="color: #FF0000">l1_ratio</span> = proportion of the L1 versus L2 penalty fixed at a value = 0.00
3. All 17 variables were used for prediction given the non-zero values of the model coefficients.
4. Out of all 17 predictors, only 3 variables were statistically significant:
    * <span style="color: #FF0000">Age</span>
    * <span style="color: #FF0000">Bilirubin</span>
    * <span style="color: #FF0000">Prothrombin</span>
5. The cross-validated model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8024
6. The apparent model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8478
7. The independent test model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8480
8. Considerable difference in the apparent and cross-validated model performance observed, indicative of the presence of moderate model overfitting.
9. Survival probability curves obtained from the groups generated by dichotomizing the risk scores demonstrated sufficient differentiation across the entire duration.
10. Hazard and survival probability estimations for 5 sampled cases demonstrated reasonably smooth profiles.


```python
##################################
# Formulating the Cox Regression model
# with No Penalty
# and generating the summary
##################################
cirrhosis_survival_coxph_L1_0_L2_0 = CoxPHFitter(penalizer=0.00, l1_ratio=0.00)
cirrhosis_survival_coxph_L1_0_L2_0.fit(cirrhosis_survival_train_modeling, duration_col='N_Days', event_col='Status')
cirrhosis_survival_coxph_L1_0_L2_0.summary
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
      <th>coef</th>
      <th>exp(coef)</th>
      <th>se(coef)</th>
      <th>coef lower 95%</th>
      <th>coef upper 95%</th>
      <th>exp(coef) lower 95%</th>
      <th>exp(coef) upper 95%</th>
      <th>cmp to</th>
      <th>z</th>
      <th>p</th>
      <th>-log2(p)</th>
    </tr>
    <tr>
      <th>covariate</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>0.376767</td>
      <td>1.457565</td>
      <td>0.138099</td>
      <td>0.106098</td>
      <td>0.647436</td>
      <td>1.111931</td>
      <td>1.910636</td>
      <td>0.0</td>
      <td>2.728240</td>
      <td>0.006367</td>
      <td>7.295096</td>
    </tr>
    <tr>
      <th>Bilirubin</th>
      <td>0.762501</td>
      <td>2.143630</td>
      <td>0.217243</td>
      <td>0.336713</td>
      <td>1.188289</td>
      <td>1.400337</td>
      <td>3.281461</td>
      <td>0.0</td>
      <td>3.509901</td>
      <td>0.000448</td>
      <td>11.123334</td>
    </tr>
    <tr>
      <th>Cholesterol</th>
      <td>0.061307</td>
      <td>1.063225</td>
      <td>0.163017</td>
      <td>-0.258200</td>
      <td>0.380814</td>
      <td>0.772441</td>
      <td>1.463476</td>
      <td>0.0</td>
      <td>0.376078</td>
      <td>0.706859</td>
      <td>0.500505</td>
    </tr>
    <tr>
      <th>Albumin</th>
      <td>-0.121698</td>
      <td>0.885416</td>
      <td>0.150453</td>
      <td>-0.416580</td>
      <td>0.173184</td>
      <td>0.659298</td>
      <td>1.189085</td>
      <td>0.0</td>
      <td>-0.808879</td>
      <td>0.418585</td>
      <td>1.256408</td>
    </tr>
    <tr>
      <th>Copper</th>
      <td>0.106671</td>
      <td>1.112568</td>
      <td>0.166238</td>
      <td>-0.219149</td>
      <td>0.432492</td>
      <td>0.803202</td>
      <td>1.541093</td>
      <td>0.0</td>
      <td>0.641678</td>
      <td>0.521083</td>
      <td>0.940416</td>
    </tr>
    <tr>
      <th>Alk_Phos</th>
      <td>0.009525</td>
      <td>1.009570</td>
      <td>0.149078</td>
      <td>-0.282664</td>
      <td>0.301713</td>
      <td>0.753773</td>
      <td>1.352173</td>
      <td>0.0</td>
      <td>0.063890</td>
      <td>0.949058</td>
      <td>0.075432</td>
    </tr>
    <tr>
      <th>SGOT</th>
      <td>0.175013</td>
      <td>1.191261</td>
      <td>0.155532</td>
      <td>-0.129825</td>
      <td>0.479850</td>
      <td>0.878249</td>
      <td>1.615832</td>
      <td>0.0</td>
      <td>1.125249</td>
      <td>0.260483</td>
      <td>1.940736</td>
    </tr>
    <tr>
      <th>Tryglicerides</th>
      <td>0.134283</td>
      <td>1.143716</td>
      <td>0.138146</td>
      <td>-0.136478</td>
      <td>0.405044</td>
      <td>0.872425</td>
      <td>1.499369</td>
      <td>0.0</td>
      <td>0.972036</td>
      <td>0.331032</td>
      <td>1.594955</td>
    </tr>
    <tr>
      <th>Platelets</th>
      <td>-0.039691</td>
      <td>0.961087</td>
      <td>0.129440</td>
      <td>-0.293388</td>
      <td>0.214007</td>
      <td>0.745733</td>
      <td>1.238631</td>
      <td>0.0</td>
      <td>-0.306634</td>
      <td>0.759122</td>
      <td>0.397596</td>
    </tr>
    <tr>
      <th>Prothrombin</th>
      <td>0.359660</td>
      <td>1.432843</td>
      <td>0.139796</td>
      <td>0.085665</td>
      <td>0.633656</td>
      <td>1.089441</td>
      <td>1.884487</td>
      <td>0.0</td>
      <td>2.572748</td>
      <td>0.010089</td>
      <td>6.631007</td>
    </tr>
    <tr>
      <th>Drug</th>
      <td>-0.236093</td>
      <td>0.789708</td>
      <td>0.250917</td>
      <td>-0.727881</td>
      <td>0.255696</td>
      <td>0.482931</td>
      <td>1.291360</td>
      <td>0.0</td>
      <td>-0.940919</td>
      <td>0.346746</td>
      <td>1.528048</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>0.088317</td>
      <td>1.092335</td>
      <td>0.354036</td>
      <td>-0.605581</td>
      <td>0.782216</td>
      <td>0.545757</td>
      <td>2.186311</td>
      <td>0.0</td>
      <td>0.249459</td>
      <td>0.803006</td>
      <td>0.316517</td>
    </tr>
    <tr>
      <th>Ascites</th>
      <td>0.101744</td>
      <td>1.107100</td>
      <td>0.401745</td>
      <td>-0.685661</td>
      <td>0.889150</td>
      <td>0.503757</td>
      <td>2.433060</td>
      <td>0.0</td>
      <td>0.253256</td>
      <td>0.800071</td>
      <td>0.321801</td>
    </tr>
    <tr>
      <th>Hepatomegaly</th>
      <td>0.050489</td>
      <td>1.051785</td>
      <td>0.282509</td>
      <td>-0.503219</td>
      <td>0.604197</td>
      <td>0.604581</td>
      <td>1.829782</td>
      <td>0.0</td>
      <td>0.178715</td>
      <td>0.858161</td>
      <td>0.220679</td>
    </tr>
    <tr>
      <th>Spiders</th>
      <td>-0.043470</td>
      <td>0.957461</td>
      <td>0.288959</td>
      <td>-0.609820</td>
      <td>0.522879</td>
      <td>0.543449</td>
      <td>1.686878</td>
      <td>0.0</td>
      <td>-0.150438</td>
      <td>0.880419</td>
      <td>0.183737</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>0.544039</td>
      <td>1.722952</td>
      <td>0.309540</td>
      <td>-0.062647</td>
      <td>1.150726</td>
      <td>0.939275</td>
      <td>3.160486</td>
      <td>0.0</td>
      <td>1.757576</td>
      <td>0.078820</td>
      <td>3.665300</td>
    </tr>
    <tr>
      <th>Stage_4.0</th>
      <td>0.224185</td>
      <td>1.251302</td>
      <td>0.301255</td>
      <td>-0.366264</td>
      <td>0.814633</td>
      <td>0.693320</td>
      <td>2.258347</td>
      <td>0.0</td>
      <td>0.744169</td>
      <td>0.456774</td>
      <td>1.130447</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting the hazard ratio of the
# formulated Cox Regression model
# with No Penalty
##################################
cirrhosis_survival_coxph_L1_0_L2_0_summary = cirrhosis_survival_coxph_L1_0_L2_0.summary
cirrhosis_survival_coxph_L1_0_L2_0_summary['hazard_ratio'] = np.exp(cirrhosis_survival_coxph_L1_0_L2_0_summary['coef'])
significant = cirrhosis_survival_coxph_L1_0_L2_0_summary['p'] < 0.05
plt.figure(figsize=(17, 8))
colors = ['#993300' if sig else '#CC9966' for sig in significant]

plt.barh(cirrhosis_survival_coxph_L1_0_L2_0_summary.index, 
         cirrhosis_survival_coxph_L1_0_L2_0_summary['hazard_ratio'], 
         xerr=cirrhosis_survival_coxph_L1_0_L2_0_summary['se(coef)'], 
         color=colors)
plt.xlabel('Hazard Ratio')
plt.ylabel('Variables')
plt.title('COXPH_NP Hazard Ratio Forest Plot')
plt.axvline(x=1, color='k', linestyle='--')
plt.gca().invert_yaxis()
plt.show()
```


    
![png](output_163_0.png)
    



```python
##################################
# Plotting the coefficient magnitude
# of the formulated Cox Regression model
# with No Penalty
##################################
plt.figure(figsize=(17, 8))
colors = ['#993300' if sig else '#CC9966' for sig in significant]
cirrhosis_survival_coxph_L1_0_L2_0_summary['coef'].plot(kind='barh', color=colors)
plt.xlabel('Variables')
plt.ylabel('Model Coefficient Value')
plt.title('COXPH_NP Model Coefficients')
plt.xticks(rotation=0, ha='right')
plt.xlim(-1,1)
plt.gca().invert_yaxis()
plt.show()
```


    
![png](output_164_0.png)
    



```python
##################################
# Gathering the apparent model performance
# as baseline for evaluating overfitting
##################################
cirrhosis_survival_coxph_L1_0_L2_0.fit(cirrhosis_survival_train_modeling, duration_col='N_Days', event_col='Status')
train_predictions = cirrhosis_survival_coxph_L1_0_L2_0.predict_partial_hazard(cirrhosis_survival_train_modeling)
cirrhosis_survival_coxph_L1_0_L2_0_train_ci = concordance_index(cirrhosis_survival_train_modeling['N_Days'], 
                                                                     -train_predictions, 
                                                                     cirrhosis_survival_train_modeling['Status'])
display(f"Apparent Concordance Index: {cirrhosis_survival_coxph_L1_0_L2_0_train_ci}")
```


    'Apparent Concordance Index: 0.8478543563068921'



```python
##################################
# Performing 5-Fold Cross-Validation
# on the training data
##################################
kf = KFold(n_splits=5, shuffle=True, random_state=88888888)
c_index_scores = []

for train_index, val_index in kf.split(cirrhosis_survival_train_modeling):
    df_train_fold = cirrhosis_survival_train_modeling.iloc[train_index]
    df_val_fold = cirrhosis_survival_train_modeling.iloc[val_index]
    
    cirrhosis_survival_coxph_L1_0_L2_0.fit(df_train_fold, duration_col='N_Days', event_col='Status')
    val_predictions = cirrhosis_survival_coxph_L1_0_L2_0.predict_partial_hazard(df_val_fold)
    c_index = concordance_index(df_val_fold['N_Days'], -val_predictions, df_val_fold['Status'])
    c_index_scores.append(c_index)

cirrhosis_survival_coxph_L1_0_L2_0_cv_ci_mean = np.mean(c_index_scores)
cirrhosis_survival_coxph_L1_0_L2_0_cv_ci_std = np.std(c_index_scores)

display(f"Cross-Validated Concordance Index: {cirrhosis_survival_coxph_L1_0_L2_0_cv_ci_mean}")
```


    'Cross-Validated Concordance Index: 0.8023642721538025'



```python
##################################
# Evaluating the model performance
# on test data
##################################
test_predictions = cirrhosis_survival_coxph_L1_0_L2_0.predict_partial_hazard(cirrhosis_survival_test_modeling)
cirrhosis_survival_coxph_L1_0_L2_0_test_ci = concordance_index(cirrhosis_survival_test_modeling['N_Days'], 
                                                                     -test_predictions, 
                                                                     cirrhosis_survival_test_modeling['Status'])
display(f"Test Concordance Index: {cirrhosis_survival_coxph_L1_0_L2_0_test_ci}")
```


    'Test Concordance Index: 0.8480725623582767'



```python
##################################
# Gathering the concordance indices
# from training, cross-validation and test
##################################
coxph_L1_0_L2_0_set = pd.DataFrame(["Train","Cross-Validation","Test"])
coxph_L1_0_L2_0_ci_values = pd.DataFrame([cirrhosis_survival_coxph_L1_0_L2_0_train_ci,
                                           cirrhosis_survival_coxph_L1_0_L2_0_cv_ci_mean,
                                           cirrhosis_survival_coxph_L1_0_L2_0_test_ci])
coxph_L1_0_L2_0_method = pd.DataFrame(["COXPH_NP"]*3)
coxph_L1_0_L2_0_summary = pd.concat([coxph_L1_0_L2_0_set, 
                                     coxph_L1_0_L2_0_ci_values,
                                     coxph_L1_0_L2_0_method], 
                                    axis=1)
coxph_L1_0_L2_0_summary.columns = ['Set', 'Concordance.Index', 'Method']
coxph_L1_0_L2_0_summary.reset_index(inplace=True, drop=True)
display(coxph_L1_0_L2_0_summary)
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
      <th>Set</th>
      <th>Concordance.Index</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Train</td>
      <td>0.847854</td>
      <td>COXPH_NP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cross-Validation</td>
      <td>0.802364</td>
      <td>COXPH_NP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Test</td>
      <td>0.848073</td>
      <td>COXPH_NP</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Binning the predicted risks
# into dichotomous groups and
# exploring the relationships with
# survival event and duration
##################################
cirrhosis_survival_test_modeling['Predicted_Risks_COXPH_NP'] = test_predictions
cirrhosis_survival_test_modeling['Predicted_RiskGroups_COXPH_NP'] = risk_groups = pd.qcut(cirrhosis_survival_test_modeling['Predicted_Risks_COXPH_NP'], 2, labels=['Low-Risk', 'High-Risk'])

plt.figure(figsize=(17, 8))
for group in risk_groups.unique():
    group_data = cirrhosis_survival_test_modeling[risk_groups == group]
    kmf.fit(group_data['N_Days'], event_observed=group_data['Status'], label=group)
    kmf.plot_survival_function()

plt.title('COXPH_NP Survival Probabilities by Predicted Risk Groups')
plt.xlabel('N_Days')
plt.ylabel('Event Survival Probability')
plt.show()
```


    
![png](output_169_0.png)
    



```python
##################################
# Gathering the predictor information
# for 5 test case samples
##################################
test_case_details = cirrhosis_survival_test_modeling.iloc[[10, 20, 30, 40, 50]]
display(test_case_details)
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
      <th>Prothrombin</th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_4.0</th>
      <th>Predicted_Risks_COXPH_NP</th>
      <th>Predicted_RiskGroups_COXPH_NP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>True</td>
      <td>1827</td>
      <td>0.226982</td>
      <td>1.530100</td>
      <td>1.302295</td>
      <td>1.331981</td>
      <td>1.916467</td>
      <td>-0.477846</td>
      <td>-0.451305</td>
      <td>2.250260</td>
      <td>...</td>
      <td>0.546417</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>8.397958</td>
      <td>High-Risk</td>
    </tr>
    <tr>
      <th>20</th>
      <td>False</td>
      <td>1447</td>
      <td>-0.147646</td>
      <td>0.061189</td>
      <td>0.793618</td>
      <td>-1.158235</td>
      <td>0.861264</td>
      <td>0.625621</td>
      <td>0.319035</td>
      <td>0.446026</td>
      <td>...</td>
      <td>-1.508571</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.466233</td>
      <td>Low-Risk</td>
    </tr>
    <tr>
      <th>30</th>
      <td>False</td>
      <td>2574</td>
      <td>0.296370</td>
      <td>-1.283677</td>
      <td>0.169685</td>
      <td>3.237777</td>
      <td>-1.008276</td>
      <td>-0.873566</td>
      <td>-0.845549</td>
      <td>-0.351236</td>
      <td>...</td>
      <td>-0.617113</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.153047</td>
      <td>Low-Risk</td>
    </tr>
    <tr>
      <th>40</th>
      <td>True</td>
      <td>3762</td>
      <td>0.392609</td>
      <td>-0.096645</td>
      <td>-0.486337</td>
      <td>1.903146</td>
      <td>-0.546292</td>
      <td>-0.247141</td>
      <td>-0.720619</td>
      <td>-0.810790</td>
      <td>...</td>
      <td>1.402075</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2.224881</td>
      <td>High-Risk</td>
    </tr>
    <tr>
      <th>50</th>
      <td>False</td>
      <td>837</td>
      <td>-0.813646</td>
      <td>1.089037</td>
      <td>0.064451</td>
      <td>0.212865</td>
      <td>2.063138</td>
      <td>-0.224432</td>
      <td>0.074987</td>
      <td>2.333282</td>
      <td>...</td>
      <td>-1.125995</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2.886051</td>
      <td>High-Risk</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



```python
##################################
# Gathering the risk-groups
# for 5 test case samples
##################################
print(cirrhosis_survival_test_modeling.loc[[10, 20, 30, 40, 50]][['Predicted_RiskGroups_COXPH_NP']])
```

       Predicted_RiskGroups_COXPH_NP
    10                     High-Risk
    20                      Low-Risk
    30                      Low-Risk
    40                     High-Risk
    50                     High-Risk
    


```python
##################################
# Estimating the cumulative hazard
# and survival functions
# for 5 test cases
##################################
test_case = cirrhosis_survival_test_modeling.iloc[[10, 20, 30, 40, 50]]
test_case_labels = ['Patient_10','Patient_20','Patient_30','Patient_40','Patient_50']

fig, axes = plt.subplots(1, 2, figsize=(17, 8))
for i, (index, row) in enumerate(test_case.iterrows()):
    survival_function = cirrhosis_survival_coxph_L1_0_L2_0.predict_survival_function(row.to_frame().T)
    axes[0].plot(survival_function, label=f'Sample {i+1}')
axes[0].set_title('COXPH_NP Survival Function for 5 Test Cases')
axes[0].set_xlabel('N_Days')
axes[0].set_ylim(0,1)
axes[0].set_ylabel('Survival Probability')
axes[0].legend(test_case_labels, loc="lower left")
for i, (index, row) in enumerate(test_case.iterrows()):
    hazard_function = cirrhosis_survival_coxph_L1_0_L2_0.predict_cumulative_hazard(row.to_frame().T)
    axes[1].plot(hazard_function, label=f'Sample {i+1}')
axes[1].set_title('COXPH_NP Cumulative Hazard for 5 Test Cases')
axes[1].set_xlabel('N_Days')
axes[1].set_ylim(0,5)
axes[1].set_ylabel('Cumulative Hazard')
axes[1].legend(test_case_labels, loc="upper left")
plt.tight_layout()
plt.show()
```


    
![png](output_172_0.png)
    



```python
##################################
# Creating the explainer object
##################################
cirrhosis_survival_coxph_L1_0_L2_0_explainer = shap.Explainer(cirrhosis_survival_coxph_L1_0_L2_0.predict_partial_hazard, 
                                                    cirrhosis_survival_train_modeling.drop(columns=["N_Days", "Status"]))
cirrhosis_survival_coxph_L1_0_L2_0_shap_values = cirrhosis_survival_coxph_L1_0_L2_0_explainer(cirrhosis_survival_train_modeling.drop(columns=["N_Days", "Status"]))
```

    PermutationExplainer explainer: 219it [00:30,  5.67it/s]                         
    


```python
##################################
# Plotting the SHAP summary plot
##################################
shap.summary_plot(cirrhosis_survival_coxph_L1_0_L2_0_shap_values, 
                  cirrhosis_survival_train_modeling.drop(columns=["N_Days", "Status"]),
                  sort=False)
```


    
![png](output_174_0.png)
    


### 1.6.3 Cox Regression With Full L1 Penalty <a class="anchor" id="1.6.3"></a>

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Survival Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) refer to statistical methods used to analyze survival data, accounting for censored observations. These models aim to describe the relationship between survival time and one or more predictor variables, and to estimate the survival function and hazard function. Survival models are essential for understanding the factors that influence time-to-event data, allowing for predictions and comparisons between different groups or treatment effects. They are widely used in clinical trials, reliability engineering, and other areas where time-to-event data is prevalent.

[Cox Proportional Hazards Regression](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1972.tb00899.x) is a semiparametric model used to study the relationship between the survival time of subjects and one or more predictor variables. The model assumes that the hazard ratio (the risk of the event occurring at a specific time) is a product of a baseline hazard function and an exponential function of the predictor variables. It also does not require the baseline hazard to be specified, thus making it a semiparametric model. As a method, it is well-established and widely used in survival analysis, can handle time-dependent covariates and provides a relatively straightforward interpretation. However, the process assumes proportional hazards, which may not hold in all datasets, and may be less flexible in capturing complex relationships between variables and survival times compared to some machine learning models. Given a dataset with survival times, event indicators, and predictor variables, the algorithm involves defining the partial likelihood function for the Cox model (which only considers the relative ordering of survival times); using optimization techniques to estimate the regression coefficients by maximizing the log-partial likelihood; estimating the baseline hazard function (although it is not explicitly required for predictions); and calculating the hazard function and survival function for new data using the estimated coefficients and baseline hazard.

[Lasso Penalty](https://lifelines.readthedocs.io/en/latest/), or L1 regularization in cox regression, adds a constraint to the sum of the absolute values of the coefficients. The penalized log-likelihood function is composed of the partial likelihood of the cox model, a tuning parameter that controls the strength of the penalty, and the sum of the absolute model coefficients. The Lasso penalty encourages sparsity in the coefficients, meaning it tends to set some coefficients exactly to zero, effectively performing variable selection.

1. The [cox proportional hazards regression model](https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html) from the <mark style="background-color: #CCECFF"><b>lifelines.CoxPHFitter</b></mark> Python library API was implemented. 
2. The model implementation used 2 hyperparameters:
    * <span style="color: #FF0000">penalizer</span> = penalty to the size of the coefficients during regression fixed at a value = 0.10
    * <span style="color: #FF0000">l1_ratio</span> = proportion of the L1 versus L2 penalty fixed at a value = 1.00
3. Only 8 out of the 17 variables were used for prediction given the non-zero values of the model coefficients.
4. Out of all 8 predictors, only 1 variable was statistically significant:
    * <span style="color: #FF0000">Bilirubin</span>
5. The cross-validated model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8111
6. The apparent model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8313
7. The independent test model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8426
8. Considerable difference in the apparent and cross-validated model performance observed, indicative of the presence of minimal model overfitting.
9. Survival probability curves obtained from the groups generated by dichotomizing the risk scores demonstrated sufficient differentiation across the entire duration.
10. Hazard and survival probability estimations for 5 sampled cases demonstrated reasonably smooth profiles.


```python
##################################
# Formulating the Cox Regression model
# with Full L1 Penalty
# and generating the summary
##################################
cirrhosis_survival_coxph_L1_100_L2_0 = CoxPHFitter(penalizer=0.10, l1_ratio=1.00)
cirrhosis_survival_coxph_L1_100_L2_0.fit(cirrhosis_survival_train_modeling, duration_col='N_Days', event_col='Status')
cirrhosis_survival_coxph_L1_100_L2_0.print_summary()
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
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.CoxPHFitter</td>
    </tr>
    <tr>
      <th>duration col</th>
      <td>'N_Days'</td>
    </tr>
    <tr>
      <th>event col</th>
      <td>'Status'</td>
    </tr>
    <tr>
      <th>penalizer</th>
      <td>0.1</td>
    </tr>
    <tr>
      <th>l1 ratio</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>baseline estimation</th>
      <td>breslow</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>218</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>87</td>
    </tr>
    <tr>
      <th>partial log-likelihood</th>
      <td>-391.01</td>
    </tr>
    <tr>
      <th>time fit was run</th>
      <td>2024-07-17 11:42:08 UTC</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">exp(coef)</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">exp(coef) lower 95%</th>
      <th style="min-width: 12px;">exp(coef) upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>0.10</td>
      <td>1.11</td>
      <td>0.11</td>
      <td>-0.12</td>
      <td>0.32</td>
      <td>0.89</td>
      <td>1.38</td>
      <td>0.00</td>
      <td>0.90</td>
      <td>0.37</td>
      <td>1.44</td>
    </tr>
    <tr>
      <th>Bilirubin</th>
      <td>0.72</td>
      <td>2.06</td>
      <td>0.16</td>
      <td>0.42</td>
      <td>1.02</td>
      <td>1.52</td>
      <td>2.79</td>
      <td>0.00</td>
      <td>4.64</td>
      <td>&lt;0.005</td>
      <td>18.14</td>
    </tr>
    <tr>
      <th>Cholesterol</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Albumin</th>
      <td>-0.06</td>
      <td>0.94</td>
      <td>0.13</td>
      <td>-0.32</td>
      <td>0.20</td>
      <td>0.73</td>
      <td>1.22</td>
      <td>0.00</td>
      <td>-0.45</td>
      <td>0.65</td>
      <td>0.61</td>
    </tr>
    <tr>
      <th>Copper</th>
      <td>0.03</td>
      <td>1.03</td>
      <td>0.13</td>
      <td>-0.24</td>
      <td>0.29</td>
      <td>0.79</td>
      <td>1.33</td>
      <td>0.00</td>
      <td>0.19</td>
      <td>0.85</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>Alk_Phos</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>SGOT</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Tryglicerides</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Platelets</th>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Prothrombin</th>
      <td>0.21</td>
      <td>1.23</td>
      <td>0.13</td>
      <td>-0.04</td>
      <td>0.47</td>
      <td>0.96</td>
      <td>1.59</td>
      <td>0.00</td>
      <td>1.62</td>
      <td>0.11</td>
      <td>3.25</td>
    </tr>
    <tr>
      <th>Drug</th>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Ascites</th>
      <td>0.16</td>
      <td>1.17</td>
      <td>0.41</td>
      <td>-0.64</td>
      <td>0.97</td>
      <td>0.53</td>
      <td>2.63</td>
      <td>0.00</td>
      <td>0.39</td>
      <td>0.70</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>Hepatomegaly</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Spiders</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>0.17</td>
      <td>1.18</td>
      <td>0.32</td>
      <td>-0.45</td>
      <td>0.79</td>
      <td>0.64</td>
      <td>2.20</td>
      <td>0.00</td>
      <td>0.53</td>
      <td>0.60</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>Stage_4.0</th>
      <td>0.03</td>
      <td>1.03</td>
      <td>0.27</td>
      <td>-0.50</td>
      <td>0.56</td>
      <td>0.61</td>
      <td>1.75</td>
      <td>0.00</td>
      <td>0.12</td>
      <td>0.91</td>
      <td>0.14</td>
    </tr>
  </tbody>
</table><br><div>
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
  <tbody>
    <tr>
      <th>Concordance</th>
      <td>0.83</td>
    </tr>
    <tr>
      <th>Partial AIC</th>
      <td>816.03</td>
    </tr>
    <tr>
      <th>log-likelihood ratio test</th>
      <td>52.70 on 17 df</td>
    </tr>
    <tr>
      <th>-log2(p) of ll-ratio test</th>
      <td>15.94</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Consolidating the detailed values
# of the model coefficients
##################################
cirrhosis_survival_coxph_L1_100_L2_0.summary
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
      <th>coef</th>
      <th>exp(coef)</th>
      <th>se(coef)</th>
      <th>coef lower 95%</th>
      <th>coef upper 95%</th>
      <th>exp(coef) lower 95%</th>
      <th>exp(coef) upper 95%</th>
      <th>cmp to</th>
      <th>z</th>
      <th>p</th>
      <th>-log2(p)</th>
    </tr>
    <tr>
      <th>covariate</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>1.019423e-01</td>
      <td>1.107320</td>
      <td>0.113381</td>
      <td>-0.120280</td>
      <td>0.324165</td>
      <td>0.886672</td>
      <td>1.382875</td>
      <td>0.0</td>
      <td>0.899114</td>
      <td>0.368592</td>
      <td>1.439903</td>
    </tr>
    <tr>
      <th>Bilirubin</th>
      <td>7.205550e-01</td>
      <td>2.055574</td>
      <td>0.155250</td>
      <td>0.416270</td>
      <td>1.024840</td>
      <td>1.516295</td>
      <td>2.786650</td>
      <td>0.0</td>
      <td>4.641247</td>
      <td>0.000003</td>
      <td>18.139495</td>
    </tr>
    <tr>
      <th>Cholesterol</th>
      <td>8.673029e-09</td>
      <td>1.000000</td>
      <td>0.000078</td>
      <td>-0.000153</td>
      <td>0.000153</td>
      <td>0.999847</td>
      <td>1.000153</td>
      <td>0.0</td>
      <td>0.000111</td>
      <td>0.999911</td>
      <td>0.000128</td>
    </tr>
    <tr>
      <th>Albumin</th>
      <td>-5.899624e-02</td>
      <td>0.942710</td>
      <td>0.132000</td>
      <td>-0.317712</td>
      <td>0.199719</td>
      <td>0.727813</td>
      <td>1.221060</td>
      <td>0.0</td>
      <td>-0.446941</td>
      <td>0.654918</td>
      <td>0.610614</td>
    </tr>
    <tr>
      <th>Copper</th>
      <td>2.602826e-02</td>
      <td>1.026370</td>
      <td>0.133726</td>
      <td>-0.236070</td>
      <td>0.288127</td>
      <td>0.789725</td>
      <td>1.333926</td>
      <td>0.0</td>
      <td>0.194639</td>
      <td>0.845676</td>
      <td>0.241823</td>
    </tr>
    <tr>
      <th>Alk_Phos</th>
      <td>6.649722e-08</td>
      <td>1.000000</td>
      <td>0.000088</td>
      <td>-0.000172</td>
      <td>0.000172</td>
      <td>0.999828</td>
      <td>1.000172</td>
      <td>0.0</td>
      <td>0.000757</td>
      <td>0.999396</td>
      <td>0.000872</td>
    </tr>
    <tr>
      <th>SGOT</th>
      <td>1.196063e-07</td>
      <td>1.000000</td>
      <td>0.000112</td>
      <td>-0.000219</td>
      <td>0.000219</td>
      <td>0.999781</td>
      <td>1.000219</td>
      <td>0.0</td>
      <td>0.001069</td>
      <td>0.999147</td>
      <td>0.001232</td>
    </tr>
    <tr>
      <th>Tryglicerides</th>
      <td>1.859293e-07</td>
      <td>1.000000</td>
      <td>0.000168</td>
      <td>-0.000329</td>
      <td>0.000330</td>
      <td>0.999671</td>
      <td>1.000330</td>
      <td>0.0</td>
      <td>0.001106</td>
      <td>0.999118</td>
      <td>0.001273</td>
    </tr>
    <tr>
      <th>Platelets</th>
      <td>-1.591429e-07</td>
      <td>1.000000</td>
      <td>0.000141</td>
      <td>-0.000277</td>
      <td>0.000277</td>
      <td>0.999723</td>
      <td>1.000277</td>
      <td>0.0</td>
      <td>-0.001127</td>
      <td>0.999101</td>
      <td>0.001298</td>
    </tr>
    <tr>
      <th>Prothrombin</th>
      <td>2.106760e-01</td>
      <td>1.234512</td>
      <td>0.130005</td>
      <td>-0.044129</td>
      <td>0.465481</td>
      <td>0.956830</td>
      <td>1.592780</td>
      <td>0.0</td>
      <td>1.620523</td>
      <td>0.105120</td>
      <td>3.249890</td>
    </tr>
    <tr>
      <th>Drug</th>
      <td>-2.885452e-08</td>
      <td>1.000000</td>
      <td>0.000157</td>
      <td>-0.000308</td>
      <td>0.000308</td>
      <td>0.999692</td>
      <td>1.000308</td>
      <td>0.0</td>
      <td>-0.000184</td>
      <td>0.999853</td>
      <td>0.000212</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>-2.332571e-07</td>
      <td>1.000000</td>
      <td>0.000274</td>
      <td>-0.000538</td>
      <td>0.000537</td>
      <td>0.999462</td>
      <td>1.000537</td>
      <td>0.0</td>
      <td>-0.000851</td>
      <td>0.999321</td>
      <td>0.000979</td>
    </tr>
    <tr>
      <th>Ascites</th>
      <td>1.608914e-01</td>
      <td>1.174557</td>
      <td>0.410628</td>
      <td>-0.643925</td>
      <td>0.965708</td>
      <td>0.525227</td>
      <td>2.626647</td>
      <td>0.0</td>
      <td>0.391818</td>
      <td>0.695193</td>
      <td>0.524514</td>
    </tr>
    <tr>
      <th>Hepatomegaly</th>
      <td>3.202570e-07</td>
      <td>1.000000</td>
      <td>0.000284</td>
      <td>-0.000557</td>
      <td>0.000557</td>
      <td>0.999443</td>
      <td>1.000558</td>
      <td>0.0</td>
      <td>0.001127</td>
      <td>0.999101</td>
      <td>0.001298</td>
    </tr>
    <tr>
      <th>Spiders</th>
      <td>1.774944e-07</td>
      <td>1.000000</td>
      <td>0.000208</td>
      <td>-0.000407</td>
      <td>0.000408</td>
      <td>0.999593</td>
      <td>1.000408</td>
      <td>0.0</td>
      <td>0.000854</td>
      <td>0.999319</td>
      <td>0.000983</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>1.676711e-01</td>
      <td>1.182548</td>
      <td>0.315894</td>
      <td>-0.451471</td>
      <td>0.786813</td>
      <td>0.636691</td>
      <td>2.196385</td>
      <td>0.0</td>
      <td>0.530782</td>
      <td>0.595570</td>
      <td>0.747657</td>
    </tr>
    <tr>
      <th>Stage_4.0</th>
      <td>3.156528e-02</td>
      <td>1.032069</td>
      <td>0.270735</td>
      <td>-0.499066</td>
      <td>0.562196</td>
      <td>0.607098</td>
      <td>1.754521</td>
      <td>0.0</td>
      <td>0.116591</td>
      <td>0.907184</td>
      <td>0.140533</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting the hazard ratio of the
# formulated Cox Regression model
# with Full L1 Penalty
##################################
cirrhosis_survival_coxph_L1_100_L2_0_summary = cirrhosis_survival_coxph_L1_100_L2_0.summary
cirrhosis_survival_coxph_L1_100_L2_0_summary['hazard_ratio'] = np.exp(cirrhosis_survival_coxph_L1_100_L2_0_summary['coef'])
significant = cirrhosis_survival_coxph_L1_100_L2_0_summary['p'] < 0.05
plt.figure(figsize=(17, 8))
colors = ['#993300' if sig else '#CC9966' for sig in significant]

plt.barh(cirrhosis_survival_coxph_L1_100_L2_0_summary.index, 
         cirrhosis_survival_coxph_L1_100_L2_0_summary['hazard_ratio'], 
         xerr=cirrhosis_survival_coxph_L1_100_L2_0_summary['se(coef)'], 
         color=colors)
plt.xlabel('Hazard Ratio')
plt.ylabel('Variables')
plt.title('COXPH_FL1P Hazard Ratio Forest Plot')
plt.axvline(x=1, color='k', linestyle='--')
plt.gca().invert_yaxis()
plt.show()
```


    
![png](output_178_0.png)
    



```python
##################################
# Plotting the coefficient magnitude
# of the formulated Cox Regression model
# with Full L1 Penalty
##################################
plt.figure(figsize=(17, 8))
colors = ['#993300' if sig else '#CC9966' for sig in significant]
cirrhosis_survival_coxph_L1_100_L2_0_summary['coef'].plot(kind='barh', color=colors)
plt.xlabel('Variables')
plt.ylabel('Model Coefficient Value')
plt.title('COXPH_FL1P Model Coefficients')
plt.xticks(rotation=0, ha='right')
plt.xlim(-1,1)
plt.gca().invert_yaxis()
plt.show()
```


    
![png](output_179_0.png)
    



```python
##################################
# Gathering the apparent model performance
# as baseline for evaluating overfitting
##################################
cirrhosis_survival_coxph_L1_100_L2_0.fit(cirrhosis_survival_train_modeling, duration_col='N_Days', event_col='Status')
train_predictions = cirrhosis_survival_coxph_L1_100_L2_0.predict_partial_hazard(cirrhosis_survival_train_modeling)
cirrhosis_survival_coxph_L1_100_L2_0_train_ci = concordance_index(cirrhosis_survival_train_modeling['N_Days'], 
                                                                     -train_predictions, 
                                                                     cirrhosis_survival_train_modeling['Status'])
display(f"Apparent Concordance Index: {cirrhosis_survival_coxph_L1_100_L2_0_train_ci}")
```


    'Apparent Concordance Index: 0.8313556566970091'



```python
##################################
# Performing 5-Fold Cross-Validation
# on the training data
##################################
kf = KFold(n_splits=5, shuffle=True, random_state=88888888)
c_index_scores = []

for train_index, val_index in kf.split(cirrhosis_survival_train_modeling):
    df_train_fold = cirrhosis_survival_train_modeling.iloc[train_index]
    df_val_fold = cirrhosis_survival_train_modeling.iloc[val_index]
    
    cirrhosis_survival_coxph_L1_100_L2_0.fit(df_train_fold, duration_col='N_Days', event_col='Status')
    val_predictions = cirrhosis_survival_coxph_L1_100_L2_0.predict_partial_hazard(df_val_fold)
    c_index = concordance_index(df_val_fold['N_Days'], -val_predictions, df_val_fold['Status'])
    c_index_scores.append(c_index)

cirrhosis_survival_coxph_L1_100_L2_0_cv_ci_mean = np.mean(c_index_scores)
cirrhosis_survival_coxph_L1_100_L2_0_cv_ci_std = np.std(c_index_scores)

display(f"Cross-Validated Concordance Index: {cirrhosis_survival_coxph_L1_100_L2_0_cv_ci_mean}")
```


    'Cross-Validated Concordance Index: 0.8111973914136674'



```python
##################################
# Evaluating the model performance
# on test data
##################################
test_predictions = cirrhosis_survival_coxph_L1_100_L2_0.predict_partial_hazard(cirrhosis_survival_test_modeling)
cirrhosis_survival_coxph_L1_100_L2_0_test_ci = concordance_index(cirrhosis_survival_test_modeling['N_Days'], 
                                                                     -test_predictions, 
                                                                     cirrhosis_survival_test_modeling['Status'])
display(f"Test Concordance Index: {cirrhosis_survival_coxph_L1_100_L2_0_test_ci}")
```


    'Test Concordance Index: 0.8426303854875283'



```python
##################################
# Gathering the concordance indices
# from training, cross-validation and test
##################################
coxph_L1_100_L2_0_set = pd.DataFrame(["Train","Cross-Validation","Test"])
coxph_L1_100_L2_0_ci_values = pd.DataFrame([cirrhosis_survival_coxph_L1_100_L2_0_train_ci,
                                           cirrhosis_survival_coxph_L1_100_L2_0_cv_ci_mean,
                                           cirrhosis_survival_coxph_L1_100_L2_0_test_ci])
coxph_L1_100_L2_0_method = pd.DataFrame(["COXPH_FL1P"]*3)
coxph_L1_100_L2_0_summary = pd.concat([coxph_L1_100_L2_0_set, 
                                     coxph_L1_100_L2_0_ci_values,
                                     coxph_L1_100_L2_0_method], 
                                    axis=1)
coxph_L1_100_L2_0_summary.columns = ['Set', 'Concordance.Index', 'Method']
coxph_L1_100_L2_0_summary.reset_index(inplace=True, drop=True)
display(coxph_L1_100_L2_0_summary)
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
      <th>Set</th>
      <th>Concordance.Index</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Train</td>
      <td>0.831356</td>
      <td>COXPH_FL1P</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cross-Validation</td>
      <td>0.811197</td>
      <td>COXPH_FL1P</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Test</td>
      <td>0.842630</td>
      <td>COXPH_FL1P</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Binning the predicted risks
# into dichotomous groups and
# exploring the relationships with
# survival event and duration
##################################
cirrhosis_survival_test_modeling['Predicted_Risks_COXPH_FL1P'] = test_predictions
cirrhosis_survival_test_modeling['Predicted_RiskGroups_COXPH_FL1P'] = risk_groups = pd.qcut(cirrhosis_survival_test_modeling['Predicted_Risks_COXPH_FL1P'], 2, labels=['Low-Risk', 'High-Risk'])

plt.figure(figsize=(17, 8))
for group in risk_groups.unique():
    group_data = cirrhosis_survival_test_modeling[risk_groups == group]
    kmf.fit(group_data['N_Days'], event_observed=group_data['Status'], label=group)
    kmf.plot_survival_function()

plt.title('COXPH_FL1P Survival Probabilities by Predicted Risk Groups')
plt.xlabel('N_Days')
plt.ylabel('Event Survival Probability')
plt.show()
```


    
![png](output_184_0.png)
    



```python
##################################
# Gathering the predictor information
# for 5 test case samples
##################################
test_case_details = cirrhosis_survival_test_modeling.iloc[[10, 20, 30, 40, 50]]
display(test_case_details)
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
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_4.0</th>
      <th>Predicted_Risks_COXPH_NP</th>
      <th>Predicted_RiskGroups_COXPH_NP</th>
      <th>Predicted_Risks_COXPH_FL1P</th>
      <th>Predicted_RiskGroups_COXPH_FL1P</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>True</td>
      <td>1827</td>
      <td>0.226982</td>
      <td>1.530100</td>
      <td>1.302295</td>
      <td>1.331981</td>
      <td>1.916467</td>
      <td>-0.477846</td>
      <td>-0.451305</td>
      <td>2.250260</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>8.397958</td>
      <td>High-Risk</td>
      <td>3.515780</td>
      <td>High-Risk</td>
    </tr>
    <tr>
      <th>20</th>
      <td>False</td>
      <td>1447</td>
      <td>-0.147646</td>
      <td>0.061189</td>
      <td>0.793618</td>
      <td>-1.158235</td>
      <td>0.861264</td>
      <td>0.625621</td>
      <td>0.319035</td>
      <td>0.446026</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.466233</td>
      <td>Low-Risk</td>
      <td>0.769723</td>
      <td>Low-Risk</td>
    </tr>
    <tr>
      <th>30</th>
      <td>False</td>
      <td>2574</td>
      <td>0.296370</td>
      <td>-1.283677</td>
      <td>0.169685</td>
      <td>3.237777</td>
      <td>-1.008276</td>
      <td>-0.873566</td>
      <td>-0.845549</td>
      <td>-0.351236</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.153047</td>
      <td>Low-Risk</td>
      <td>0.320200</td>
      <td>Low-Risk</td>
    </tr>
    <tr>
      <th>40</th>
      <td>True</td>
      <td>3762</td>
      <td>0.392609</td>
      <td>-0.096645</td>
      <td>-0.486337</td>
      <td>1.903146</td>
      <td>-0.546292</td>
      <td>-0.247141</td>
      <td>-0.720619</td>
      <td>-0.810790</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2.224881</td>
      <td>High-Risk</td>
      <td>1.167777</td>
      <td>High-Risk</td>
    </tr>
    <tr>
      <th>50</th>
      <td>False</td>
      <td>837</td>
      <td>-0.813646</td>
      <td>1.089037</td>
      <td>0.064451</td>
      <td>0.212865</td>
      <td>2.063138</td>
      <td>-0.224432</td>
      <td>0.074987</td>
      <td>2.333282</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2.886051</td>
      <td>High-Risk</td>
      <td>1.832490</td>
      <td>High-Risk</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



```python
##################################
# Gathering the risk-groups
# for 5 test case samples
##################################
print(cirrhosis_survival_test_modeling.loc[[10, 20, 30, 40, 50]][['Predicted_RiskGroups_COXPH_FL1P']])
```

       Predicted_RiskGroups_COXPH_FL1P
    10                       High-Risk
    20                        Low-Risk
    30                        Low-Risk
    40                       High-Risk
    50                       High-Risk
    


```python
##################################
# Estimating the cumulative hazard
# and survival functions
# for 5 test cases
##################################
test_case = cirrhosis_survival_test_modeling.iloc[[10, 20, 30, 40, 50]]
test_case_labels = ['Patient_10','Patient_20','Patient_30','Patient_40','Patient_50']

fig, axes = plt.subplots(1, 2, figsize=(17, 8))
for i, (index, row) in enumerate(test_case.iterrows()):
    survival_function = cirrhosis_survival_coxph_L1_100_L2_0.predict_survival_function(row.to_frame().T)
    axes[0].plot(survival_function, label=f'Sample {i+1}')
axes[0].set_title('COXPH_FL1P Survival Function for 5 Test Cases')
axes[0].set_xlabel('N_Days')
axes[0].set_ylim(0,1)
axes[0].set_ylabel('Survival Probability')
axes[0].legend(test_case_labels, loc="lower left")
for i, (index, row) in enumerate(test_case.iterrows()):
    hazard_function = cirrhosis_survival_coxph_L1_100_L2_0.predict_cumulative_hazard(row.to_frame().T)
    axes[1].plot(hazard_function, label=f'Sample {i+1}')
axes[1].set_title('COXPH_FL1P Cumulative Hazard for 5 Test Cases')
axes[1].set_xlabel('N_Days')
axes[1].set_ylim(0,5)
axes[1].set_ylabel('Cumulative Hazard')
axes[1].legend(test_case_labels, loc="upper left")
plt.tight_layout()
plt.show()
```


    
![png](output_187_0.png)
    



```python
##################################
# Creating the explainer object
##################################
cirrhosis_survival_coxph_L1_100_L2_0_explainer = shap.Explainer(cirrhosis_survival_coxph_L1_100_L2_0.predict_partial_hazard, 
                                                    cirrhosis_survival_train_modeling.drop(columns=["N_Days", "Status"]))
cirrhosis_survival_coxph_L1_100_L2_0_shap_values = cirrhosis_survival_coxph_L1_100_L2_0_explainer(cirrhosis_survival_train_modeling.drop(columns=["N_Days", "Status"]))
```

    PermutationExplainer explainer: 219it [00:22,  5.36it/s]                         
    


```python
##################################
# Plotting the SHAP summary plot
##################################
shap.summary_plot(cirrhosis_survival_coxph_L1_100_L2_0_shap_values, 
                  cirrhosis_survival_train_modeling.drop(columns=["N_Days", "Status"]),
                  sort=False)
```


    
![png](output_189_0.png)
    


### 1.6.4 Cox Regression With Full L2 Penalty <a class="anchor" id="1.6.4"></a>

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Survival Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) refer to statistical methods used to analyze survival data, accounting for censored observations. These models aim to describe the relationship between survival time and one or more predictor variables, and to estimate the survival function and hazard function. Survival models are essential for understanding the factors that influence time-to-event data, allowing for predictions and comparisons between different groups or treatment effects. They are widely used in clinical trials, reliability engineering, and other areas where time-to-event data is prevalent.

[Cox Proportional Hazards Regression](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1972.tb00899.x) is a semiparametric model used to study the relationship between the survival time of subjects and one or more predictor variables. The model assumes that the hazard ratio (the risk of the event occurring at a specific time) is a product of a baseline hazard function and an exponential function of the predictor variables. It also does not require the baseline hazard to be specified, thus making it a semiparametric model. As a method, it is well-established and widely used in survival analysis, can handle time-dependent covariates and provides a relatively straightforward interpretation. However, the process assumes proportional hazards, which may not hold in all datasets, and may be less flexible in capturing complex relationships between variables and survival times compared to some machine learning models. Given a dataset with survival times, event indicators, and predictor variables, the algorithm involves defining the partial likelihood function for the Cox model (which only considers the relative ordering of survival times); using optimization techniques to estimate the regression coefficients by maximizing the log-partial likelihood; estimating the baseline hazard function (although it is not explicitly required for predictions); and calculating the hazard function and survival function for new data using the estimated coefficients and baseline hazard.

[Ridge Penalty](https://lifelines.readthedocs.io/en/latest/), or L2 regularization in cox regression, adds a constraint to the sum of the squared values of the coefficients. The penalized log-likelihood function is composed of the partial likelihood of the cox model, a tuning parameter that controls the strength of the penalty, and the sum of the squared model coefficients. The Ridge penalty shrinks the coefficients towards zero but does not set them exactly to zero, which can be beneficial in dealing with multicollinearity among predictors.

1. The [cox proportional hazards regression model](https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html) from the <mark style="background-color: #CCECFF"><b>lifelines.CoxPHFitter</b></mark> Python library API was implemented. 
2. The model implementation used 2 hyperparameters:
    * <span style="color: #FF0000">penalizer</span> = penalty to the size of the coefficients during regression fixed at a value = 0.10
    * <span style="color: #FF0000">l1_ratio</span> = proportion of the L1 versus L2 penalty fixed at a value = 0.00
3. All 17 variables were used for prediction given the non-zero values of the model coefficients.
4. Out of all 17 predictors, only 3 variables were statistically significant:
    * <span style="color: #FF0000">Age</span>
    * <span style="color: #FF0000">Bilirubin</span>
    * <span style="color: #FF0000">Prothrombin</span>
5. The cross-validated model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8099
6. The apparent model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8533
7. The independent test model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8675
8. Considerable difference in the apparent and cross-validated model performance observed, indicative of the presence of moderate model overfitting.
9. Survival probability curves obtained from the groups generated by dichotomizing the risk scores demonstrated sufficient differentiation across the entire duration.
10. Hazard and survival probability estimations for 5 sampled cases demonstrated reasonably smooth profiles.


```python
##################################
# Formulating the Cox Regression model
# with Full L2 Penalty and generating the summary
##################################
cirrhosis_survival_coxph_L1_0_L2_100 = CoxPHFitter(penalizer=0.10, l1_ratio=0.00)
cirrhosis_survival_coxph_L1_0_L2_100.fit(cirrhosis_survival_train_modeling, duration_col='N_Days', event_col='Status')
cirrhosis_survival_coxph_L1_0_L2_100.print_summary()
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
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.CoxPHFitter</td>
    </tr>
    <tr>
      <th>duration col</th>
      <td>'N_Days'</td>
    </tr>
    <tr>
      <th>event col</th>
      <td>'Status'</td>
    </tr>
    <tr>
      <th>penalizer</th>
      <td>0.1</td>
    </tr>
    <tr>
      <th>l1 ratio</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>baseline estimation</th>
      <td>breslow</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>218</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>87</td>
    </tr>
    <tr>
      <th>partial log-likelihood</th>
      <td>-360.56</td>
    </tr>
    <tr>
      <th>time fit was run</th>
      <td>2024-07-17 11:42:37 UTC</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">exp(coef)</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">exp(coef) lower 95%</th>
      <th style="min-width: 12px;">exp(coef) upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>0.27</td>
      <td>1.31</td>
      <td>0.11</td>
      <td>0.06</td>
      <td>0.49</td>
      <td>1.06</td>
      <td>1.63</td>
      <td>0.00</td>
      <td>2.48</td>
      <td>0.01</td>
      <td>6.25</td>
    </tr>
    <tr>
      <th>Bilirubin</th>
      <td>0.49</td>
      <td>1.63</td>
      <td>0.14</td>
      <td>0.22</td>
      <td>0.77</td>
      <td>1.24</td>
      <td>2.15</td>
      <td>0.00</td>
      <td>3.50</td>
      <td>&lt;0.005</td>
      <td>11.08</td>
    </tr>
    <tr>
      <th>Cholesterol</th>
      <td>0.08</td>
      <td>1.09</td>
      <td>0.12</td>
      <td>-0.16</td>
      <td>0.32</td>
      <td>0.86</td>
      <td>1.38</td>
      <td>0.00</td>
      <td>0.68</td>
      <td>0.50</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>Albumin</th>
      <td>-0.14</td>
      <td>0.87</td>
      <td>0.12</td>
      <td>-0.38</td>
      <td>0.09</td>
      <td>0.69</td>
      <td>1.09</td>
      <td>0.00</td>
      <td>-1.20</td>
      <td>0.23</td>
      <td>2.13</td>
    </tr>
    <tr>
      <th>Copper</th>
      <td>0.16</td>
      <td>1.18</td>
      <td>0.12</td>
      <td>-0.07</td>
      <td>0.40</td>
      <td>0.93</td>
      <td>1.49</td>
      <td>0.00</td>
      <td>1.34</td>
      <td>0.18</td>
      <td>2.48</td>
    </tr>
    <tr>
      <th>Alk_Phos</th>
      <td>0.04</td>
      <td>1.04</td>
      <td>0.12</td>
      <td>-0.19</td>
      <td>0.27</td>
      <td>0.83</td>
      <td>1.30</td>
      <td>0.00</td>
      <td>0.34</td>
      <td>0.73</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>SGOT</th>
      <td>0.17</td>
      <td>1.19</td>
      <td>0.12</td>
      <td>-0.06</td>
      <td>0.40</td>
      <td>0.94</td>
      <td>1.49</td>
      <td>0.00</td>
      <td>1.47</td>
      <td>0.14</td>
      <td>2.81</td>
    </tr>
    <tr>
      <th>Tryglicerides</th>
      <td>0.10</td>
      <td>1.11</td>
      <td>0.11</td>
      <td>-0.11</td>
      <td>0.32</td>
      <td>0.89</td>
      <td>1.37</td>
      <td>0.00</td>
      <td>0.92</td>
      <td>0.36</td>
      <td>1.49</td>
    </tr>
    <tr>
      <th>Platelets</th>
      <td>-0.07</td>
      <td>0.93</td>
      <td>0.11</td>
      <td>-0.27</td>
      <td>0.14</td>
      <td>0.76</td>
      <td>1.15</td>
      <td>0.00</td>
      <td>-0.64</td>
      <td>0.52</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>Prothrombin</th>
      <td>0.29</td>
      <td>1.34</td>
      <td>0.11</td>
      <td>0.07</td>
      <td>0.52</td>
      <td>1.07</td>
      <td>1.67</td>
      <td>0.00</td>
      <td>2.54</td>
      <td>0.01</td>
      <td>6.51</td>
    </tr>
    <tr>
      <th>Drug</th>
      <td>-0.17</td>
      <td>0.85</td>
      <td>0.21</td>
      <td>-0.57</td>
      <td>0.24</td>
      <td>0.56</td>
      <td>1.28</td>
      <td>0.00</td>
      <td>-0.79</td>
      <td>0.43</td>
      <td>1.23</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>0.01</td>
      <td>1.01</td>
      <td>0.29</td>
      <td>-0.55</td>
      <td>0.58</td>
      <td>0.57</td>
      <td>1.79</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.96</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>Ascites</th>
      <td>0.23</td>
      <td>1.26</td>
      <td>0.35</td>
      <td>-0.46</td>
      <td>0.91</td>
      <td>0.63</td>
      <td>2.49</td>
      <td>0.00</td>
      <td>0.66</td>
      <td>0.51</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>Hepatomegaly</th>
      <td>0.14</td>
      <td>1.15</td>
      <td>0.23</td>
      <td>-0.31</td>
      <td>0.58</td>
      <td>0.73</td>
      <td>1.79</td>
      <td>0.00</td>
      <td>0.60</td>
      <td>0.55</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>Spiders</th>
      <td>0.03</td>
      <td>1.03</td>
      <td>0.24</td>
      <td>-0.43</td>
      <td>0.50</td>
      <td>0.65</td>
      <td>1.65</td>
      <td>0.00</td>
      <td>0.14</td>
      <td>0.89</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>0.51</td>
      <td>1.67</td>
      <td>0.27</td>
      <td>-0.01</td>
      <td>1.04</td>
      <td>0.99</td>
      <td>2.82</td>
      <td>0.00</td>
      <td>1.91</td>
      <td>0.06</td>
      <td>4.16</td>
    </tr>
    <tr>
      <th>Stage_4.0</th>
      <td>0.24</td>
      <td>1.27</td>
      <td>0.24</td>
      <td>-0.23</td>
      <td>0.71</td>
      <td>0.79</td>
      <td>2.04</td>
      <td>0.00</td>
      <td>0.99</td>
      <td>0.32</td>
      <td>1.63</td>
    </tr>
  </tbody>
</table><br><div>
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
  <tbody>
    <tr>
      <th>Concordance</th>
      <td>0.85</td>
    </tr>
    <tr>
      <th>Partial AIC</th>
      <td>755.12</td>
    </tr>
    <tr>
      <th>log-likelihood ratio test</th>
      <td>113.61 on 17 df</td>
    </tr>
    <tr>
      <th>-log2(p) of ll-ratio test</th>
      <td>51.82</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Consolidating the detailed values
# of the model coefficients
##################################
cirrhosis_survival_coxph_L1_0_L2_100.summary
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
      <th>coef</th>
      <th>exp(coef)</th>
      <th>se(coef)</th>
      <th>coef lower 95%</th>
      <th>coef upper 95%</th>
      <th>exp(coef) lower 95%</th>
      <th>exp(coef) upper 95%</th>
      <th>cmp to</th>
      <th>z</th>
      <th>p</th>
      <th>-log2(p)</th>
    </tr>
    <tr>
      <th>covariate</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>0.273503</td>
      <td>1.314562</td>
      <td>0.110305</td>
      <td>0.057310</td>
      <td>0.489697</td>
      <td>1.058984</td>
      <td>1.631822</td>
      <td>0.0</td>
      <td>2.479520</td>
      <td>0.013156</td>
      <td>6.248144</td>
    </tr>
    <tr>
      <th>Bilirubin</th>
      <td>0.491176</td>
      <td>1.634238</td>
      <td>0.140256</td>
      <td>0.216279</td>
      <td>0.766073</td>
      <td>1.241449</td>
      <td>2.151303</td>
      <td>0.0</td>
      <td>3.501995</td>
      <td>0.000462</td>
      <td>11.080482</td>
    </tr>
    <tr>
      <th>Cholesterol</th>
      <td>0.081915</td>
      <td>1.085364</td>
      <td>0.120941</td>
      <td>-0.155125</td>
      <td>0.318956</td>
      <td>0.856308</td>
      <td>1.375691</td>
      <td>0.0</td>
      <td>0.677314</td>
      <td>0.498207</td>
      <td>1.005184</td>
    </tr>
    <tr>
      <th>Albumin</th>
      <td>-0.143698</td>
      <td>0.866150</td>
      <td>0.119365</td>
      <td>-0.377650</td>
      <td>0.090254</td>
      <td>0.685471</td>
      <td>1.094452</td>
      <td>0.0</td>
      <td>-1.203847</td>
      <td>0.228649</td>
      <td>2.128796</td>
    </tr>
    <tr>
      <th>Copper</th>
      <td>0.163286</td>
      <td>1.177373</td>
      <td>0.121420</td>
      <td>-0.074692</td>
      <td>0.401264</td>
      <td>0.928029</td>
      <td>1.493712</td>
      <td>0.0</td>
      <td>1.344807</td>
      <td>0.178687</td>
      <td>2.484490</td>
    </tr>
    <tr>
      <th>Alk_Phos</th>
      <td>0.039447</td>
      <td>1.040236</td>
      <td>0.115607</td>
      <td>-0.187139</td>
      <td>0.266034</td>
      <td>0.829328</td>
      <td>1.304779</td>
      <td>0.0</td>
      <td>0.341217</td>
      <td>0.732941</td>
      <td>0.448232</td>
    </tr>
    <tr>
      <th>SGOT</th>
      <td>0.171530</td>
      <td>1.187119</td>
      <td>0.116855</td>
      <td>-0.057501</td>
      <td>0.400561</td>
      <td>0.944121</td>
      <td>1.492661</td>
      <td>0.0</td>
      <td>1.467888</td>
      <td>0.142135</td>
      <td>2.814669</td>
    </tr>
    <tr>
      <th>Tryglicerides</th>
      <td>0.100959</td>
      <td>1.106231</td>
      <td>0.109604</td>
      <td>-0.113861</td>
      <td>0.315779</td>
      <td>0.892382</td>
      <td>1.371327</td>
      <td>0.0</td>
      <td>0.921122</td>
      <td>0.356987</td>
      <td>1.486058</td>
    </tr>
    <tr>
      <th>Platelets</th>
      <td>-0.067680</td>
      <td>0.934559</td>
      <td>0.105398</td>
      <td>-0.274257</td>
      <td>0.138897</td>
      <td>0.760137</td>
      <td>1.149006</td>
      <td>0.0</td>
      <td>-0.642136</td>
      <td>0.520785</td>
      <td>0.941240</td>
    </tr>
    <tr>
      <th>Prothrombin</th>
      <td>0.290927</td>
      <td>1.337667</td>
      <td>0.114357</td>
      <td>0.066792</td>
      <td>0.515062</td>
      <td>1.069073</td>
      <td>1.673742</td>
      <td>0.0</td>
      <td>2.544032</td>
      <td>0.010958</td>
      <td>6.511858</td>
    </tr>
    <tr>
      <th>Drug</th>
      <td>-0.165487</td>
      <td>0.847480</td>
      <td>0.208410</td>
      <td>-0.573964</td>
      <td>0.242989</td>
      <td>0.563288</td>
      <td>1.275055</td>
      <td>0.0</td>
      <td>-0.794046</td>
      <td>0.427168</td>
      <td>1.227123</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>0.013801</td>
      <td>1.013897</td>
      <td>0.289483</td>
      <td>-0.553576</td>
      <td>0.581178</td>
      <td>0.574890</td>
      <td>1.788144</td>
      <td>0.0</td>
      <td>0.047674</td>
      <td>0.961976</td>
      <td>0.055928</td>
    </tr>
    <tr>
      <th>Ascites</th>
      <td>0.229115</td>
      <td>1.257486</td>
      <td>0.349057</td>
      <td>-0.455025</td>
      <td>0.913254</td>
      <td>0.634432</td>
      <td>2.492420</td>
      <td>0.0</td>
      <td>0.656382</td>
      <td>0.511579</td>
      <td>0.966972</td>
    </tr>
    <tr>
      <th>Hepatomegaly</th>
      <td>0.136962</td>
      <td>1.146785</td>
      <td>0.227782</td>
      <td>-0.309482</td>
      <td>0.583406</td>
      <td>0.733827</td>
      <td>1.792132</td>
      <td>0.0</td>
      <td>0.601287</td>
      <td>0.547649</td>
      <td>0.868676</td>
    </tr>
    <tr>
      <th>Spiders</th>
      <td>0.032709</td>
      <td>1.033250</td>
      <td>0.238344</td>
      <td>-0.434436</td>
      <td>0.499855</td>
      <td>0.647630</td>
      <td>1.648482</td>
      <td>0.0</td>
      <td>0.137235</td>
      <td>0.890845</td>
      <td>0.166753</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>0.512456</td>
      <td>1.669387</td>
      <td>0.268173</td>
      <td>-0.013154</td>
      <td>1.038067</td>
      <td>0.986932</td>
      <td>2.823753</td>
      <td>0.0</td>
      <td>1.910914</td>
      <td>0.056016</td>
      <td>4.158026</td>
    </tr>
    <tr>
      <th>Stage_4.0</th>
      <td>0.238038</td>
      <td>1.268758</td>
      <td>0.241160</td>
      <td>-0.234626</td>
      <td>0.710703</td>
      <td>0.790866</td>
      <td>2.035421</td>
      <td>0.0</td>
      <td>0.987056</td>
      <td>0.323615</td>
      <td>1.627650</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting the hazard ratio of the
# formulated Cox Regression model
# with Full L2 Penalty
##################################
cirrhosis_survival_coxph_L1_0_L2_100_summary = cirrhosis_survival_coxph_L1_0_L2_100.summary
cirrhosis_survival_coxph_L1_0_L2_100_summary['hazard_ratio'] = np.exp(cirrhosis_survival_coxph_L1_0_L2_100_summary['coef'])
significant = cirrhosis_survival_coxph_L1_0_L2_100_summary['p'] < 0.05
plt.figure(figsize=(17, 8))
colors = ['#993300' if sig else '#CC9966' for sig in significant]

plt.barh(cirrhosis_survival_coxph_L1_0_L2_100_summary.index, 
         cirrhosis_survival_coxph_L1_0_L2_100_summary['hazard_ratio'], 
         xerr=cirrhosis_survival_coxph_L1_0_L2_100_summary['se(coef)'], 
         color=colors)
plt.xlabel('Hazard Ratio')
plt.ylabel('Variables')
plt.title('COXPH_FL2P Hazard Ratio Forest Plot')
plt.axvline(x=1, color='k', linestyle='--')
plt.gca().invert_yaxis()
plt.show()
```


    
![png](output_193_0.png)
    



```python
##################################
# Plotting the coefficient magnitude
# of the formulated Cox Regression model
# with Full L2 Penalty
##################################
plt.figure(figsize=(17, 8))
colors = ['#993300' if sig else '#CC9966' for sig in significant]
cirrhosis_survival_coxph_L1_0_L2_100_summary['coef'].plot(kind='barh', color=colors)
plt.xlabel('Variables')
plt.ylabel('Model Coefficient Value')
plt.title('COXPH_FL2P Model Coefficients')
plt.xticks(rotation=0, ha='right')
plt.xlim(-1,1)
plt.gca().invert_yaxis()
plt.show()
```


    
![png](output_194_0.png)
    



```python
##################################
# Gathering the apparent model performance
# as baseline for evaluating overfitting
##################################
cirrhosis_survival_coxph_L1_0_L2_100.fit(cirrhosis_survival_train_modeling, duration_col='N_Days', event_col='Status')
train_predictions = cirrhosis_survival_coxph_L1_0_L2_100.predict_partial_hazard(cirrhosis_survival_train_modeling)
cirrhosis_survival_coxph_L1_0_L2_100_train_ci = concordance_index(cirrhosis_survival_train_modeling['N_Days'], 
                                                                     -train_predictions, 
                                                                     cirrhosis_survival_train_modeling['Status'])
display(f"Apparent Concordance Index: {cirrhosis_survival_coxph_L1_0_L2_100_train_ci}")
```


    'Apparent Concordance Index: 0.8533810143042913'



```python
##################################
# Performing 5-Fold Cross-Validation
# on the training data
##################################
kf = KFold(n_splits=5, shuffle=True, random_state=88888888)
c_index_scores = []

for train_index, val_index in kf.split(cirrhosis_survival_train_modeling):
    df_train_fold = cirrhosis_survival_train_modeling.iloc[train_index]
    df_val_fold = cirrhosis_survival_train_modeling.iloc[val_index]
    
    cirrhosis_survival_coxph_L1_0_L2_100.fit(df_train_fold, duration_col='N_Days', event_col='Status')
    val_predictions = cirrhosis_survival_coxph_L1_0_L2_100.predict_partial_hazard(df_val_fold)
    c_index = concordance_index(df_val_fold['N_Days'], -val_predictions, df_val_fold['Status'])
    c_index_scores.append(c_index)

cirrhosis_survival_coxph_L1_0_L2_100_cv_ci_mean = np.mean(c_index_scores)
cirrhosis_survival_coxph_L1_0_L2_100_cv_ci_std = np.std(c_index_scores)

display(f"Cross-Validated Concordance Index: {cirrhosis_survival_coxph_L1_0_L2_100_cv_ci_mean}")
```


    'Cross-Validated Concordance Index: 0.8099834493754214'



```python
##################################
# Evaluating the model performance
# on test data
##################################
test_predictions = cirrhosis_survival_coxph_L1_0_L2_100.predict_partial_hazard(cirrhosis_survival_test_modeling)
cirrhosis_survival_coxph_L1_0_L2_100_test_ci = concordance_index(cirrhosis_survival_test_modeling['N_Days'], 
                                                                     -test_predictions, 
                                                                     cirrhosis_survival_test_modeling['Status'])
display(f"Test Concordance Index: {cirrhosis_survival_coxph_L1_0_L2_100_test_ci}")
```


    'Test Concordance Index: 0.8675736961451247'



```python
##################################
# Gathering the concordance indices
# from training, cross-validation and test
##################################
coxph_L1_0_L2_100_set = pd.DataFrame(["Train","Cross-Validation","Test"])
coxph_L1_0_L2_100_ci_values = pd.DataFrame([cirrhosis_survival_coxph_L1_0_L2_100_train_ci,
                                           cirrhosis_survival_coxph_L1_0_L2_100_cv_ci_mean,
                                           cirrhosis_survival_coxph_L1_0_L2_100_test_ci])
coxph_L1_0_L2_100_method = pd.DataFrame(["COXPH_FL2P"]*3)
coxph_L1_0_L2_100_summary = pd.concat([coxph_L1_0_L2_100_set, 
                                     coxph_L1_0_L2_100_ci_values,
                                     coxph_L1_0_L2_100_method], 
                                    axis=1)
coxph_L1_0_L2_100_summary.columns = ['Set', 'Concordance.Index', 'Method']
coxph_L1_0_L2_100_summary.reset_index(inplace=True, drop=True)
display(coxph_L1_0_L2_100_summary)
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
      <th>Set</th>
      <th>Concordance.Index</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Train</td>
      <td>0.853381</td>
      <td>COXPH_FL2P</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cross-Validation</td>
      <td>0.809983</td>
      <td>COXPH_FL2P</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Test</td>
      <td>0.867574</td>
      <td>COXPH_FL2P</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Binning the predicted risks
# into dichotomous groups and
# exploring the relationships with
# survival event and duration
##################################
cirrhosis_survival_test_modeling['Predicted_Risks_COXPH_FL2P'] = test_predictions
cirrhosis_survival_test_modeling['Predicted_RiskGroups_COXPH_FL2P'] = risk_groups = pd.qcut(cirrhosis_survival_test_modeling['Predicted_Risks_COXPH_FL2P'], 2, labels=['Low-Risk', 'High-Risk'])

plt.figure(figsize=(17, 8))
for group in risk_groups.unique():
    group_data = cirrhosis_survival_test_modeling[risk_groups == group]
    kmf.fit(group_data['N_Days'], event_observed=group_data['Status'], label=group)
    kmf.plot_survival_function()

plt.title('COXPH_FL2P Survival Probabilities by Predicted Risk Groups')
plt.xlabel('N_Days')
plt.ylabel('Event Survival Probability')
plt.show()
```


    
![png](output_199_0.png)
    



```python
##################################
# Gathering the predictor information
# for 5 test case samples
##################################
test_case_details = cirrhosis_survival_test_modeling.iloc[[10, 20, 30, 40, 50]]
display(test_case_details)
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
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_4.0</th>
      <th>Predicted_Risks_COXPH_NP</th>
      <th>Predicted_RiskGroups_COXPH_NP</th>
      <th>Predicted_Risks_COXPH_FL1P</th>
      <th>Predicted_RiskGroups_COXPH_FL1P</th>
      <th>Predicted_Risks_COXPH_FL2P</th>
      <th>Predicted_RiskGroups_COXPH_FL2P</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>True</td>
      <td>1827</td>
      <td>0.226982</td>
      <td>1.530100</td>
      <td>1.302295</td>
      <td>1.331981</td>
      <td>1.916467</td>
      <td>-0.477846</td>
      <td>-0.451305</td>
      <td>2.250260</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>8.397958</td>
      <td>High-Risk</td>
      <td>3.515780</td>
      <td>High-Risk</td>
      <td>4.748349</td>
      <td>High-Risk</td>
    </tr>
    <tr>
      <th>20</th>
      <td>False</td>
      <td>1447</td>
      <td>-0.147646</td>
      <td>0.061189</td>
      <td>0.793618</td>
      <td>-1.158235</td>
      <td>0.861264</td>
      <td>0.625621</td>
      <td>0.319035</td>
      <td>0.446026</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.466233</td>
      <td>Low-Risk</td>
      <td>0.769723</td>
      <td>Low-Risk</td>
      <td>0.681968</td>
      <td>Low-Risk</td>
    </tr>
    <tr>
      <th>30</th>
      <td>False</td>
      <td>2574</td>
      <td>0.296370</td>
      <td>-1.283677</td>
      <td>0.169685</td>
      <td>3.237777</td>
      <td>-1.008276</td>
      <td>-0.873566</td>
      <td>-0.845549</td>
      <td>-0.351236</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.153047</td>
      <td>Low-Risk</td>
      <td>0.320200</td>
      <td>Low-Risk</td>
      <td>0.165574</td>
      <td>Low-Risk</td>
    </tr>
    <tr>
      <th>40</th>
      <td>True</td>
      <td>3762</td>
      <td>0.392609</td>
      <td>-0.096645</td>
      <td>-0.486337</td>
      <td>1.903146</td>
      <td>-0.546292</td>
      <td>-0.247141</td>
      <td>-0.720619</td>
      <td>-0.810790</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2.224881</td>
      <td>High-Risk</td>
      <td>1.167777</td>
      <td>High-Risk</td>
      <td>1.591530</td>
      <td>High-Risk</td>
    </tr>
    <tr>
      <th>50</th>
      <td>False</td>
      <td>837</td>
      <td>-0.813646</td>
      <td>1.089037</td>
      <td>0.064451</td>
      <td>0.212865</td>
      <td>2.063138</td>
      <td>-0.224432</td>
      <td>0.074987</td>
      <td>2.333282</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2.886051</td>
      <td>High-Risk</td>
      <td>1.832490</td>
      <td>High-Risk</td>
      <td>2.541904</td>
      <td>High-Risk</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



```python
##################################
# Gathering the risk-groups
# for 5 test case samples
##################################
print(cirrhosis_survival_test_modeling.loc[[10, 20, 30, 40, 50]][['Predicted_RiskGroups_COXPH_FL2P']])
```

       Predicted_RiskGroups_COXPH_FL2P
    10                       High-Risk
    20                        Low-Risk
    30                        Low-Risk
    40                       High-Risk
    50                       High-Risk
    


```python
##################################
# Estimating the cumulative hazard
# and survival functions
# for 5 test cases
##################################
test_case = cirrhosis_survival_test_modeling.iloc[[10, 20, 30, 40, 50]]
test_case_labels = ['Patient_10','Patient_20','Patient_30','Patient_40','Patient_50']

fig, axes = plt.subplots(1, 2, figsize=(17, 8))
for i, (index, row) in enumerate(test_case.iterrows()):
    survival_function = cirrhosis_survival_coxph_L1_0_L2_100.predict_survival_function(row.to_frame().T)
    axes[0].plot(survival_function, label=f'Sample {i+1}')
axes[0].set_title('COXPH_FL2P Survival Function for 5 Test Cases')
axes[0].set_xlabel('N_Days')
axes[0].set_ylim(0,1)
axes[0].set_ylabel('Survival Probability')
axes[0].legend(test_case_labels, loc="lower left")
for i, (index, row) in enumerate(test_case.iterrows()):
    hazard_function = cirrhosis_survival_coxph_L1_0_L2_100.predict_cumulative_hazard(row.to_frame().T)
    axes[1].plot(hazard_function, label=f'Sample {i+1}')
axes[1].set_title('COXPH_FL2P Cumulative Hazard for 5 Test Cases')
axes[1].set_xlabel('N_Days')
axes[1].set_ylim(0,5)
axes[1].set_ylabel('Cumulative Hazard')
axes[1].legend(test_case_labels, loc="upper left")
plt.tight_layout()
plt.show()
```


    
![png](output_202_0.png)
    



```python
##################################
# Creating the explainer object
##################################
cirrhosis_survival_coxph_L1_0_L2_100_explainer = shap.Explainer(cirrhosis_survival_coxph_L1_0_L2_100.predict_partial_hazard, 
                                                    cirrhosis_survival_train_modeling.drop(columns=["N_Days", "Status"]))
cirrhosis_survival_coxph_L1_0_L2_100_shap_values = cirrhosis_survival_coxph_L1_0_L2_100_explainer(cirrhosis_survival_train_modeling.drop(columns=["N_Days", "Status"]))
```

    PermutationExplainer explainer: 219it [00:21,  5.38it/s]                         
    


```python
##################################
# Plotting the SHAP summary plot
##################################
shap.summary_plot(cirrhosis_survival_coxph_L1_0_L2_100_shap_values, 
                  cirrhosis_survival_train_modeling.drop(columns=["N_Days", "Status"]),
                  sort=False)
```


    
![png](output_204_0.png)
    


### 1.6.5 Cox Regression With Equal L1|L2 Penalty <a class="anchor" id="1.6.5"></a>

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Survival Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) refer to statistical methods used to analyze survival data, accounting for censored observations. These models aim to describe the relationship between survival time and one or more predictor variables, and to estimate the survival function and hazard function. Survival models are essential for understanding the factors that influence time-to-event data, allowing for predictions and comparisons between different groups or treatment effects. They are widely used in clinical trials, reliability engineering, and other areas where time-to-event data is prevalent.

[Cox Proportional Hazards Regression](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1972.tb00899.x) is a semiparametric model used to study the relationship between the survival time of subjects and one or more predictor variables. The model assumes that the hazard ratio (the risk of the event occurring at a specific time) is a product of a baseline hazard function and an exponential function of the predictor variables. It also does not require the baseline hazard to be specified, thus making it a semiparametric model. As a method, it is well-established and widely used in survival analysis, can handle time-dependent covariates and provides a relatively straightforward interpretation. However, the process assumes proportional hazards, which may not hold in all datasets, and may be less flexible in capturing complex relationships between variables and survival times compared to some machine learning models. Given a dataset with survival times, event indicators, and predictor variables, the algorithm involves defining the partial likelihood function for the Cox model (which only considers the relative ordering of survival times); using optimization techniques to estimate the regression coefficients by maximizing the log-partial likelihood; estimating the baseline hazard function (although it is not explicitly required for predictions); and calculating the hazard function and survival function for new data using the estimated coefficients and baseline hazard.

[Elastic Net Penalty](https://lifelines.readthedocs.io/en/latest/), or combined L1 and L2 regularization in cox regression, adds a constraint to both the sum of the absolute and squared values of the coefficients. The penalized log-likelihood function is composed of the partial likelihood of the cox model, a tuning parameter that controls the strength of both lasso and ridge penalties, the sum of the absolute model coefficients, and the sum of the squared model coefficients. The Elastic Net penalty combines the benefits of both Lasso and Ridge, promoting sparsity while also dealing with multicollinearity.

1. The [cox proportional hazards regression model](https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html) from the <mark style="background-color: #CCECFF"><b>lifelines.CoxPHFitter</b></mark> Python library API was implemented. 
2. The model implementation used 2 hyperparameters:
    * <span style="color: #FF0000">penalizer</span> = penalty to the size of the coefficients during regression fixed at a value = 0.10
    * <span style="color: #FF0000">l1_ratio</span> = proportion of the L1 versus L2 penalty fixed at a value = 0.50
3. Only 10 out of the 17 variables were used for prediction given the non-zero values of the model coefficients.
4. Out of all 10 predictors, only 2 variables were statistically significant:
    * <span style="color: #FF0000">Bilirubin</span>
    * <span style="color: #FF0000">Prothrombin</span>
5. The cross-validated model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8162
6. The apparent model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8465
7. The independent test model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8630
8. Considerable difference in the apparent and cross-validated model performance observed, indicative of the presence of minimal model overfitting.
9. Survival probability curves obtained from the groups generated by dichotomizing the risk scores demonstrated sufficient differentiation across the entire duration.
10. Hazard and survival probability estimations for 5 sampled cases demonstrated reasonably smooth profiles.


```python
##################################
# Formulating the Cox Regression model
# with Equal L1 and l2 Penalty
# and generating the summary
##################################
cirrhosis_survival_coxph_L1_50_L2_50 = CoxPHFitter(penalizer=0.10, l1_ratio=0.50)
cirrhosis_survival_coxph_L1_50_L2_50.fit(cirrhosis_survival_train_modeling, duration_col='N_Days', event_col='Status')
cirrhosis_survival_coxph_L1_50_L2_50.print_summary()
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
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.CoxPHFitter</td>
    </tr>
    <tr>
      <th>duration col</th>
      <td>'N_Days'</td>
    </tr>
    <tr>
      <th>event col</th>
      <td>'Status'</td>
    </tr>
    <tr>
      <th>penalizer</th>
      <td>0.1</td>
    </tr>
    <tr>
      <th>l1 ratio</th>
      <td>0.5</td>
    </tr>
    <tr>
      <th>baseline estimation</th>
      <td>breslow</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>218</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>87</td>
    </tr>
    <tr>
      <th>partial log-likelihood</th>
      <td>-378.64</td>
    </tr>
    <tr>
      <th>time fit was run</th>
      <td>2024-07-17 11:43:03 UTC</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">exp(coef)</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">exp(coef) lower 95%</th>
      <th style="min-width: 12px;">exp(coef) upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>0.17</td>
      <td>1.19</td>
      <td>0.11</td>
      <td>-0.03</td>
      <td>0.38</td>
      <td>0.97</td>
      <td>1.46</td>
      <td>0.00</td>
      <td>1.64</td>
      <td>0.10</td>
      <td>3.30</td>
    </tr>
    <tr>
      <th>Bilirubin</th>
      <td>0.62</td>
      <td>1.86</td>
      <td>0.15</td>
      <td>0.32</td>
      <td>0.92</td>
      <td>1.38</td>
      <td>2.52</td>
      <td>0.00</td>
      <td>4.07</td>
      <td>&lt;0.005</td>
      <td>14.38</td>
    </tr>
    <tr>
      <th>Cholesterol</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Albumin</th>
      <td>-0.11</td>
      <td>0.90</td>
      <td>0.12</td>
      <td>-0.35</td>
      <td>0.14</td>
      <td>0.70</td>
      <td>1.15</td>
      <td>0.00</td>
      <td>-0.86</td>
      <td>0.39</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>Copper</th>
      <td>0.12</td>
      <td>1.13</td>
      <td>0.12</td>
      <td>-0.11</td>
      <td>0.36</td>
      <td>0.89</td>
      <td>1.44</td>
      <td>0.00</td>
      <td>1.03</td>
      <td>0.30</td>
      <td>1.72</td>
    </tr>
    <tr>
      <th>Alk_Phos</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>SGOT</th>
      <td>0.05</td>
      <td>1.06</td>
      <td>0.12</td>
      <td>-0.19</td>
      <td>0.30</td>
      <td>0.83</td>
      <td>1.34</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.66</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>Tryglicerides</th>
      <td>0.05</td>
      <td>1.05</td>
      <td>0.11</td>
      <td>-0.17</td>
      <td>0.26</td>
      <td>0.85</td>
      <td>1.30</td>
      <td>0.00</td>
      <td>0.43</td>
      <td>0.67</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>Platelets</th>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Prothrombin</th>
      <td>0.25</td>
      <td>1.29</td>
      <td>0.12</td>
      <td>0.01</td>
      <td>0.49</td>
      <td>1.01</td>
      <td>1.63</td>
      <td>0.00</td>
      <td>2.07</td>
      <td>0.04</td>
      <td>4.71</td>
    </tr>
    <tr>
      <th>Drug</th>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Ascites</th>
      <td>0.18</td>
      <td>1.20</td>
      <td>0.37</td>
      <td>-0.54</td>
      <td>0.91</td>
      <td>0.58</td>
      <td>2.47</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.62</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>Hepatomegaly</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.24</td>
      <td>-0.47</td>
      <td>0.47</td>
      <td>0.63</td>
      <td>1.60</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Spiders</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>0.36</td>
      <td>1.44</td>
      <td>0.28</td>
      <td>-0.18</td>
      <td>0.91</td>
      <td>0.83</td>
      <td>2.48</td>
      <td>0.00</td>
      <td>1.31</td>
      <td>0.19</td>
      <td>2.39</td>
    </tr>
    <tr>
      <th>Stage_4.0</th>
      <td>0.15</td>
      <td>1.16</td>
      <td>0.26</td>
      <td>-0.35</td>
      <td>0.66</td>
      <td>0.70</td>
      <td>1.93</td>
      <td>0.00</td>
      <td>0.59</td>
      <td>0.56</td>
      <td>0.85</td>
    </tr>
  </tbody>
</table><br><div>
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
  <tbody>
    <tr>
      <th>Concordance</th>
      <td>0.85</td>
    </tr>
    <tr>
      <th>Partial AIC</th>
      <td>791.29</td>
    </tr>
    <tr>
      <th>log-likelihood ratio test</th>
      <td>77.44 on 17 df</td>
    </tr>
    <tr>
      <th>-log2(p) of ll-ratio test</th>
      <td>29.78</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Consolidating the detailed values
# of the model coefficients
##################################
cirrhosis_survival_coxph_L1_50_L2_50.summary
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
      <th>coef</th>
      <th>exp(coef)</th>
      <th>se(coef)</th>
      <th>coef lower 95%</th>
      <th>coef upper 95%</th>
      <th>exp(coef) lower 95%</th>
      <th>exp(coef) upper 95%</th>
      <th>cmp to</th>
      <th>z</th>
      <th>p</th>
      <th>-log2(p)</th>
    </tr>
    <tr>
      <th>covariate</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>1.735119e-01</td>
      <td>1.189475</td>
      <td>0.105893</td>
      <td>-0.034035</td>
      <td>0.381059</td>
      <td>0.966538</td>
      <td>1.463834</td>
      <td>0.0</td>
      <td>1.638554</td>
      <td>0.101306</td>
      <td>3.303207</td>
    </tr>
    <tr>
      <th>Bilirubin</th>
      <td>6.232373e-01</td>
      <td>1.864956</td>
      <td>0.153110</td>
      <td>0.323146</td>
      <td>0.923328</td>
      <td>1.381468</td>
      <td>2.517655</td>
      <td>0.0</td>
      <td>4.070510</td>
      <td>0.000047</td>
      <td>14.379736</td>
    </tr>
    <tr>
      <th>Cholesterol</th>
      <td>1.027031e-07</td>
      <td>1.000000</td>
      <td>0.000158</td>
      <td>-0.000310</td>
      <td>0.000311</td>
      <td>0.999690</td>
      <td>1.000311</td>
      <td>0.0</td>
      <td>0.000648</td>
      <td>0.999483</td>
      <td>0.000746</td>
    </tr>
    <tr>
      <th>Albumin</th>
      <td>-1.062850e-01</td>
      <td>0.899168</td>
      <td>0.124211</td>
      <td>-0.349734</td>
      <td>0.137164</td>
      <td>0.704875</td>
      <td>1.147017</td>
      <td>0.0</td>
      <td>-0.855681</td>
      <td>0.392175</td>
      <td>1.350432</td>
    </tr>
    <tr>
      <th>Copper</th>
      <td>1.247938e-01</td>
      <td>1.132915</td>
      <td>0.121539</td>
      <td>-0.113419</td>
      <td>0.363006</td>
      <td>0.892777</td>
      <td>1.437645</td>
      <td>0.0</td>
      <td>1.026777</td>
      <td>0.304525</td>
      <td>1.715365</td>
    </tr>
    <tr>
      <th>Alk_Phos</th>
      <td>1.833222e-07</td>
      <td>1.000000</td>
      <td>0.000194</td>
      <td>-0.000380</td>
      <td>0.000380</td>
      <td>0.999620</td>
      <td>1.000380</td>
      <td>0.0</td>
      <td>0.000945</td>
      <td>0.999246</td>
      <td>0.001089</td>
    </tr>
    <tr>
      <th>SGOT</th>
      <td>5.418539e-02</td>
      <td>1.055680</td>
      <td>0.123088</td>
      <td>-0.187063</td>
      <td>0.295434</td>
      <td>0.829391</td>
      <td>1.343710</td>
      <td>0.0</td>
      <td>0.440215</td>
      <td>0.659781</td>
      <td>0.599941</td>
    </tr>
    <tr>
      <th>Tryglicerides</th>
      <td>4.676406e-02</td>
      <td>1.047875</td>
      <td>0.108783</td>
      <td>-0.166447</td>
      <td>0.259975</td>
      <td>0.846668</td>
      <td>1.296898</td>
      <td>0.0</td>
      <td>0.429883</td>
      <td>0.667280</td>
      <td>0.583635</td>
    </tr>
    <tr>
      <th>Platelets</th>
      <td>-6.958829e-07</td>
      <td>0.999999</td>
      <td>0.001610</td>
      <td>-0.003157</td>
      <td>0.003155</td>
      <td>0.996848</td>
      <td>1.003160</td>
      <td>0.0</td>
      <td>-0.000432</td>
      <td>0.999655</td>
      <td>0.000498</td>
    </tr>
    <tr>
      <th>Prothrombin</th>
      <td>2.508953e-01</td>
      <td>1.285176</td>
      <td>0.121104</td>
      <td>0.013535</td>
      <td>0.488255</td>
      <td>1.013627</td>
      <td>1.629471</td>
      <td>0.0</td>
      <td>2.071730</td>
      <td>0.038291</td>
      <td>4.706865</td>
    </tr>
    <tr>
      <th>Drug</th>
      <td>-1.917876e-07</td>
      <td>1.000000</td>
      <td>0.000314</td>
      <td>-0.000615</td>
      <td>0.000614</td>
      <td>0.999385</td>
      <td>1.000615</td>
      <td>0.0</td>
      <td>-0.000612</td>
      <td>0.999512</td>
      <td>0.000704</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>-3.698238e-07</td>
      <td>1.000000</td>
      <td>0.000495</td>
      <td>-0.000970</td>
      <td>0.000969</td>
      <td>0.999030</td>
      <td>1.000970</td>
      <td>0.0</td>
      <td>-0.000748</td>
      <td>0.999404</td>
      <td>0.000861</td>
    </tr>
    <tr>
      <th>Ascites</th>
      <td>1.846251e-01</td>
      <td>1.202767</td>
      <td>0.367758</td>
      <td>-0.536167</td>
      <td>0.905418</td>
      <td>0.584986</td>
      <td>2.472964</td>
      <td>0.0</td>
      <td>0.502029</td>
      <td>0.615647</td>
      <td>0.699824</td>
    </tr>
    <tr>
      <th>Hepatomegaly</th>
      <td>5.622110e-04</td>
      <td>1.000562</td>
      <td>0.238476</td>
      <td>-0.466841</td>
      <td>0.467966</td>
      <td>0.626980</td>
      <td>1.596743</td>
      <td>0.0</td>
      <td>0.002358</td>
      <td>0.998119</td>
      <td>0.002716</td>
    </tr>
    <tr>
      <th>Spiders</th>
      <td>3.890060e-07</td>
      <td>1.000000</td>
      <td>0.000424</td>
      <td>-0.000831</td>
      <td>0.000832</td>
      <td>0.999169</td>
      <td>1.000832</td>
      <td>0.0</td>
      <td>0.000917</td>
      <td>0.999268</td>
      <td>0.001056</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>3.639376e-01</td>
      <td>1.438984</td>
      <td>0.278544</td>
      <td>-0.182000</td>
      <td>0.909875</td>
      <td>0.833602</td>
      <td>2.484011</td>
      <td>0.0</td>
      <td>1.306569</td>
      <td>0.191359</td>
      <td>2.385645</td>
    </tr>
    <tr>
      <th>Stage_4.0</th>
      <td>1.520894e-01</td>
      <td>1.164264</td>
      <td>0.258242</td>
      <td>-0.354055</td>
      <td>0.658234</td>
      <td>0.701836</td>
      <td>1.931379</td>
      <td>0.0</td>
      <td>0.588942</td>
      <td>0.555900</td>
      <td>0.847102</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting the hazard ratio of the
# formulated Cox Regression model
# with Equal L1 and l2 Penalty
##################################
cirrhosis_survival_coxph_L1_50_L2_50_summary = cirrhosis_survival_coxph_L1_50_L2_50.summary
cirrhosis_survival_coxph_L1_50_L2_50_summary['hazard_ratio'] = np.exp(cirrhosis_survival_coxph_L1_50_L2_50_summary['coef'])
significant = cirrhosis_survival_coxph_L1_50_L2_50_summary['p'] < 0.05
plt.figure(figsize=(17, 8))
colors = ['#993300' if sig else '#CC9966' for sig in significant]

plt.barh(cirrhosis_survival_coxph_L1_50_L2_50_summary.index, 
         cirrhosis_survival_coxph_L1_50_L2_50_summary['hazard_ratio'], 
         xerr=cirrhosis_survival_coxph_L1_50_L2_50_summary['se(coef)'], 
         color=colors)
plt.xlabel('Hazard Ratio')
plt.ylabel('Variables')
plt.title('COXPH_EL1L2P Hazard Ratio Forest Plot')
plt.axvline(x=1, color='k', linestyle='--')
plt.gca().invert_yaxis()
plt.show()
```


    
![png](output_208_0.png)
    



```python
##################################
# Plotting the coefficient magnitude
# of the formulated Cox Regression model
# with Equal L1 and l2 Penalty
##################################
plt.figure(figsize=(17, 8))
colors = ['#993300' if sig else '#CC9966' for sig in significant]
cirrhosis_survival_coxph_L1_50_L2_50_summary['coef'].plot(kind='barh', color=colors)
plt.xlabel('Variables')
plt.ylabel('Model Coefficient Value')
plt.title('COXPH_EL1L2P Model Coefficients')
plt.xticks(rotation=0, ha='right')
plt.xlim(-1,1)
plt.gca().invert_yaxis()
plt.show()
```


    
![png](output_209_0.png)
    



```python
##################################
# Gathering the apparent model performance
# as baseline for evaluating overfitting
##################################
cirrhosis_survival_coxph_L1_50_L2_50.fit(cirrhosis_survival_train_modeling, duration_col='N_Days', event_col='Status')
train_predictions = cirrhosis_survival_coxph_L1_50_L2_50.predict_partial_hazard(cirrhosis_survival_train_modeling)
cirrhosis_survival_coxph_L1_50_L2_50_train_ci = concordance_index(cirrhosis_survival_train_modeling['N_Days'], 
                                                                     -train_predictions, 
                                                                     cirrhosis_survival_train_modeling['Status'])
display(f"Apparent Concordance Index: {cirrhosis_survival_coxph_L1_50_L2_50_train_ci}")
```


    'Apparent Concordance Index: 0.846553966189857'



```python
##################################
# Performing 5-Fold Cross-Validation
# on the training data
##################################
kf = KFold(n_splits=5, shuffle=True, random_state=88888888)
c_index_scores = []

for train_index, val_index in kf.split(cirrhosis_survival_train_modeling):
    df_train_fold = cirrhosis_survival_train_modeling.iloc[train_index]
    df_val_fold = cirrhosis_survival_train_modeling.iloc[val_index]
    
    cirrhosis_survival_coxph_L1_50_L2_50.fit(df_train_fold, duration_col='N_Days', event_col='Status')
    val_predictions = cirrhosis_survival_coxph_L1_50_L2_50.predict_partial_hazard(df_val_fold)
    c_index = concordance_index(df_val_fold['N_Days'], -val_predictions, df_val_fold['Status'])
    c_index_scores.append(c_index)

cirrhosis_survival_coxph_L1_50_L2_50_cv_ci_mean = np.mean(c_index_scores)
cirrhosis_survival_coxph_L1_50_L2_50_cv_ci_std = np.std(c_index_scores)

display(f"Cross-Validated Concordance Index: {cirrhosis_survival_coxph_L1_50_L2_50_cv_ci_mean}")
```


    'Cross-Validated Concordance Index: 0.8162802125265027'



```python
##################################
# Evaluating the model performance
# on test data
##################################
test_predictions = cirrhosis_survival_coxph_L1_50_L2_50.predict_partial_hazard(cirrhosis_survival_test_modeling)
cirrhosis_survival_coxph_L1_50_L2_50_test_ci = concordance_index(cirrhosis_survival_test_modeling['N_Days'], 
                                                                     -test_predictions, 
                                                                     cirrhosis_survival_test_modeling['Status'])
display(f"Test Concordance Index: {cirrhosis_survival_coxph_L1_50_L2_50_test_ci}")
```


    'Test Concordance Index: 0.8630385487528345'



```python
##################################
# Gathering the concordance indices
# from training, cross-validation and test
##################################
coxph_L1_50_L2_50_set = pd.DataFrame(["Train","Cross-Validation","Test"])
coxph_L1_50_L2_50_ci_values = pd.DataFrame([cirrhosis_survival_coxph_L1_50_L2_50_train_ci,
                                           cirrhosis_survival_coxph_L1_50_L2_50_cv_ci_mean,
                                           cirrhosis_survival_coxph_L1_50_L2_50_test_ci])
coxph_L1_50_L2_50_method = pd.DataFrame(["COXPH_EL1L2P"]*3)
coxph_L1_50_L2_50_summary = pd.concat([coxph_L1_50_L2_50_set, 
                                     coxph_L1_50_L2_50_ci_values,
                                     coxph_L1_50_L2_50_method], 
                                    axis=1)
coxph_L1_50_L2_50_summary.columns = ['Set', 'Concordance.Index', 'Method']
coxph_L1_50_L2_50_summary.reset_index(inplace=True, drop=True)
display(coxph_L1_50_L2_50_summary)
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
      <th>Set</th>
      <th>Concordance.Index</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Train</td>
      <td>0.846554</td>
      <td>COXPH_EL1L2P</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cross-Validation</td>
      <td>0.816280</td>
      <td>COXPH_EL1L2P</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Test</td>
      <td>0.863039</td>
      <td>COXPH_EL1L2P</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Binning the predicted risks
# into dichotomous groups and
# exploring the relationships with
# survival event and duration
##################################
cirrhosis_survival_test_modeling['Predicted_Risks_COXPH_EL1L2P'] = test_predictions
cirrhosis_survival_test_modeling['Predicted_RiskGroups_COXPH_EL1L2P'] = risk_groups = pd.qcut(cirrhosis_survival_test_modeling['Predicted_Risks_COXPH_EL1L2P'], 2, labels=['Low-Risk', 'High-Risk'])

plt.figure(figsize=(17, 8))
for group in risk_groups.unique():
    group_data = cirrhosis_survival_test_modeling[risk_groups == group]
    kmf.fit(group_data['N_Days'], event_observed=group_data['Status'], label=group)
    kmf.plot_survival_function()

plt.title('COXPH_EL1L2P Survival Probabilities by Predicted Risk Groups')
plt.xlabel('N_Days')
plt.ylabel('Event Survival Probability')
plt.show()
```


    
![png](output_214_0.png)
    



```python
##################################
# Gathering the predictor information
# for 5 test case samples
##################################
test_case_details = cirrhosis_survival_test_modeling.iloc[[10, 20, 30, 40, 50]]
display(test_case_details)
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
      <th>Edema</th>
      <th>Stage_4.0</th>
      <th>Predicted_Risks_COXPH_NP</th>
      <th>Predicted_RiskGroups_COXPH_NP</th>
      <th>Predicted_Risks_COXPH_FL1P</th>
      <th>Predicted_RiskGroups_COXPH_FL1P</th>
      <th>Predicted_Risks_COXPH_FL2P</th>
      <th>Predicted_RiskGroups_COXPH_FL2P</th>
      <th>Predicted_Risks_COXPH_EL1L2P</th>
      <th>Predicted_RiskGroups_COXPH_EL1L2P</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>True</td>
      <td>1827</td>
      <td>0.226982</td>
      <td>1.530100</td>
      <td>1.302295</td>
      <td>1.331981</td>
      <td>1.916467</td>
      <td>-0.477846</td>
      <td>-0.451305</td>
      <td>2.250260</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>8.397958</td>
      <td>High-Risk</td>
      <td>3.515780</td>
      <td>High-Risk</td>
      <td>4.748349</td>
      <td>High-Risk</td>
      <td>4.442277</td>
      <td>High-Risk</td>
    </tr>
    <tr>
      <th>20</th>
      <td>False</td>
      <td>1447</td>
      <td>-0.147646</td>
      <td>0.061189</td>
      <td>0.793618</td>
      <td>-1.158235</td>
      <td>0.861264</td>
      <td>0.625621</td>
      <td>0.319035</td>
      <td>0.446026</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0.466233</td>
      <td>Low-Risk</td>
      <td>0.769723</td>
      <td>Low-Risk</td>
      <td>0.681968</td>
      <td>Low-Risk</td>
      <td>0.741409</td>
      <td>Low-Risk</td>
    </tr>
    <tr>
      <th>30</th>
      <td>False</td>
      <td>2574</td>
      <td>0.296370</td>
      <td>-1.283677</td>
      <td>0.169685</td>
      <td>3.237777</td>
      <td>-1.008276</td>
      <td>-0.873566</td>
      <td>-0.845549</td>
      <td>-0.351236</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0.153047</td>
      <td>Low-Risk</td>
      <td>0.320200</td>
      <td>Low-Risk</td>
      <td>0.165574</td>
      <td>Low-Risk</td>
      <td>0.255343</td>
      <td>Low-Risk</td>
    </tr>
    <tr>
      <th>40</th>
      <td>True</td>
      <td>3762</td>
      <td>0.392609</td>
      <td>-0.096645</td>
      <td>-0.486337</td>
      <td>1.903146</td>
      <td>-0.546292</td>
      <td>-0.247141</td>
      <td>-0.720619</td>
      <td>-0.810790</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>2.224881</td>
      <td>High-Risk</td>
      <td>1.167777</td>
      <td>High-Risk</td>
      <td>1.591530</td>
      <td>High-Risk</td>
      <td>1.275445</td>
      <td>High-Risk</td>
    </tr>
    <tr>
      <th>50</th>
      <td>False</td>
      <td>837</td>
      <td>-0.813646</td>
      <td>1.089037</td>
      <td>0.064451</td>
      <td>0.212865</td>
      <td>2.063138</td>
      <td>-0.224432</td>
      <td>0.074987</td>
      <td>2.333282</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>2.886051</td>
      <td>High-Risk</td>
      <td>1.832490</td>
      <td>High-Risk</td>
      <td>2.541904</td>
      <td>High-Risk</td>
      <td>2.125042</td>
      <td>High-Risk</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



```python
##################################
# Gathering the risk-groups
# for 5 test case samples
##################################
print(cirrhosis_survival_test_modeling.loc[[10, 20, 30, 40, 50]][['Predicted_RiskGroups_COXPH_EL1L2P']])
```

       Predicted_RiskGroups_COXPH_EL1L2P
    10                         High-Risk
    20                          Low-Risk
    30                          Low-Risk
    40                         High-Risk
    50                         High-Risk
    


```python
##################################
# Estimating the cumulative hazard
# and survival functions
# for 5 test cases
##################################
test_case = cirrhosis_survival_test_modeling.iloc[[10, 20, 30, 40, 50]]
test_case_labels = ['Patient_10','Patient_20','Patient_30','Patient_40','Patient_50']

fig, axes = plt.subplots(1, 2, figsize=(17, 8))
for i, (index, row) in enumerate(test_case.iterrows()):
    survival_function = cirrhosis_survival_coxph_L1_50_L2_50.predict_survival_function(row.to_frame().T)
    axes[0].plot(survival_function, label=f'Sample {i+1}')
axes[0].set_title('COXPH_EL1L2P Survival Function for 5 Test Cases')
axes[0].set_xlabel('N_Days')
axes[0].set_ylim(0,1)
axes[0].set_ylabel('Survival Probability')
axes[0].legend(test_case_labels, loc="lower left")
for i, (index, row) in enumerate(test_case.iterrows()):
    hazard_function = cirrhosis_survival_coxph_L1_50_L2_50.predict_cumulative_hazard(row.to_frame().T)
    axes[1].plot(hazard_function, label=f'Sample {i+1}')
axes[1].set_title('COXPH_EL1L2P Cumulative Hazard for 5 Test Cases')
axes[1].set_xlabel('N_Days')
axes[1].set_ylim(0,5)
axes[1].set_ylabel('Cumulative Hazard')
axes[1].legend(test_case_labels, loc="upper left")
plt.tight_layout()
plt.show()
```


    
![png](output_217_0.png)
    



```python
##################################
# Creating the explainer object
##################################
cirrhosis_survival_coxph_L1_50_L2_50_explainer = shap.Explainer(cirrhosis_survival_coxph_L1_50_L2_50.predict_partial_hazard, 
                                                    cirrhosis_survival_train_modeling.drop(columns=["N_Days", "Status"]))
cirrhosis_survival_coxph_L1_50_L2_50_shap_values = cirrhosis_survival_coxph_L1_50_L2_50_explainer(cirrhosis_survival_train_modeling.drop(columns=["N_Days", "Status"]))
```

    PermutationExplainer explainer: 219it [00:22,  5.35it/s]                         
    


```python
##################################
# Plotting the SHAP summary plot
##################################
shap.summary_plot(cirrhosis_survival_coxph_L1_50_L2_50_shap_values, 
                  cirrhosis_survival_train_modeling.drop(columns=["N_Days", "Status"]),
                  sort=False)
```


    
![png](output_219_0.png)
    


### 1.6.6 Cox Regression With Predominantly L1-Weighted|L2 Penalty <a class="anchor" id="1.6.6"></a>

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Survival Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) refer to statistical methods used to analyze survival data, accounting for censored observations. These models aim to describe the relationship between survival time and one or more predictor variables, and to estimate the survival function and hazard function. Survival models are essential for understanding the factors that influence time-to-event data, allowing for predictions and comparisons between different groups or treatment effects. They are widely used in clinical trials, reliability engineering, and other areas where time-to-event data is prevalent.

[Cox Proportional Hazards Regression](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1972.tb00899.x) is a semiparametric model used to study the relationship between the survival time of subjects and one or more predictor variables. The model assumes that the hazard ratio (the risk of the event occurring at a specific time) is a product of a baseline hazard function and an exponential function of the predictor variables. It also does not require the baseline hazard to be specified, thus making it a semiparametric model. As a method, it is well-established and widely used in survival analysis, can handle time-dependent covariates and provides a relatively straightforward interpretation. However, the process assumes proportional hazards, which may not hold in all datasets, and may be less flexible in capturing complex relationships between variables and survival times compared to some machine learning models. Given a dataset with survival times, event indicators, and predictor variables, the algorithm involves defining the partial likelihood function for the Cox model (which only considers the relative ordering of survival times); using optimization techniques to estimate the regression coefficients by maximizing the log-partial likelihood; estimating the baseline hazard function (although it is not explicitly required for predictions); and calculating the hazard function and survival function for new data using the estimated coefficients and baseline hazard.

[Elastic Net Penalty](https://lifelines.readthedocs.io/en/latest/), or combined L1 and L2 regularization in cox regression, adds a constraint to both the sum of the absolute and squared values of the coefficients. The penalized log-likelihood function is composed of the partial likelihood of the cox model, a tuning parameter that controls the strength of both lasso and ridge penalties, the sum of the absolute model coefficients, and the sum of the squared model coefficients. The Elastic Net penalty combines the benefits of both Lasso and Ridge, promoting sparsity while also dealing with multicollinearity.

1. The [cox proportional hazards regression model](https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html) from the <mark style="background-color: #CCECFF"><b>lifelines.CoxPHFitter</b></mark> Python library API was implemented. 
2. The model implementation used 2 hyperparameters:
    * <span style="color: #FF0000">penalizer</span> = penalty to the size of the coefficients during regression fixed at a value = 0.10
    * <span style="color: #FF0000">l1_ratio</span> = proportion of the L1 versus L2 penalty fixed at a value = 0.75
3. Only 8 out of the 17 variables were used for prediction given the non-zero values of the model coefficients.
4. Out of all 8 predictors, only 2 variables were statistically significant:
    * <span style="color: #FF0000">Bilirubin</span>
5. The cross-validated model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8101
6. The apparent model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8394
7. The independent test model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8526
8. Considerable difference in the apparent and cross-validated model performance observed, indicative of the presence of minimal model overfitting.
9. Survival probability curves obtained from the groups generated by dichotomizing the risk scores demonstrated sufficient differentiation across the entire duration.
10. Hazard and survival probability estimations for 5 sampled cases demonstrated reasonably smooth profiles.


```python
##################################
# Formulating the Cox Regression model
# with Predominantly L1-Weighted and L2 Penalty
# and generating the summary
##################################
cirrhosis_survival_coxph_L1_75_L2_25 = CoxPHFitter(penalizer=0.10, l1_ratio=0.75)
cirrhosis_survival_coxph_L1_75_L2_25.fit(cirrhosis_survival_train_modeling, duration_col='N_Days', event_col='Status')
cirrhosis_survival_coxph_L1_75_L2_25.print_summary()
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
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.CoxPHFitter</td>
    </tr>
    <tr>
      <th>duration col</th>
      <td>'N_Days'</td>
    </tr>
    <tr>
      <th>event col</th>
      <td>'Status'</td>
    </tr>
    <tr>
      <th>penalizer</th>
      <td>0.1</td>
    </tr>
    <tr>
      <th>l1 ratio</th>
      <td>0.75</td>
    </tr>
    <tr>
      <th>baseline estimation</th>
      <td>breslow</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>218</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>87</td>
    </tr>
    <tr>
      <th>partial log-likelihood</th>
      <td>-385.37</td>
    </tr>
    <tr>
      <th>time fit was run</th>
      <td>2024-07-17 11:43:32 UTC</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">exp(coef)</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">exp(coef) lower 95%</th>
      <th style="min-width: 12px;">exp(coef) upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>0.13</td>
      <td>1.14</td>
      <td>0.11</td>
      <td>-0.08</td>
      <td>0.35</td>
      <td>0.93</td>
      <td>1.42</td>
      <td>0.00</td>
      <td>1.24</td>
      <td>0.21</td>
      <td>2.23</td>
    </tr>
    <tr>
      <th>Bilirubin</th>
      <td>0.68</td>
      <td>1.98</td>
      <td>0.15</td>
      <td>0.39</td>
      <td>0.97</td>
      <td>1.48</td>
      <td>2.64</td>
      <td>0.00</td>
      <td>4.64</td>
      <td>&lt;0.005</td>
      <td>18.16</td>
    </tr>
    <tr>
      <th>Cholesterol</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Albumin</th>
      <td>-0.09</td>
      <td>0.92</td>
      <td>0.13</td>
      <td>-0.34</td>
      <td>0.17</td>
      <td>0.71</td>
      <td>1.18</td>
      <td>0.00</td>
      <td>-0.66</td>
      <td>0.51</td>
      <td>0.98</td>
    </tr>
    <tr>
      <th>Copper</th>
      <td>0.09</td>
      <td>1.09</td>
      <td>0.13</td>
      <td>-0.16</td>
      <td>0.34</td>
      <td>0.85</td>
      <td>1.40</td>
      <td>0.00</td>
      <td>0.69</td>
      <td>0.49</td>
      <td>1.04</td>
    </tr>
    <tr>
      <th>Alk_Phos</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>SGOT</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Tryglicerides</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.11</td>
      <td>-0.22</td>
      <td>0.22</td>
      <td>0.80</td>
      <td>1.25</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Platelets</th>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Prothrombin</th>
      <td>0.23</td>
      <td>1.25</td>
      <td>0.13</td>
      <td>-0.02</td>
      <td>0.47</td>
      <td>0.98</td>
      <td>1.60</td>
      <td>0.00</td>
      <td>1.78</td>
      <td>0.07</td>
      <td>3.75</td>
    </tr>
    <tr>
      <th>Drug</th>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Ascites</th>
      <td>0.19</td>
      <td>1.21</td>
      <td>0.39</td>
      <td>-0.57</td>
      <td>0.95</td>
      <td>0.57</td>
      <td>2.60</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.62</td>
      <td>0.69</td>
    </tr>
    <tr>
      <th>Hepatomegaly</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Spiders</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>0.27</td>
      <td>1.31</td>
      <td>0.29</td>
      <td>-0.31</td>
      <td>0.84</td>
      <td>0.74</td>
      <td>2.33</td>
      <td>0.00</td>
      <td>0.91</td>
      <td>0.36</td>
      <td>1.47</td>
    </tr>
    <tr>
      <th>Stage_4.0</th>
      <td>0.09</td>
      <td>1.09</td>
      <td>0.26</td>
      <td>-0.42</td>
      <td>0.59</td>
      <td>0.66</td>
      <td>1.80</td>
      <td>0.00</td>
      <td>0.34</td>
      <td>0.74</td>
      <td>0.44</td>
    </tr>
  </tbody>
</table><br><div>
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
  <tbody>
    <tr>
      <th>Concordance</th>
      <td>0.84</td>
    </tr>
    <tr>
      <th>Partial AIC</th>
      <td>804.74</td>
    </tr>
    <tr>
      <th>log-likelihood ratio test</th>
      <td>63.99 on 17 df</td>
    </tr>
    <tr>
      <th>-log2(p) of ll-ratio test</th>
      <td>22.07</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Consolidating the detailed values
# of the model coefficients
##################################
cirrhosis_survival_coxph_L1_75_L2_25.summary
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
      <th>coef</th>
      <th>exp(coef)</th>
      <th>se(coef)</th>
      <th>coef lower 95%</th>
      <th>coef upper 95%</th>
      <th>exp(coef) lower 95%</th>
      <th>exp(coef) upper 95%</th>
      <th>cmp to</th>
      <th>z</th>
      <th>p</th>
      <th>-log2(p)</th>
    </tr>
    <tr>
      <th>covariate</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>1.349698e-01</td>
      <td>1.144502</td>
      <td>0.108439</td>
      <td>-0.077568</td>
      <td>0.347507</td>
      <td>0.925364</td>
      <td>1.415535</td>
      <td>0.0</td>
      <td>1.244656</td>
      <td>0.213258</td>
      <td>2.229327</td>
    </tr>
    <tr>
      <th>Bilirubin</th>
      <td>6.831940e-01</td>
      <td>1.980192</td>
      <td>0.147122</td>
      <td>0.394841</td>
      <td>0.971547</td>
      <td>1.484148</td>
      <td>2.642029</td>
      <td>0.0</td>
      <td>4.643737</td>
      <td>0.000003</td>
      <td>18.156883</td>
    </tr>
    <tr>
      <th>Cholesterol</th>
      <td>2.576816e-08</td>
      <td>1.000000</td>
      <td>0.000081</td>
      <td>-0.000159</td>
      <td>0.000159</td>
      <td>0.999841</td>
      <td>1.000159</td>
      <td>0.0</td>
      <td>0.000317</td>
      <td>0.999747</td>
      <td>0.000365</td>
    </tr>
    <tr>
      <th>Albumin</th>
      <td>-8.508496e-02</td>
      <td>0.918434</td>
      <td>0.128576</td>
      <td>-0.337088</td>
      <td>0.166918</td>
      <td>0.713846</td>
      <td>1.181658</td>
      <td>0.0</td>
      <td>-0.661751</td>
      <td>0.508131</td>
      <td>0.976728</td>
    </tr>
    <tr>
      <th>Copper</th>
      <td>8.805786e-02</td>
      <td>1.092051</td>
      <td>0.126767</td>
      <td>-0.160401</td>
      <td>0.336516</td>
      <td>0.851802</td>
      <td>1.400062</td>
      <td>0.0</td>
      <td>0.694644</td>
      <td>0.487278</td>
      <td>1.037182</td>
    </tr>
    <tr>
      <th>Alk_Phos</th>
      <td>6.930976e-08</td>
      <td>1.000000</td>
      <td>0.000098</td>
      <td>-0.000192</td>
      <td>0.000192</td>
      <td>0.999808</td>
      <td>1.000192</td>
      <td>0.0</td>
      <td>0.000709</td>
      <td>0.999435</td>
      <td>0.000816</td>
    </tr>
    <tr>
      <th>SGOT</th>
      <td>1.623334e-07</td>
      <td>1.000000</td>
      <td>0.000202</td>
      <td>-0.000396</td>
      <td>0.000396</td>
      <td>0.999604</td>
      <td>1.000397</td>
      <td>0.0</td>
      <td>0.000803</td>
      <td>0.999359</td>
      <td>0.000925</td>
    </tr>
    <tr>
      <th>Tryglicerides</th>
      <td>1.163402e-04</td>
      <td>1.000116</td>
      <td>0.112144</td>
      <td>-0.219682</td>
      <td>0.219914</td>
      <td>0.802774</td>
      <td>1.245970</td>
      <td>0.0</td>
      <td>0.001037</td>
      <td>0.999172</td>
      <td>0.001195</td>
    </tr>
    <tr>
      <th>Platelets</th>
      <td>-1.524115e-07</td>
      <td>1.000000</td>
      <td>0.000185</td>
      <td>-0.000363</td>
      <td>0.000362</td>
      <td>0.999637</td>
      <td>1.000362</td>
      <td>0.0</td>
      <td>-0.000824</td>
      <td>0.999343</td>
      <td>0.000949</td>
    </tr>
    <tr>
      <th>Prothrombin</th>
      <td>2.251074e-01</td>
      <td>1.252457</td>
      <td>0.126126</td>
      <td>-0.022095</td>
      <td>0.472310</td>
      <td>0.978147</td>
      <td>1.603694</td>
      <td>0.0</td>
      <td>1.784781</td>
      <td>0.074297</td>
      <td>3.750555</td>
    </tr>
    <tr>
      <th>Drug</th>
      <td>-3.710982e-08</td>
      <td>1.000000</td>
      <td>0.000161</td>
      <td>-0.000315</td>
      <td>0.000315</td>
      <td>0.999685</td>
      <td>1.000315</td>
      <td>0.0</td>
      <td>-0.000231</td>
      <td>0.999816</td>
      <td>0.000266</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>-1.881487e-07</td>
      <td>1.000000</td>
      <td>0.000282</td>
      <td>-0.000553</td>
      <td>0.000553</td>
      <td>0.999447</td>
      <td>1.000553</td>
      <td>0.0</td>
      <td>-0.000667</td>
      <td>0.999468</td>
      <td>0.000768</td>
    </tr>
    <tr>
      <th>Ascites</th>
      <td>1.933495e-01</td>
      <td>1.213307</td>
      <td>0.388559</td>
      <td>-0.568213</td>
      <td>0.954912</td>
      <td>0.566537</td>
      <td>2.598442</td>
      <td>0.0</td>
      <td>0.497606</td>
      <td>0.618762</td>
      <td>0.692544</td>
    </tr>
    <tr>
      <th>Hepatomegaly</th>
      <td>3.022681e-07</td>
      <td>1.000000</td>
      <td>0.000366</td>
      <td>-0.000716</td>
      <td>0.000717</td>
      <td>0.999284</td>
      <td>1.000717</td>
      <td>0.0</td>
      <td>0.000827</td>
      <td>0.999340</td>
      <td>0.000952</td>
    </tr>
    <tr>
      <th>Spiders</th>
      <td>1.424028e-07</td>
      <td>1.000000</td>
      <td>0.000214</td>
      <td>-0.000418</td>
      <td>0.000419</td>
      <td>0.999582</td>
      <td>1.000419</td>
      <td>0.0</td>
      <td>0.000667</td>
      <td>0.999468</td>
      <td>0.000768</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>2.686481e-01</td>
      <td>1.308195</td>
      <td>0.293772</td>
      <td>-0.307134</td>
      <td>0.844430</td>
      <td>0.735552</td>
      <td>2.326651</td>
      <td>0.0</td>
      <td>0.914479</td>
      <td>0.360465</td>
      <td>1.472069</td>
    </tr>
    <tr>
      <th>Stage_4.0</th>
      <td>8.598185e-02</td>
      <td>1.089787</td>
      <td>0.255667</td>
      <td>-0.415116</td>
      <td>0.587080</td>
      <td>0.660264</td>
      <td>1.798728</td>
      <td>0.0</td>
      <td>0.336304</td>
      <td>0.736642</td>
      <td>0.440965</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting the hazard ratio of the
# formulated Cox Regression model
# with Predominantly L1-Weighted and L2 Penalty
##################################
cirrhosis_survival_coxph_L1_75_L2_25_summary = cirrhosis_survival_coxph_L1_75_L2_25.summary
cirrhosis_survival_coxph_L1_75_L2_25_summary['hazard_ratio'] = np.exp(cirrhosis_survival_coxph_L1_75_L2_25_summary['coef'])
significant = cirrhosis_survival_coxph_L1_75_L2_25_summary['p'] < 0.05
plt.figure(figsize=(17, 8))
colors = ['#993300' if sig else '#CC9966' for sig in significant]

plt.barh(cirrhosis_survival_coxph_L1_75_L2_25_summary.index, 
         cirrhosis_survival_coxph_L1_75_L2_25_summary['hazard_ratio'], 
         xerr=cirrhosis_survival_coxph_L1_75_L2_25_summary['se(coef)'], 
         color=colors)
plt.xlabel('Hazard Ratio')
plt.ylabel('Variables')
plt.title('COXPH_PWL1L2P Hazard Ratio Forest Plot')
plt.axvline(x=1, color='k', linestyle='--')
plt.gca().invert_yaxis()
plt.show()
```


    
![png](output_223_0.png)
    



```python
##################################
# Plotting the coefficient magnitude
# of the formulated Cox Regression model
# with Predominantly L1-Weighted and L2 Penalty
##################################
plt.figure(figsize=(17, 8))
colors = ['#993300' if sig else '#CC9966' for sig in significant]
cirrhosis_survival_coxph_L1_75_L2_25_summary['coef'].plot(kind='barh', color=colors)
plt.xlabel('Variables')
plt.ylabel('Model Coefficient Value')
plt.title('COXPH_PWL1L2P Model Coefficients')
plt.xticks(rotation=0, ha='right')
plt.xlim(-1,1)
plt.gca().invert_yaxis()
plt.show()
```


    
![png](output_224_0.png)
    



```python
##################################
# Gathering the apparent model performance
# as baseline for evaluating overfitting
##################################
cirrhosis_survival_coxph_L1_75_L2_25.fit(cirrhosis_survival_train_modeling, duration_col='N_Days', event_col='Status')
train_predictions = cirrhosis_survival_coxph_L1_75_L2_25.predict_partial_hazard(cirrhosis_survival_train_modeling)
cirrhosis_survival_coxph_L1_75_L2_25_train_ci = concordance_index(cirrhosis_survival_train_modeling['N_Days'], 
                                                                     -train_predictions, 
                                                                     cirrhosis_survival_train_modeling['Status'])
display(f"Apparent Concordance Index: {cirrhosis_survival_coxph_L1_75_L2_25_train_ci}")
```


    'Apparent Concordance Index: 0.8394018205461639'



```python
##################################
# Performing 5-Fold Cross-Validation
# on the training data
##################################
kf = KFold(n_splits=5, shuffle=True, random_state=88888888)
c_index_scores = []

for train_index, val_index in kf.split(cirrhosis_survival_train_modeling):
    df_train_fold = cirrhosis_survival_train_modeling.iloc[train_index]
    df_val_fold = cirrhosis_survival_train_modeling.iloc[val_index]
    
    cirrhosis_survival_coxph_L1_75_L2_25.fit(df_train_fold, duration_col='N_Days', event_col='Status')
    val_predictions = cirrhosis_survival_coxph_L1_75_L2_25.predict_partial_hazard(df_val_fold)
    c_index = concordance_index(df_val_fold['N_Days'], -val_predictions, df_val_fold['Status'])
    c_index_scores.append(c_index)

cirrhosis_survival_coxph_L1_75_L2_25_cv_ci_mean = np.mean(c_index_scores)
cirrhosis_survival_coxph_L1_75_L2_25_cv_ci_std = np.std(c_index_scores)

display(f"Cross-Validated Concordance Index: {cirrhosis_survival_coxph_L1_75_L2_25_cv_ci_mean}")
```


    'Cross-Validated Concordance Index: 0.810177644183905'



```python
##################################
# Evaluating the model performance
# on test data
##################################
test_predictions = cirrhosis_survival_coxph_L1_75_L2_25.predict_partial_hazard(cirrhosis_survival_test_modeling)
cirrhosis_survival_coxph_L1_75_L2_25_test_ci = concordance_index(cirrhosis_survival_test_modeling['N_Days'], 
                                                                     -test_predictions, 
                                                                     cirrhosis_survival_test_modeling['Status'])
display(f"Test Concordance Index: {cirrhosis_survival_coxph_L1_75_L2_25_test_ci}")
```


    'Test Concordance Index: 0.8526077097505669'



```python
##################################
# Gathering the concordance indices
# from training, cross-validation and test
##################################
coxph_L1_75_L2_25_set = pd.DataFrame(["Train","Cross-Validation","Test"])
coxph_L1_75_L2_25_ci_values = pd.DataFrame([cirrhosis_survival_coxph_L1_75_L2_25_train_ci,
                                           cirrhosis_survival_coxph_L1_75_L2_25_cv_ci_mean,
                                           cirrhosis_survival_coxph_L1_75_L2_25_test_ci])
coxph_L1_75_L2_25_method = pd.DataFrame(["COXPH_PWL1L2P"]*3)
coxph_L1_75_L2_25_summary = pd.concat([coxph_L1_75_L2_25_set, 
                                     coxph_L1_75_L2_25_ci_values,
                                     coxph_L1_75_L2_25_method], 
                                    axis=1)
coxph_L1_75_L2_25_summary.columns = ['Set', 'Concordance.Index', 'Method']
coxph_L1_75_L2_25_summary.reset_index(inplace=True, drop=True)
display(coxph_L1_75_L2_25_summary)
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
      <th>Set</th>
      <th>Concordance.Index</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Train</td>
      <td>0.839402</td>
      <td>COXPH_PWL1L2P</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cross-Validation</td>
      <td>0.810178</td>
      <td>COXPH_PWL1L2P</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Test</td>
      <td>0.852608</td>
      <td>COXPH_PWL1L2P</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Binning the predicted risks
# into dichotomous groups and
# exploring the relationships with
# survival event and duration
##################################
cirrhosis_survival_test_modeling['Predicted_Risks_COXPH_PWL1L2P'] = test_predictions
cirrhosis_survival_test_modeling['Predicted_RiskGroups_COXPH_PWL1L2P'] = risk_groups = pd.qcut(cirrhosis_survival_test_modeling['Predicted_Risks_COXPH_PWL1L2P'], 2, labels=['Low-Risk', 'High-Risk'])

plt.figure(figsize=(17, 8))
for group in risk_groups.unique():
    group_data = cirrhosis_survival_test_modeling[risk_groups == group]
    kmf.fit(group_data['N_Days'], event_observed=group_data['Status'], label=group)
    kmf.plot_survival_function()

plt.title('COXPH_PWL1L2P Survival Probabilities by Predicted Risk Groups')
plt.xlabel('N_Days')
plt.ylabel('Event Survival Probability')
plt.show()
```


    
![png](output_229_0.png)
    



```python
##################################
# Gathering the predictor information
# for 5 test case samples
##################################
test_case_details = cirrhosis_survival_test_modeling.iloc[[10, 20, 30, 40, 50]]
display(test_case_details)
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
      <th>Predicted_Risks_COXPH_NP</th>
      <th>Predicted_RiskGroups_COXPH_NP</th>
      <th>Predicted_Risks_COXPH_FL1P</th>
      <th>Predicted_RiskGroups_COXPH_FL1P</th>
      <th>Predicted_Risks_COXPH_FL2P</th>
      <th>Predicted_RiskGroups_COXPH_FL2P</th>
      <th>Predicted_Risks_COXPH_EL1L2P</th>
      <th>Predicted_RiskGroups_COXPH_EL1L2P</th>
      <th>Predicted_Risks_COXPH_PWL1L2P</th>
      <th>Predicted_RiskGroups_COXPH_PWL1L2P</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>True</td>
      <td>1827</td>
      <td>0.226982</td>
      <td>1.530100</td>
      <td>1.302295</td>
      <td>1.331981</td>
      <td>1.916467</td>
      <td>-0.477846</td>
      <td>-0.451305</td>
      <td>2.250260</td>
      <td>...</td>
      <td>8.397958</td>
      <td>High-Risk</td>
      <td>3.515780</td>
      <td>High-Risk</td>
      <td>4.748349</td>
      <td>High-Risk</td>
      <td>4.442277</td>
      <td>High-Risk</td>
      <td>3.761430</td>
      <td>High-Risk</td>
    </tr>
    <tr>
      <th>20</th>
      <td>False</td>
      <td>1447</td>
      <td>-0.147646</td>
      <td>0.061189</td>
      <td>0.793618</td>
      <td>-1.158235</td>
      <td>0.861264</td>
      <td>0.625621</td>
      <td>0.319035</td>
      <td>0.446026</td>
      <td>...</td>
      <td>0.466233</td>
      <td>Low-Risk</td>
      <td>0.769723</td>
      <td>Low-Risk</td>
      <td>0.681968</td>
      <td>Low-Risk</td>
      <td>0.741409</td>
      <td>Low-Risk</td>
      <td>0.747044</td>
      <td>Low-Risk</td>
    </tr>
    <tr>
      <th>30</th>
      <td>False</td>
      <td>2574</td>
      <td>0.296370</td>
      <td>-1.283677</td>
      <td>0.169685</td>
      <td>3.237777</td>
      <td>-1.008276</td>
      <td>-0.873566</td>
      <td>-0.845549</td>
      <td>-0.351236</td>
      <td>...</td>
      <td>0.153047</td>
      <td>Low-Risk</td>
      <td>0.320200</td>
      <td>Low-Risk</td>
      <td>0.165574</td>
      <td>Low-Risk</td>
      <td>0.255343</td>
      <td>Low-Risk</td>
      <td>0.295159</td>
      <td>Low-Risk</td>
    </tr>
    <tr>
      <th>40</th>
      <td>True</td>
      <td>3762</td>
      <td>0.392609</td>
      <td>-0.096645</td>
      <td>-0.486337</td>
      <td>1.903146</td>
      <td>-0.546292</td>
      <td>-0.247141</td>
      <td>-0.720619</td>
      <td>-0.810790</td>
      <td>...</td>
      <td>2.224881</td>
      <td>High-Risk</td>
      <td>1.167777</td>
      <td>High-Risk</td>
      <td>1.591530</td>
      <td>High-Risk</td>
      <td>1.275445</td>
      <td>High-Risk</td>
      <td>1.214395</td>
      <td>High-Risk</td>
    </tr>
    <tr>
      <th>50</th>
      <td>False</td>
      <td>837</td>
      <td>-0.813646</td>
      <td>1.089037</td>
      <td>0.064451</td>
      <td>0.212865</td>
      <td>2.063138</td>
      <td>-0.224432</td>
      <td>0.074987</td>
      <td>2.333282</td>
      <td>...</td>
      <td>2.886051</td>
      <td>High-Risk</td>
      <td>1.832490</td>
      <td>High-Risk</td>
      <td>2.541904</td>
      <td>High-Risk</td>
      <td>2.125042</td>
      <td>High-Risk</td>
      <td>1.832724</td>
      <td>High-Risk</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>



```python
##################################
# Gathering the risk-groups
# for 5 test case samples
##################################
print(cirrhosis_survival_test_modeling.loc[[10, 20, 30, 40, 50]][['Predicted_RiskGroups_COXPH_PWL1L2P']])
```

       Predicted_RiskGroups_COXPH_PWL1L2P
    10                          High-Risk
    20                           Low-Risk
    30                           Low-Risk
    40                          High-Risk
    50                          High-Risk
    


```python
##################################
# Estimating the cumulative hazard
# and survival functions
# for 5 test cases
##################################
test_case = cirrhosis_survival_test_modeling.iloc[[10, 20, 30, 40, 50]]
test_case_labels = ['Patient_10','Patient_20','Patient_30','Patient_40','Patient_50']

fig, axes = plt.subplots(1, 2, figsize=(17, 8))
for i, (index, row) in enumerate(test_case.iterrows()):
    survival_function = cirrhosis_survival_coxph_L1_75_L2_25.predict_survival_function(row.to_frame().T)
    axes[0].plot(survival_function, label=f'Sample {i+1}')
axes[0].set_title('COXPH_PWL1L2P Survival Function for 5 Test Cases')
axes[0].set_xlabel('N_Days')
axes[0].set_ylim(0,1)
axes[0].set_ylabel('Survival Probability')
axes[0].legend(test_case_labels, loc="lower left")
for i, (index, row) in enumerate(test_case.iterrows()):
    hazard_function = cirrhosis_survival_coxph_L1_75_L2_25.predict_cumulative_hazard(row.to_frame().T)
    axes[1].plot(hazard_function, label=f'Sample {i+1}')
axes[1].set_title('COXPH_PWL1L2P Cumulative Hazard for 5 Test Cases')
axes[1].set_xlabel('N_Days')
axes[1].set_ylim(0,5)
axes[1].set_ylabel('Cumulative Hazard')
axes[1].legend(test_case_labels, loc="upper left")
plt.tight_layout()
plt.show()
```


    
![png](output_232_0.png)
    



```python
##################################
# Creating the explainer object
##################################
cirrhosis_survival_coxph_L1_75_L2_25_explainer = shap.Explainer(cirrhosis_survival_coxph_L1_75_L2_25.predict_partial_hazard, 
                                                    cirrhosis_survival_train_modeling.drop(columns=["N_Days", "Status"]))
cirrhosis_survival_coxph_L1_75_L2_25_shap_values = cirrhosis_survival_coxph_L1_75_L2_25_explainer(cirrhosis_survival_train_modeling.drop(columns=["N_Days", "Status"]))
```

    PermutationExplainer explainer: 219it [00:22,  5.43it/s]                         
    


```python
##################################
# Plotting the SHAP summary plot
##################################
shap.summary_plot(cirrhosis_survival_coxph_L1_75_L2_25_shap_values, 
                  cirrhosis_survival_train_modeling.drop(columns=["N_Days", "Status"]),
                  sort=False)
```


    
![png](output_234_0.png)
    


### 1.6.7 Cox Regression With Predominantly L2-Weighted|L1 Penalty <a class="anchor" id="1.6.7"></a>

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Survival Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) refer to statistical methods used to analyze survival data, accounting for censored observations. These models aim to describe the relationship between survival time and one or more predictor variables, and to estimate the survival function and hazard function. Survival models are essential for understanding the factors that influence time-to-event data, allowing for predictions and comparisons between different groups or treatment effects. They are widely used in clinical trials, reliability engineering, and other areas where time-to-event data is prevalent.

[Cox Proportional Hazards Regression](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1972.tb00899.x) is a semiparametric model used to study the relationship between the survival time of subjects and one or more predictor variables. The model assumes that the hazard ratio (the risk of the event occurring at a specific time) is a product of a baseline hazard function and an exponential function of the predictor variables. It also does not require the baseline hazard to be specified, thus making it a semiparametric model. As a method, it is well-established and widely used in survival analysis, can handle time-dependent covariates and provides a relatively straightforward interpretation. However, the process assumes proportional hazards, which may not hold in all datasets, and may be less flexible in capturing complex relationships between variables and survival times compared to some machine learning models. Given a dataset with survival times, event indicators, and predictor variables, the algorithm involves defining the partial likelihood function for the Cox model (which only considers the relative ordering of survival times); using optimization techniques to estimate the regression coefficients by maximizing the log-partial likelihood; estimating the baseline hazard function (although it is not explicitly required for predictions); and calculating the hazard function and survival function for new data using the estimated coefficients and baseline hazard.

[Elastic Net Penalty](https://lifelines.readthedocs.io/en/latest/), or combined L1 and L2 regularization in cox regression, adds a constraint to both the sum of the absolute and squared values of the coefficients. The penalized log-likelihood function is composed of the partial likelihood of the cox model, a tuning parameter that controls the strength of both lasso and ridge penalties, the sum of the absolute model coefficients, and the sum of the squared model coefficients. The Elastic Net penalty combines the benefits of both Lasso and Ridge, promoting sparsity while also dealing with multicollinearity.

1. The [cox proportional hazards regression model](https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html) from the <mark style="background-color: #CCECFF"><b>lifelines.CoxPHFitter</b></mark> Python library API was implemented. 
2. The model implementation used 2 hyperparameters:
    * <span style="color: #FF0000">penalizer</span> = penalty to the size of the coefficients during regression fixed at a value = 0.10
    * <span style="color: #FF0000">l1_ratio</span> = proportion of the L1 versus L2 penalty fixed at a value = 0.25
3. Only 12 out of the 17 variables were used for prediction given the non-zero values of the model coefficients.
4. Out of all 12 predictors, only 2 variables were statistically significant:
    * <span style="color: #FF0000">Age</span>
    * <span style="color: #FF0000">Bilirubin</span>
    * <span style="color: #FF0000">Prothrombin</span>
5. The cross-validated model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8152
6. The apparent model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8501
7. The independent test model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8671
8. Considerable difference in the apparent and cross-validated model performance observed, indicative of the presence of moderate model overfitting.
9. Survival probability curves obtained from the groups generated by dichotomizing the risk scores demonstrated sufficient differentiation across the entire duration.
10. Hazard and survival probability estimations for 5 sampled cases demonstrated reasonably smooth profiles.


```python
##################################
# Formulating the Cox Regression model
# with Predominantly L2-Weighted and L1 Penalty
# and generating the summary
##################################
cirrhosis_survival_coxph_L1_25_L2_75 = CoxPHFitter(penalizer=0.10, l1_ratio=0.25)
cirrhosis_survival_coxph_L1_25_L2_75.fit(cirrhosis_survival_train_modeling, duration_col='N_Days', event_col='Status')
cirrhosis_survival_coxph_L1_25_L2_75.print_summary()
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
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.CoxPHFitter</td>
    </tr>
    <tr>
      <th>duration col</th>
      <td>'N_Days'</td>
    </tr>
    <tr>
      <th>event col</th>
      <td>'Status'</td>
    </tr>
    <tr>
      <th>penalizer</th>
      <td>0.1</td>
    </tr>
    <tr>
      <th>l1 ratio</th>
      <td>0.25</td>
    </tr>
    <tr>
      <th>baseline estimation</th>
      <td>breslow</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>218</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>87</td>
    </tr>
    <tr>
      <th>partial log-likelihood</th>
      <td>-370.52</td>
    </tr>
    <tr>
      <th>time fit was run</th>
      <td>2024-07-17 11:44:02 UTC</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">exp(coef)</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">exp(coef) lower 95%</th>
      <th style="min-width: 12px;">exp(coef) upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>0.21</td>
      <td>1.23</td>
      <td>0.11</td>
      <td>0.00</td>
      <td>0.42</td>
      <td>1.00</td>
      <td>1.52</td>
      <td>0.00</td>
      <td>1.99</td>
      <td>0.05</td>
      <td>4.41</td>
    </tr>
    <tr>
      <th>Bilirubin</th>
      <td>0.56</td>
      <td>1.76</td>
      <td>0.14</td>
      <td>0.28</td>
      <td>0.84</td>
      <td>1.33</td>
      <td>2.32</td>
      <td>0.00</td>
      <td>3.94</td>
      <td>&lt;0.005</td>
      <td>13.59</td>
    </tr>
    <tr>
      <th>Cholesterol</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Albumin</th>
      <td>-0.13</td>
      <td>0.88</td>
      <td>0.12</td>
      <td>-0.37</td>
      <td>0.11</td>
      <td>0.69</td>
      <td>1.11</td>
      <td>0.00</td>
      <td>-1.07</td>
      <td>0.29</td>
      <td>1.81</td>
    </tr>
    <tr>
      <th>Copper</th>
      <td>0.15</td>
      <td>1.16</td>
      <td>0.12</td>
      <td>-0.08</td>
      <td>0.38</td>
      <td>0.92</td>
      <td>1.46</td>
      <td>0.00</td>
      <td>1.28</td>
      <td>0.20</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>Alk_Phos</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>SGOT</th>
      <td>0.12</td>
      <td>1.13</td>
      <td>0.12</td>
      <td>-0.11</td>
      <td>0.36</td>
      <td>0.90</td>
      <td>1.43</td>
      <td>0.00</td>
      <td>1.03</td>
      <td>0.30</td>
      <td>1.72</td>
    </tr>
    <tr>
      <th>Tryglicerides</th>
      <td>0.09</td>
      <td>1.09</td>
      <td>0.11</td>
      <td>-0.12</td>
      <td>0.29</td>
      <td>0.89</td>
      <td>1.34</td>
      <td>0.00</td>
      <td>0.81</td>
      <td>0.42</td>
      <td>1.25</td>
    </tr>
    <tr>
      <th>Platelets</th>
      <td>-0.02</td>
      <td>0.98</td>
      <td>0.10</td>
      <td>-0.22</td>
      <td>0.18</td>
      <td>0.80</td>
      <td>1.20</td>
      <td>0.00</td>
      <td>-0.20</td>
      <td>0.84</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>Prothrombin</th>
      <td>0.27</td>
      <td>1.31</td>
      <td>0.12</td>
      <td>0.04</td>
      <td>0.50</td>
      <td>1.04</td>
      <td>1.65</td>
      <td>0.00</td>
      <td>2.34</td>
      <td>0.02</td>
      <td>5.68</td>
    </tr>
    <tr>
      <th>Drug</th>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Ascites</th>
      <td>0.17</td>
      <td>1.19</td>
      <td>0.35</td>
      <td>-0.52</td>
      <td>0.86</td>
      <td>0.59</td>
      <td>2.37</td>
      <td>0.00</td>
      <td>0.49</td>
      <td>0.63</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th>Hepatomegaly</th>
      <td>0.07</td>
      <td>1.07</td>
      <td>0.23</td>
      <td>-0.38</td>
      <td>0.53</td>
      <td>0.68</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>0.31</td>
      <td>0.76</td>
      <td>0.40</td>
    </tr>
    <tr>
      <th>Spiders</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>0.43</td>
      <td>1.54</td>
      <td>0.27</td>
      <td>-0.10</td>
      <td>0.96</td>
      <td>0.90</td>
      <td>2.61</td>
      <td>0.00</td>
      <td>1.59</td>
      <td>0.11</td>
      <td>3.16</td>
    </tr>
    <tr>
      <th>Stage_4.0</th>
      <td>0.19</td>
      <td>1.21</td>
      <td>0.25</td>
      <td>-0.29</td>
      <td>0.68</td>
      <td>0.75</td>
      <td>1.97</td>
      <td>0.00</td>
      <td>0.78</td>
      <td>0.44</td>
      <td>1.20</td>
    </tr>
  </tbody>
</table><br><div>
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
  <tbody>
    <tr>
      <th>Concordance</th>
      <td>0.85</td>
    </tr>
    <tr>
      <th>Partial AIC</th>
      <td>775.05</td>
    </tr>
    <tr>
      <th>log-likelihood ratio test</th>
      <td>93.69 on 17 df</td>
    </tr>
    <tr>
      <th>-log2(p) of ll-ratio test</th>
      <td>39.49</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Consolidating the detailed values
# of the model coefficients
##################################
cirrhosis_survival_coxph_L1_25_L2_75.summary
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
      <th>coef</th>
      <th>exp(coef)</th>
      <th>se(coef)</th>
      <th>coef lower 95%</th>
      <th>coef upper 95%</th>
      <th>exp(coef) lower 95%</th>
      <th>exp(coef) upper 95%</th>
      <th>cmp to</th>
      <th>z</th>
      <th>p</th>
      <th>-log2(p)</th>
    </tr>
    <tr>
      <th>covariate</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>2.098302e-01</td>
      <td>1.233469</td>
      <td>0.105619</td>
      <td>0.002821</td>
      <td>0.416839</td>
      <td>1.002825</td>
      <td>1.517158</td>
      <td>0.0</td>
      <td>1.986676</td>
      <td>0.046958</td>
      <td>4.412476</td>
    </tr>
    <tr>
      <th>Bilirubin</th>
      <td>5.633892e-01</td>
      <td>1.756616</td>
      <td>0.142932</td>
      <td>0.283247</td>
      <td>0.843532</td>
      <td>1.327433</td>
      <td>2.324562</td>
      <td>0.0</td>
      <td>3.941649</td>
      <td>0.000081</td>
      <td>13.593084</td>
    </tr>
    <tr>
      <th>Cholesterol</th>
      <td>5.292292e-07</td>
      <td>1.000001</td>
      <td>0.000566</td>
      <td>-0.001109</td>
      <td>0.001110</td>
      <td>0.998892</td>
      <td>1.001110</td>
      <td>0.0</td>
      <td>0.000935</td>
      <td>0.999254</td>
      <td>0.001077</td>
    </tr>
    <tr>
      <th>Albumin</th>
      <td>-1.292244e-01</td>
      <td>0.878777</td>
      <td>0.121025</td>
      <td>-0.366429</td>
      <td>0.107980</td>
      <td>0.693205</td>
      <td>1.114026</td>
      <td>0.0</td>
      <td>-1.067749</td>
      <td>0.285634</td>
      <td>1.807762</td>
    </tr>
    <tr>
      <th>Copper</th>
      <td>1.485291e-01</td>
      <td>1.160127</td>
      <td>0.116375</td>
      <td>-0.079562</td>
      <td>0.376621</td>
      <td>0.923520</td>
      <td>1.457351</td>
      <td>0.0</td>
      <td>1.276293</td>
      <td>0.201852</td>
      <td>2.308632</td>
    </tr>
    <tr>
      <th>Alk_Phos</th>
      <td>6.288623e-07</td>
      <td>1.000001</td>
      <td>0.000719</td>
      <td>-0.001409</td>
      <td>0.001411</td>
      <td>0.998592</td>
      <td>1.001412</td>
      <td>0.0</td>
      <td>0.000874</td>
      <td>0.999303</td>
      <td>0.001007</td>
    </tr>
    <tr>
      <th>SGOT</th>
      <td>1.228628e-01</td>
      <td>1.130729</td>
      <td>0.119216</td>
      <td>-0.110796</td>
      <td>0.356522</td>
      <td>0.895121</td>
      <td>1.428352</td>
      <td>0.0</td>
      <td>1.030591</td>
      <td>0.302733</td>
      <td>1.723884</td>
    </tr>
    <tr>
      <th>Tryglicerides</th>
      <td>8.503213e-02</td>
      <td>1.088752</td>
      <td>0.105471</td>
      <td>-0.121687</td>
      <td>0.291752</td>
      <td>0.885425</td>
      <td>1.338771</td>
      <td>0.0</td>
      <td>0.806212</td>
      <td>0.420120</td>
      <td>1.251125</td>
    </tr>
    <tr>
      <th>Platelets</th>
      <td>-2.019627e-02</td>
      <td>0.980006</td>
      <td>0.101236</td>
      <td>-0.218615</td>
      <td>0.178222</td>
      <td>0.803631</td>
      <td>1.195091</td>
      <td>0.0</td>
      <td>-0.199497</td>
      <td>0.841874</td>
      <td>0.248324</td>
    </tr>
    <tr>
      <th>Prothrombin</th>
      <td>2.734613e-01</td>
      <td>1.314506</td>
      <td>0.117072</td>
      <td>0.044005</td>
      <td>0.502918</td>
      <td>1.044987</td>
      <td>1.653539</td>
      <td>0.0</td>
      <td>2.335843</td>
      <td>0.019499</td>
      <td>5.680423</td>
    </tr>
    <tr>
      <th>Drug</th>
      <td>-1.579953e-06</td>
      <td>0.999998</td>
      <td>0.002154</td>
      <td>-0.004223</td>
      <td>0.004220</td>
      <td>0.995786</td>
      <td>1.004229</td>
      <td>0.0</td>
      <td>-0.000734</td>
      <td>0.999415</td>
      <td>0.000845</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>-3.844239e-07</td>
      <td>1.000000</td>
      <td>0.000832</td>
      <td>-0.001631</td>
      <td>0.001630</td>
      <td>0.998370</td>
      <td>1.001632</td>
      <td>0.0</td>
      <td>-0.000462</td>
      <td>0.999631</td>
      <td>0.000532</td>
    </tr>
    <tr>
      <th>Ascites</th>
      <td>1.722505e-01</td>
      <td>1.187975</td>
      <td>0.353328</td>
      <td>-0.520260</td>
      <td>0.864761</td>
      <td>0.594366</td>
      <td>2.374439</td>
      <td>0.0</td>
      <td>0.487509</td>
      <td>0.625898</td>
      <td>0.676001</td>
    </tr>
    <tr>
      <th>Hepatomegaly</th>
      <td>7.177581e-02</td>
      <td>1.074414</td>
      <td>0.232008</td>
      <td>-0.382951</td>
      <td>0.526503</td>
      <td>0.681846</td>
      <td>1.693001</td>
      <td>0.0</td>
      <td>0.309368</td>
      <td>0.757042</td>
      <td>0.401555</td>
    </tr>
    <tr>
      <th>Spiders</th>
      <td>9.054306e-07</td>
      <td>1.000001</td>
      <td>0.000961</td>
      <td>-0.001883</td>
      <td>0.001885</td>
      <td>0.998119</td>
      <td>1.001887</td>
      <td>0.0</td>
      <td>0.000942</td>
      <td>0.999248</td>
      <td>0.001085</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>4.290145e-01</td>
      <td>1.535743</td>
      <td>0.269937</td>
      <td>-0.100052</td>
      <td>0.958081</td>
      <td>0.904791</td>
      <td>2.606688</td>
      <td>0.0</td>
      <td>1.589316</td>
      <td>0.111989</td>
      <td>3.158569</td>
    </tr>
    <tr>
      <th>Stage_4.0</th>
      <td>1.920942e-01</td>
      <td>1.211785</td>
      <td>0.246854</td>
      <td>-0.291731</td>
      <td>0.675920</td>
      <td>0.746969</td>
      <td>1.965840</td>
      <td>0.0</td>
      <td>0.778169</td>
      <td>0.436470</td>
      <td>1.196047</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting the hazard ratio of the
# formulated Cox Regression model
# with Predominantly L2-Weighted and L1 Penalty
##################################
cirrhosis_survival_coxph_L1_25_L2_75_summary = cirrhosis_survival_coxph_L1_25_L2_75.summary
cirrhosis_survival_coxph_L1_25_L2_75_summary['hazard_ratio'] = np.exp(cirrhosis_survival_coxph_L1_25_L2_75_summary['coef'])
significant = cirrhosis_survival_coxph_L1_25_L2_75_summary['p'] < 0.05
plt.figure(figsize=(17, 8))
colors = ['#993300' if sig else '#CC9966' for sig in significant]

plt.barh(cirrhosis_survival_coxph_L1_25_L2_75_summary.index, 
         cirrhosis_survival_coxph_L1_25_L2_75_summary['hazard_ratio'], 
         xerr=cirrhosis_survival_coxph_L1_25_L2_75_summary['se(coef)'], 
         color=colors)
plt.xlabel('Hazard Ratio')
plt.ylabel('Variables')
plt.title('COXPH_PWL2L1P Hazard Ratio Forest Plot')
plt.axvline(x=1, color='k', linestyle='--')
plt.gca().invert_yaxis()
plt.show()
```


    
![png](output_238_0.png)
    



```python
##################################
# Plotting the coefficient magnitude
# of the formulated Cox Regression model
# with Predominantly L2-Weighted and L1 Penalty
##################################
plt.figure(figsize=(17, 8))
colors = ['#993300' if sig else '#CC9966' for sig in significant]
cirrhosis_survival_coxph_L1_25_L2_75_summary['coef'].plot(kind='barh', color=colors)
plt.xlabel('Variables')
plt.ylabel('Model Coefficient Value')
plt.title('COXPH_PWL2L1P Model Coefficients')
plt.xticks(rotation=0, ha='right')
plt.xlim(-1,1)
plt.gca().invert_yaxis()
plt.show()
```


    
![png](output_239_0.png)
    



```python
##################################
# Gathering the apparent model performance
# as baseline for evaluating overfitting
##################################
cirrhosis_survival_coxph_L1_25_L2_75.fit(cirrhosis_survival_train_modeling, duration_col='N_Days', event_col='Status')
train_predictions = cirrhosis_survival_coxph_L1_25_L2_75.predict_partial_hazard(cirrhosis_survival_train_modeling)
cirrhosis_survival_coxph_L1_25_L2_75_train_ci = concordance_index(cirrhosis_survival_train_modeling['N_Days'], 
                                                                     -train_predictions, 
                                                                     cirrhosis_survival_train_modeling['Status'])
display(f"Apparent Concordance Index: {cirrhosis_survival_coxph_L1_25_L2_75_train_ci}")
```


    'Apparent Concordance Index: 0.8501300390117035'



```python
##################################
# Performing 5-Fold Cross-Validation
# on the training data
##################################
kf = KFold(n_splits=5, shuffle=True, random_state=88888888)
c_index_scores = []

for train_index, val_index in kf.split(cirrhosis_survival_train_modeling):
    df_train_fold = cirrhosis_survival_train_modeling.iloc[train_index]
    df_val_fold = cirrhosis_survival_train_modeling.iloc[val_index]
    
    cirrhosis_survival_coxph_L1_25_L2_75.fit(df_train_fold, duration_col='N_Days', event_col='Status')
    val_predictions = cirrhosis_survival_coxph_L1_25_L2_75.predict_partial_hazard(df_val_fold)
    c_index = concordance_index(df_val_fold['N_Days'], -val_predictions, df_val_fold['Status'])
    c_index_scores.append(c_index)

cirrhosis_survival_coxph_L1_25_L2_75_cv_ci_mean = np.mean(c_index_scores)
cirrhosis_survival_coxph_L1_25_L2_75_cv_ci_std = np.std(c_index_scores)

display(f"Cross-Validated Concordance Index: {cirrhosis_survival_coxph_L1_25_L2_75_cv_ci_mean}")
```


    'Cross-Validated Concordance Index: 0.8152592390610126'



```python
##################################
# Evaluating the model performance
# on test data
##################################
test_predictions = cirrhosis_survival_coxph_L1_25_L2_75.predict_partial_hazard(cirrhosis_survival_test_modeling)
cirrhosis_survival_coxph_L1_25_L2_75_test_ci = concordance_index(cirrhosis_survival_test_modeling['N_Days'], 
                                                                     -test_predictions, 
                                                                     cirrhosis_survival_test_modeling['Status'])
display(f"Test Concordance Index: {cirrhosis_survival_coxph_L1_25_L2_75_test_ci}")
```


    'Test Concordance Index: 0.8671201814058956'



```python
##################################
# Gathering the concordance indices
# from training, cross-validation and test
##################################
coxph_L1_25_L2_75_set = pd.DataFrame(["Train","Cross-Validation","Test"])
coxph_L1_25_L2_75_ci_values = pd.DataFrame([cirrhosis_survival_coxph_L1_25_L2_75_train_ci,
                                           cirrhosis_survival_coxph_L1_25_L2_75_cv_ci_mean,
                                           cirrhosis_survival_coxph_L1_25_L2_75_test_ci])
coxph_L1_25_L2_75_method = pd.DataFrame(["COXPH_PWL2L1P"]*3)
coxph_L1_25_L2_75_summary = pd.concat([coxph_L1_25_L2_75_set, 
                                     coxph_L1_25_L2_75_ci_values,
                                     coxph_L1_25_L2_75_method], 
                                    axis=1)
coxph_L1_25_L2_75_summary.columns = ['Set', 'Concordance.Index', 'Method']
coxph_L1_25_L2_75_summary.reset_index(inplace=True, drop=True)
display(coxph_L1_25_L2_75_summary)
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
      <th>Set</th>
      <th>Concordance.Index</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Train</td>
      <td>0.850130</td>
      <td>COXPH_PWL2L1P</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cross-Validation</td>
      <td>0.815259</td>
      <td>COXPH_PWL2L1P</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Test</td>
      <td>0.867120</td>
      <td>COXPH_PWL2L1P</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Binning the predicted risks
# into dichotomous groups and
# exploring the relationships with
# survival event and duration
##################################
cirrhosis_survival_test_modeling['Predicted_Risks_COXPH_PWL2L1P'] = test_predictions
cirrhosis_survival_test_modeling['Predicted_RiskGroups_COXPH_PWL2L1P'] = risk_groups = pd.qcut(cirrhosis_survival_test_modeling['Predicted_Risks_COXPH_PWL2L1P'], 2, labels=['Low-Risk', 'High-Risk'])

plt.figure(figsize=(17, 8))
for group in risk_groups.unique():
    group_data = cirrhosis_survival_test_modeling[risk_groups == group]
    kmf.fit(group_data['N_Days'], event_observed=group_data['Status'], label=group)
    kmf.plot_survival_function()

plt.title('COXPH_PWL2L1P Survival Probabilities by Predicted Risk Groups')
plt.xlabel('N_Days')
plt.ylabel('Event Survival Probability')
plt.show()
```


    
![png](output_244_0.png)
    



```python
##################################
# Gathering the predictor information
# for 5 test case samples
##################################
test_case_details = cirrhosis_survival_test_modeling.iloc[[10, 20, 30, 40, 50]]
display(test_case_details)
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
      <th>Predicted_Risks_COXPH_FL1P</th>
      <th>Predicted_RiskGroups_COXPH_FL1P</th>
      <th>Predicted_Risks_COXPH_FL2P</th>
      <th>Predicted_RiskGroups_COXPH_FL2P</th>
      <th>Predicted_Risks_COXPH_EL1L2P</th>
      <th>Predicted_RiskGroups_COXPH_EL1L2P</th>
      <th>Predicted_Risks_COXPH_PWL1L2P</th>
      <th>Predicted_RiskGroups_COXPH_PWL1L2P</th>
      <th>Predicted_Risks_COXPH_PWL2L1P</th>
      <th>Predicted_RiskGroups_COXPH_PWL2L1P</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>True</td>
      <td>1827</td>
      <td>0.226982</td>
      <td>1.530100</td>
      <td>1.302295</td>
      <td>1.331981</td>
      <td>1.916467</td>
      <td>-0.477846</td>
      <td>-0.451305</td>
      <td>2.250260</td>
      <td>...</td>
      <td>3.515780</td>
      <td>High-Risk</td>
      <td>4.748349</td>
      <td>High-Risk</td>
      <td>4.442277</td>
      <td>High-Risk</td>
      <td>3.761430</td>
      <td>High-Risk</td>
      <td>4.348925</td>
      <td>High-Risk</td>
    </tr>
    <tr>
      <th>20</th>
      <td>False</td>
      <td>1447</td>
      <td>-0.147646</td>
      <td>0.061189</td>
      <td>0.793618</td>
      <td>-1.158235</td>
      <td>0.861264</td>
      <td>0.625621</td>
      <td>0.319035</td>
      <td>0.446026</td>
      <td>...</td>
      <td>0.769723</td>
      <td>Low-Risk</td>
      <td>0.681968</td>
      <td>Low-Risk</td>
      <td>0.741409</td>
      <td>Low-Risk</td>
      <td>0.747044</td>
      <td>Low-Risk</td>
      <td>0.694654</td>
      <td>Low-Risk</td>
    </tr>
    <tr>
      <th>30</th>
      <td>False</td>
      <td>2574</td>
      <td>0.296370</td>
      <td>-1.283677</td>
      <td>0.169685</td>
      <td>3.237777</td>
      <td>-1.008276</td>
      <td>-0.873566</td>
      <td>-0.845549</td>
      <td>-0.351236</td>
      <td>...</td>
      <td>0.320200</td>
      <td>Low-Risk</td>
      <td>0.165574</td>
      <td>Low-Risk</td>
      <td>0.255343</td>
      <td>Low-Risk</td>
      <td>0.295159</td>
      <td>Low-Risk</td>
      <td>0.202767</td>
      <td>Low-Risk</td>
    </tr>
    <tr>
      <th>40</th>
      <td>True</td>
      <td>3762</td>
      <td>0.392609</td>
      <td>-0.096645</td>
      <td>-0.486337</td>
      <td>1.903146</td>
      <td>-0.546292</td>
      <td>-0.247141</td>
      <td>-0.720619</td>
      <td>-0.810790</td>
      <td>...</td>
      <td>1.167777</td>
      <td>High-Risk</td>
      <td>1.591530</td>
      <td>High-Risk</td>
      <td>1.275445</td>
      <td>High-Risk</td>
      <td>1.214395</td>
      <td>High-Risk</td>
      <td>1.404015</td>
      <td>High-Risk</td>
    </tr>
    <tr>
      <th>50</th>
      <td>False</td>
      <td>837</td>
      <td>-0.813646</td>
      <td>1.089037</td>
      <td>0.064451</td>
      <td>0.212865</td>
      <td>2.063138</td>
      <td>-0.224432</td>
      <td>0.074987</td>
      <td>2.333282</td>
      <td>...</td>
      <td>1.832490</td>
      <td>High-Risk</td>
      <td>2.541904</td>
      <td>High-Risk</td>
      <td>2.125042</td>
      <td>High-Risk</td>
      <td>1.832724</td>
      <td>High-Risk</td>
      <td>2.288206</td>
      <td>High-Risk</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



```python
##################################
# Gathering the risk-groups
# for 5 test case samples
##################################
print(cirrhosis_survival_test_modeling.loc[[10, 20, 30, 40, 50]][['Predicted_RiskGroups_COXPH_PWL2L1P']])
```

       Predicted_RiskGroups_COXPH_PWL2L1P
    10                          High-Risk
    20                           Low-Risk
    30                           Low-Risk
    40                          High-Risk
    50                          High-Risk
    


```python
##################################
# Estimating the cumulative hazard
# and survival functions
# for 5 test cases
##################################
test_case = cirrhosis_survival_test_modeling.iloc[[10, 20, 30, 40, 50]]
test_case_labels = ['Patient_10','Patient_20','Patient_30','Patient_40','Patient_50']

fig, axes = plt.subplots(1, 2, figsize=(17, 8))
for i, (index, row) in enumerate(test_case.iterrows()):
    survival_function = cirrhosis_survival_coxph_L1_25_L2_75.predict_survival_function(row.to_frame().T)
    axes[0].plot(survival_function, label=f'Sample {i+1}')
axes[0].set_title('COXPH_PWL2L1P Survival Function for 5 Test Cases')
axes[0].set_xlabel('N_Days')
axes[0].set_ylim(0,1)
axes[0].set_ylabel('Survival Probability')
axes[0].legend(test_case_labels, loc="lower left")
for i, (index, row) in enumerate(test_case.iterrows()):
    hazard_function = cirrhosis_survival_coxph_L1_25_L2_75.predict_cumulative_hazard(row.to_frame().T)
    axes[1].plot(hazard_function, label=f'Sample {i+1}')
axes[1].set_title('COXPH_PWL2L1P Cumulative Hazard for 5 Test Cases')
axes[1].set_xlabel('N_Days')
axes[1].set_ylim(0,5)
axes[1].set_ylabel('Cumulative Hazard')
axes[1].legend(test_case_labels, loc="upper left")
plt.tight_layout()
plt.show()
```


    
![png](output_247_0.png)
    



```python
##################################
# Creating the explainer object
##################################
cirrhosis_survival_coxph_L1_25_L2_75_explainer = shap.Explainer(cirrhosis_survival_coxph_L1_25_L2_75.predict_partial_hazard, 
                                                    cirrhosis_survival_train_modeling.drop(columns=["N_Days", "Status"]))
cirrhosis_survival_coxph_L1_25_L2_75_shap_values = cirrhosis_survival_coxph_L1_25_L2_75_explainer(cirrhosis_survival_train_modeling.drop(columns=["N_Days", "Status"]))
```

    PermutationExplainer explainer: 219it [00:22,  5.36it/s]                         
    


```python
##################################
# Plotting the SHAP summary plot
##################################
shap.summary_plot(cirrhosis_survival_coxph_L1_25_L2_75_shap_values, 
                  cirrhosis_survival_train_modeling.drop(columns=["N_Days", "Status"]),
                  sort=False)
```


    
![png](output_249_0.png)
    


## 1.7. Consolidated Findings <a class="anchor" id="1.7"></a>




```python
##################################
# Consolidating all the
# model performance metrics
##################################
model_performance_comparison = pd.concat([coxph_L1_0_L2_0_summary, 
                                          coxph_L1_100_L2_0_summary,
                                          coxph_L1_0_L2_100_summary, 
                                          coxph_L1_50_L2_50_summary,
                                          coxph_L1_75_L2_25_summary,
                                          coxph_L1_25_L2_75_summary], 
                                         axis=0,
                                         ignore_index=True)
print('Cox Regression Model Comparison: ')
display(model_performance_comparison)
```

    Cox Regression Model Comparison: 
    


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
      <th>Set</th>
      <th>Concordance.Index</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Train</td>
      <td>0.847854</td>
      <td>COXPH_NP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cross-Validation</td>
      <td>0.802364</td>
      <td>COXPH_NP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Test</td>
      <td>0.848073</td>
      <td>COXPH_NP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Train</td>
      <td>0.831356</td>
      <td>COXPH_FL1P</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cross-Validation</td>
      <td>0.811197</td>
      <td>COXPH_FL1P</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Test</td>
      <td>0.842630</td>
      <td>COXPH_FL1P</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Train</td>
      <td>0.853381</td>
      <td>COXPH_FL2P</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cross-Validation</td>
      <td>0.809983</td>
      <td>COXPH_FL2P</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Test</td>
      <td>0.867574</td>
      <td>COXPH_FL2P</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Train</td>
      <td>0.846554</td>
      <td>COXPH_EL1L2P</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cross-Validation</td>
      <td>0.816280</td>
      <td>COXPH_EL1L2P</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Test</td>
      <td>0.863039</td>
      <td>COXPH_EL1L2P</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Train</td>
      <td>0.839402</td>
      <td>COXPH_PWL1L2P</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Cross-Validation</td>
      <td>0.810178</td>
      <td>COXPH_PWL1L2P</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Test</td>
      <td>0.852608</td>
      <td>COXPH_PWL1L2P</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Train</td>
      <td>0.850130</td>
      <td>COXPH_PWL2L1P</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Cross-Validation</td>
      <td>0.815259</td>
      <td>COXPH_PWL2L1P</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Test</td>
      <td>0.867120</td>
      <td>COXPH_PWL2L1P</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Consolidating the concordance indices
# for all sets and models
##################################
set_labels = ['Train','Cross-Validation','Test']
coxph_np_ci = model_performance_comparison[((model_performance_comparison['Set'] == 'Train') |
                                            (model_performance_comparison['Set'] == 'Cross-Validation') |
                                            (model_performance_comparison['Set'] == 'Test')) & 
                                           (model_performance_comparison['Method']=='COXPH_NP')]['Concordance.Index'].values
coxph_fl1p_ci = model_performance_comparison[((model_performance_comparison['Set'] == 'Train') |
                                              (model_performance_comparison['Set'] == 'Cross-Validation') |
                                              (model_performance_comparison['Set'] == 'Test')) & 
                                             (model_performance_comparison['Method']=='COXPH_FL1P')]['Concordance.Index'].values
coxph_fl2p_ci = model_performance_comparison[((model_performance_comparison['Set'] == 'Train') |
                                              (model_performance_comparison['Set'] == 'Cross-Validation') |
                                              (model_performance_comparison['Set'] == 'Test')) & 
                                             (model_performance_comparison['Method']=='COXPH_FL2P')]['Concordance.Index'].values
coxph_el1l2p_ci = model_performance_comparison[((model_performance_comparison['Set'] == 'Train') |
                                                (model_performance_comparison['Set'] == 'Cross-Validation') |
                                                (model_performance_comparison['Set'] == 'Test')) &  
                                               (model_performance_comparison['Method']=='COXPH_EL1L2P')]['Concordance.Index'].values
coxph_pwl1l2p_ci = model_performance_comparison[((model_performance_comparison['Set'] == 'Train') |
                                                 (model_performance_comparison['Set'] == 'Cross-Validation') |
                                                 (model_performance_comparison['Set'] == 'Test')) & 
                                                (model_performance_comparison['Method']=='COXPH_PWL1L2P')]['Concordance.Index'].values
coxph_pwl2l1p_ci = model_performance_comparison[((model_performance_comparison['Set'] == 'Train') |
                                                 (model_performance_comparison['Set'] == 'Cross-Validation') |
                                                 (model_performance_comparison['Set'] == 'Test')) & 
                                                (model_performance_comparison['Method']=='COXPH_PWL2L1P')]['Concordance.Index'].values
```


```python
##################################
# Plotting the values for the
# concordance indices
# for all models
##################################
ci_plot = pd.DataFrame({'COXPH_NP': list(coxph_np_ci),
                        'COXPH_FL1P': list(coxph_fl1p_ci),
                        'COXPH_FL2P': list(coxph_fl2p_ci),
                        'COXPH_EL1L2P': list(coxph_el1l2p_ci),
                        'COXPH_PWL1L2P': list(coxph_pwl1l2p_ci),
                        'COXPH_PWL2L1P': list(coxph_pwl2l1p_ci)},
                       index = set_labels)
display(ci_plot)
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
      <th>COXPH_NP</th>
      <th>COXPH_FL1P</th>
      <th>COXPH_FL2P</th>
      <th>COXPH_EL1L2P</th>
      <th>COXPH_PWL1L2P</th>
      <th>COXPH_PWL2L1P</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Train</th>
      <td>0.847854</td>
      <td>0.831356</td>
      <td>0.853381</td>
      <td>0.846554</td>
      <td>0.839402</td>
      <td>0.850130</td>
    </tr>
    <tr>
      <th>Cross-Validation</th>
      <td>0.802364</td>
      <td>0.811197</td>
      <td>0.809983</td>
      <td>0.816280</td>
      <td>0.810178</td>
      <td>0.815259</td>
    </tr>
    <tr>
      <th>Test</th>
      <td>0.848073</td>
      <td>0.842630</td>
      <td>0.867574</td>
      <td>0.863039</td>
      <td>0.852608</td>
      <td>0.867120</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting all the concordance indices
# for all models
##################################
ci_plot = ci_plot.plot.barh(figsize=(10, 6), width=0.90)
ci_plot.set_xlim(0.00,1.00)
ci_plot.set_title("Model Comparison by Concordance Indices")
ci_plot.set_xlabel("Concordance Index")
ci_plot.set_ylabel("Data Set")
ci_plot.grid(False)
ci_plot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
for container in ci_plot.containers:
    ci_plot.bar_label(container, fmt='%.5f', padding=-50, color='white', fontweight='bold')
```


    
![png](output_254_0.png)
    


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

