# fairness-ai-review

## Datasets info

### Diabetes

The "Diabetes 130-Hospitals" dataset represents 10 years of clinical care at 130 U.S. hospitals and delivery networks,
collected from 1999 to 2008.
Each record represents the hospital admission record for a patient diagnosed with diabetes
whose stay lasted between one and fourteen days.
The features describing each encounter include demographics, diagnoses,
diabetic medications, number of visits in the year preceding the encounter, and payer information, as well as whether
the patient was readmitted after release, and whether the readmission occurred within 30 days of the release.

| Info                 | #                                           |
|----------------------|---------------------------------------------|
| Instances            | 101766                                      |
| Columns              | 25                                          |
| Targets              | readmitted, readmit_binary, readmit_30_days |
| Target type          | binary                                      |
| Protected attributes | race, gender, age, medicare?, medicaid?     |

More info [here](https://fairlearn.org/v0.10/user_guide/datasets/diabetes_hospital_data.html).


### ACSIncome

The ACSIncome dataset is one of five datasets created by
[Ding et al.](https://fairlearn.org/v0.10/user_guide/datasets/acs_income.html#footcite-ding2021retiring) as an improved
alternative to the popular UCI Adult dataset. Briefly, the UCI Adult dataset is commonly used as a benchmark dataset
when comparing different algorithmic fairness interventions. ACSIncome offers a few improvements, such as providing more
datapoints (1,664,500 vs. 48,842) and more recent data (2018 vs. 1994). Further, the binary labels in the UCI Adult
dataset indicate whether an individual earned more than $50k US dollars in that year. Ding et al. show that the choice
of threshold impacts the amount of disparity in proportion of positives, so they allow users to define any threshold
rather than fixing it at $50k.

Ding et al. compiled data from the American Community Survey (ACS) Public Use Microdata Sample (PUMS). Note that this
is a different source than the Annual Social and Economic Supplement (ASEC) of the Current Population Survey (CPS) used
to construct the original UCI Adult dataset. Ding et al. filtered the data such that ACSIncome only includes individuals
above 16 years old who worked at least 1 hour per week in the past year and had an income of at least $100.

| Info                 | #                                      |
|----------------------|----------------------------------------|
| Instances            | 1664500                                |
| Columns              | 11                                     |
| Targets              | PINCP (total annual income per person) |
| Target type          | numeric                                |
| Protected attributes | possibly all attributes                |

More info [here](https://fairlearn.org/v0.10/user_guide/datasets/acs_income.html).


### Bank marketing

The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were
based on phone calls. Often, more than one contact to the same client was required, in order to access if the product
(bank term deposit) would be (or not) subscribed.

The classification goal is to predict if the client will subscribe a term deposit.

| Info                 | #                            |
|----------------------|------------------------------|
| Instances            | 45211                        |
| Columns              | 17                           |
| Targets              | y (yes or no)                |
| Target type          | binary                       |
| Protected attributes | age, job, marital, education |

More info [here](https://openml.org/search?type=data&status=active&id=1461).


### Default of credit card clients

The dataset is about default payments in Taiwan.

| Info                 | #                                      |
|----------------------|----------------------------------------|
| Instances            | 30000                                  |
| Columns              | 24                                     |
| Targets              | y (yes or no)                          |
| Target type          | binary                                 |
| Protected attributes | gender, education, marital status, age |

More info [here](https://openml.org/search?type=data&status=active&id=42477).
