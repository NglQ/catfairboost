# catfairboost

## Authors
Edoardo Fusa <edoardo.fusa@studio.unibo.it>

Angelo Quarta <angelo.quarta@studio.unibo.it>

## Datasets info

### Diabetes Easy

The "Diabetes 130-Hospitals" dataset represents 10 years of clinical care at 130 U.S. hospitals and delivery networks,
collected from 1999 to 2008.
It is the same dataset of `Diabetes Hard`, but it has been processed to make the classification task much easier.

| Info                       | #        |
|----------------------------|----------|
| Instances                  | 100000   |
| Columns                    | 11       |
| Target                     | diabetes |
| Target type                | binary   |
| Chosen sensitive attribute | race     |

More info [here](https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset/data).


### Diabetes Hard

The "Diabetes 130-Hospitals" dataset represents 10 years of clinical care at 130 U.S. hospitals and delivery networks,
collected from 1999 to 2008.
Each record represents the hospital admission record for a patient diagnosed with diabetes
whose stay lasted between one and fourteen days.
The features describing each encounter include demographics, diagnoses,
diabetic medications, number of visits in the year preceding the encounter, and payer information, as well as whether
the patient was readmitted after release, and whether the readmission occurred within 30 days of the release.

| Info                       | #          |
|----------------------------|------------|
| Instances                  | 101766     |
| Columns                    | 47         |
| Target                     | readmitted |
| Target type                | binary     |
| Chosen sensitive attribute | race       |

More info [here](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).


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

Our dataset variant (that can be chosen with the `folktables` python library API) is
`ACSDataSource(survey_year='2018', horizon='1-Year', survey='person').get_data(states=['CA', 'TX', 'WA'])`.


| Info                       | #                                      |
|----------------------------|----------------------------------------|
| Instances                  | 371533                                 |
| Columns                    | 11                                     |
| Target                     | PINCP (total annual income per person) |
| Target type                | binary                                 |
| Chosen sensitive attribute | RAC1P (race)                           |

More info [here](https://github.com/socialfoundations/folktables).


### ACSEmployment

Our dataset variant (that can be chosen with the `folktables` python library API) is
`ACSDataSource(survey_year='2018', horizon='1-Year', survey='person').get_data(states=['CA', 'TX', 'WA'])`.

| Info                       | #                       |
|----------------------------|-------------------------|
| Instances                  | 723142                  |
| Columns                    | 17                      |
| Target                     | ESR (employment status) |
| Target type                | binary                  |
| Chosen sensitive attribute | RAC1P (race)            |

More info [here](https://github.com/socialfoundations/folktables).


### ACSMobility

Our dataset variant (that can be chosen with the `folktables` python library API) is
`ACSDataSource(survey_year='2018', horizon='1-Year', survey='person').get_data(states=['CA', 'TX', 'WA'])`.

| Info                       | #                                             |
|----------------------------|-----------------------------------------------|
| Instances                  | 148678                                        |
| Columns                    | 22                                            |
| Target                     | MIG (Mobility status (lived here 1 year ago)) |
| Target type                | binary                                        |
| Chosen sensitive attribute | RAC1P (race)                                  |

More info [here](https://github.com/socialfoundations/folktables).


### ACSIncomePovertyRatio

Our dataset variant (that can be chosen with the `folktables` python library API) is
`ACSDataSource(survey_year='2018', horizon='1-Year', survey='person').get_data(states=['CA', 'TX', 'WA'])`.

| Info                       | #                                |
|----------------------------|----------------------------------|
| Instances                  | 723142                           |
| Columns                    | 21                               |
| Target                     | POVPIP (Income-to-poverty ratio) |
| Target type                | binary                           |
| Chosen sensitive attribute | RAC1P (race)                     |

More info [here](https://github.com/socialfoundations/folktables).
