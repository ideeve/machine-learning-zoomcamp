
# Customer Banking Dataset README

## Overview

This dataset contains information about bank customers, capturing various attributes related to their banking and demographic details. It is useful for analyzing customer behavior, predicting churn, and understanding demographic distributions.

## Data Description

The dataset comprises the following columns:

1. **id**: Unique identifier for each record.
2. **CustomerId**: Unique identifier for each customer.
3. **Surname**: Last name of the customer.
4. **CreditScore**: The credit score of the customer.
5. **Geography**: The country of the customer (e.g., France, Spain, Germany).
6. **Gender**: The gender of the customer (Male/Female).
7. **Age**: The age of the customer.
8. **Tenure**: Number of years the customer has been with the bank.
9. **Balance**: Account balance of the customer.
10. **NumOfProducts**: Number of bank products used by the customer.
11. **HasCrCard**: Indicates whether the customer has a credit card (1: Yes, 0: No).
12. **IsActiveMember**: Indicates whether the customer is an active member (1: Yes, 0: No).
13. **EstimatedSalary**: Estimated annual salary of the customer.
14. **Exited**: Indicates whether the customer has exited (closed account with the bank) (1: Yes, 0: No).

## Data Format

The dataset is presented in a comma-separated values (CSV) format.

## Usage Scenarios

This dataset can be used for various analytical and machine learning purposes, such as:

- Predicting customer churn.
- Analyzing customer demographics.
- Assessing credit risk based on customer profiles.
- Personalized marketing based on customer behavior and demographics.



### Dataset: "bank.csv"

## Introduction
Embarking on the exploration of the "Adult Income" dataset involves a step-by-step journey into its intricacies, aiming to unravel patterns and glean valuable insights. Let's navigate through each phase of the Exploratory Data Analysis (EDA).

## Procedure
1. **Data Loading:** Our journey begins by loading the dataset, a collection of diverse features encompassing age, work class, education, and income etc. This initial step sets the stage for our exploration.

2. **Data Summary:** A swift overview of the dataset provides crucial information about its structure, data types, and the potential presence of missing values. This summary aids in comprehending the overall landscape of the data.

3. **Descriptive Statistics:** To delve deeper, we extract descriptive statistics that reveal the central tendencies, dispersions, and other key metrics of the numerical features. This step lays the foundation for a nuanced understanding of the dataset's numerical aspects.

4. **Missing Values:** A meticulous examination follows to identify and address any missing values. This ensures the integrity of the dataset, forming a solid basis for subsequent analyses.

5. **Data Distribution:** Visualizing the distribution of numerical features sheds light on the spread and concentration of data points. Histograms and other visualizations provide a glimpse into the patterns inherent in the dataset.

6. **Categorical Data Distribution:** Turning our attention to categorical features, we explore the count distribution across different categories. This step unveils trends and disparities within the dataset, particularly regarding income.

7. **Income Distribution by Education Level, Gender, and Race:** Our exploration culminates in dissecting income distribution across key demographic variables—education level, gender, and race. These analyses provide insights into how income varies within different segments of the population.

This comprehensive journey through the "Adult Income" dataset equips us with a nuanced understanding, paving the way for deeper analyses and potential insights into socioeconomic dynamics captured by the data.

# Getting Started

**Prerequisites:**
- Python
- Pipenv
- Docker

## Installing Dependencies

Install the dependencies with pipenv (because the version of model XgBoost that I used on this project has to be the same), as they are specified in the Pipfile and Pipfile.lock, by running the following commands:

```bash
pipenv install
pipenv shell
```

## Building the model

Execute either the train.py file (This file is in the final_model folder) to carry out all the necessary steps for training the final model used in this project.

```bash
python train.py
```

## Testing the model 

To test the model:

In one terminal run 

```bash
python predict.py
```

In another terminal run 

```bash
python predict_test.py
```

## Using Docker 

Also, you can use the model with Docker:

Make sure you have Docker installed, then go to the CapStone-1 folder and run 

### Create Docker image

```bash
docker build -t CapStone-2 .
```

### Run the image 
```bash
docker run -it --rm -p 8080:8080 CapStone-2:latest
```

### Testing the model

Run
```bash
python test.py
```