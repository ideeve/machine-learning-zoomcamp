# Adult Income Prediction Project

## Problem Statement
Understanding the Impact of Socio-Demographic Factors on Adult Income

### Key Questions to Explore:
- Influence of Education on Income
- Occupational Disparities in income distribution
- Marital Status and Income
- Gender Disparities in adult income
  - Is there a gender-based income gap within the dataset?
- Age as a Contributing Factor in income
- Geographical Impact on income

### Potential Insights to Derive:
- Identify socio-demographic factors that serve as strong indicators of higher income levels.
- Uncover any disparities or biases in income distribution based on gender, marital status, or occupation.
- Understand the role of education in shaping income outcomes and whether it acts as a key driver of economic success.
- Explore the potential influence of geographic location on income, considering the native country variable.

## Overall Objective:
Predicting adult income based on socio-economic factors provides valuable insights for policymakers, educational institutions, organizations, and individuals. These insights can guide targeted interventions and social welfare programs, allocate resources efficiently, and inform labor market planning. Educational institutions can offer more informed career guidance, aligning academic pursuits with higher income paths. Organizations benefit by identifying and addressing income disparities within their workforce, contributing to diversity and inclusion efforts. Individuals can make better-informed decisions about personal finance and career choices. Social mobility programs can be tailored based on predictive insights, facilitating upward mobility. Researchers gain valuable information for academic studies on societal structures and economic outcomes. Economic forecasters, businesses, and policymakers can anticipate changes in the economic landscape. In essence, predictive analytics in adult income prediction contributes to more equitable societies, informed decision-making, and targeted strategies to address income disparities.

### Dataset: "adult.csv"

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
docker build -t CapStone-1 .
```

### Run the image 
```bash
docker run -it --rm -p 8080:8080 CapStone-1:latest
```

### Testing the model

Run
```bash
python test.py
```