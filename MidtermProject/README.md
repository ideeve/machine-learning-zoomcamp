# Breast Cancer Survival Prediction Project

## Introduction
Breast cancer is a very common and dangerous disease that affects many women all over the world. It can be very serious, but if we can find it early, it's easier to treat and people have a better chance of getting better. So, I want to create a computer program that can guess if a person with breast cancer will get better or not. To do this, I will use some information about the person, like their age, the size of the tumor, and other things. The computer program will look at all this information and make a guess about whether the person will survive or not. This is important because if we can predict who is more likely to survive, doctors can focus on them and give them the best treatment. It can help save lives and make sure people get the right care.

## Project Overview
We're using a computer program (built in Python) that's like a smart assistant. It looks at lots of information about breast cancer patients, such as their age, the size of their tumor, and other things. With all this information, our computer program will try to guess if a patient is likely to survive or not.


## Importance
If we can make good guesses, doctors can focus more on patients who might be in more danger and need extra attention. By using this computer program, we hope to help doctors save lives and make sure people get the best care. We'll use things like accuracy, recall, and precision to make sure our computer program is doing a good job at making these predictions.


# Getting Started

To run this project, follow these steps:

## Prerequisites

- Docker: You need to have Docker installed on your system.

## Installation

1. Clone this repository.
2. Build the Docker image using the following command:

```bash
docker build -t <image-name> .
```

3. Run the Docker container with the following command:

```bash
docker run -it --rm -p 4041:4041 <image-name>
```

4. Install development packages with the following command:

```bash
pipenv install --dev
```

## Using the Prediction Service

1. Once you have the container up and running, you can start using the model by running:

```bash
python test-predict.py
```

2. Edit the dictionary in test-predict.py with your custom records to get your predictions.

3. You can also modify the test-predict.py file with samples from the patient-test-dataset file.

## Note

- The prediction service processes one record in a single request.

## Warning

- This model is for educational purposes only and may not provide accurate or reliable predictions. Please use caution and do not make critical decisions based solely on its outputs.
- Always consult with domain experts and consider real-world data before relying on this model for practical applications.

