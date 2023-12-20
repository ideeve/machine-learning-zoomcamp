import requests

url = 'http://localhost:8080/'

data = {"person" : {
    "age" : 32, 
    "workclass": "Private", 
    "final_weight": 334409 , 
    "education": "Master", 
    "educationnum": 12,
    "marital_status":"Married-civ-spouse", 
    "occupation": "Prof-specialty", 
    "relationship": "Wife" , 
    "race": "White", 
    "gender": "Female",
    "capital_gain": 0, 
    "capital_loss":0, 
    "hours_per_week":30, 
    "native_country": "cuba"
}}

result = requests.post(url, json=data).json()
print(result)
