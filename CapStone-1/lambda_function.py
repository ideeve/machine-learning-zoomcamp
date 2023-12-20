import pickle

with open('model_logistic.bin', 'rb') as f:
   model, dv = pickle.load(f)



def predict(person):
    X = dv.transform([person])
    y_pred = model.predict(X)
    prediction = ' <=50K' if y_pred[0] == 1 else ' >50K'
    if prediction == " <=50K":
        
        return('Earns less than 50k')
    else:
        return('Earns more than 50k.')


def lamdba_handler(event, context):
    person = event["person"]
    result = predict(person)
    return result
    