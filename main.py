import pickle
from pathlib import Path
import numpy as np
from fastapi import FastAPI



app = FastAPI()

@app.get("/my-first-api")
def hello(name = None, a = 2, b=3.0):

    if name is None:
        text = 'Hello!'

    else:
        text = 'Hello ' + name + '!'

    s = text + str(a), str(type(a)) + str(b), str(type(b))
    res = s

    return text

def test(a):
    print('test', a)
    return 'c'

@app.get("/my-second-api")
def hello(name = None, a = 2, b=3.0):

    if name is None:
        text = 'Hello!'

    else:
        text = 'Hello ' + name + '!'

    s = text + str(a), str(type(a)) + str(b), str(type(b))
    res = test(a)


    return res


# Config
max_woz = 599289.0
model_file = Path('models/model_lin.pkl')

def preprocess(area, married_no_kids, married_with_kids, not_married_no_kids, not_married_with_kids, other, single, single_parent, total,
               percentage_bebouwd, percentage_erf, percentage_groen,
               percentage_land, percentage_terrein, percentage_wegen):

    # Collect preprocessed inputs in list
    model_inputs = []

    # Preprocess fca inputs --> Population to Ratio of total residents
    fca_inputs = [married_no_kids, married_with_kids, not_married_no_kids, not_married_with_kids, other, single, single_parent]
    for input in fca_inputs:
        model_inputs.append(float(input) / float(total))

    fca_inputs = [percentage_bebouwd, percentage_erf, percentage_groen,
                  percentage_land, percentage_terrein, percentage_wegen]
    for input in fca_inputs:
        model_inputs.append(float(input) / 100)


    # Process Area data: One-Hot Encode district
    for d in ['A', 'E', 'F', 'K', 'M', 'N', 'T']:
        model_inputs.append(d==area[0])

    return np.array(model_inputs, dtype=np.float64)


def run_inference(model_inputs: list):
    if not model_file.exists():
        error_msg = f'Model path does not exist: {model_file} from {Path.cwd()}'
        return error_msg
    # Test Save by Loading
    with open(model_file, 'rb') as file:
        model = pickle.load(file)

    # Preprocessing -> scaling, extracting area code and create hot encoding
    model_inputs = preprocess(*model_inputs)

    # pass [model_inputs] to get 2D input of shape (n_datapoints=1, n_parameters=20)
    prediction = model.predict([model_inputs]) * max_woz
    return prediction[0]


@app.get("/get_woz_value")
def run_predictions(
        area,
        married_no_kids,
        married_with_kids,
        not_married_no_kids,
        not_married_with_kids,
        other,
        single,
        single_parent,
        total,
        percentage_bebouwd,
        percentage_erf,
        percentage_groen,
        percentage_land,
        percentage_terrein,
        percentage_wegen,
):
    model_inputs = [area,
                    married_no_kids,
                    married_with_kids,
                    not_married_no_kids,
                    not_married_with_kids,
                    other,
                    single,
                    single_parent,
                    total,
                    percentage_bebouwd,
                    percentage_erf,
                    percentage_groen,
                    percentage_land,
                    percentage_terrein,
                    percentage_wegen, ]
    predicted = run_inference(model_inputs)

    return predicted
