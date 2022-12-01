from flask import Flask
from flask import jsonify
import trainermodel as tr
import numpy as np
import pickle
model=pickle.load(open('model.pkl','rb'))
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def welcome():
    return "Hello World!"
@app.route('/[0,0,0,6]/')
def success():
    return ('You have perfectly typed [0,0,0,6]')


@app.route('/predict/<inputs>')
def show_post(inputs):
    inputs=str(inputs)
    inputs=inputs.split('_')
    symptoms = tr.X.columns.values
    symptom_index={}
    for index, value in enumerate(symptoms):
        symptom = value
        symptom_index[index] = value
    
    data_dict = {
        
        "predictions_classes":tr.encoder.classes_,
        "symptom_index":symptom_index
    }
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in inputs:
        index = int(symptom)
        input_data[index] = 1
        
    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
    final_pred=model.predict(input_data)[0]
    return jsonify(code=str(final_pred),
    name=tr.name_maper[final_pred]
    )   


    
    
    


    
    
    #use post title to fetch the record from db
@app.route('/numbers/')
def print_list():
    return jsonify(list(range(5)))    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)