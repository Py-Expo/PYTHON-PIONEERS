from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load the necessary data
df = pd.read_csv("datas.csv")
rain = pd.read_csv("Rainfall Predicted.csv")

# Preprocess the data
data = df.dropna()
s = list(data["Season"].unique())

@app.route('/')
def index():
    return render_template('index.html')
   
@app.route('/predict', methods=['POST'])
def predict():
    district_name = request.form['districtName']
    season = request.form['season']
    area_in_hectares = float(request.form['area'])
    # Assuming you have the dropdown selection in your HTML form
    data_type = request.form['dropdown']
    # Filter data based on user input
    data_cu = data[(data["District_Name"] == district_name.upper()) & (data["Season"] == season.title())]

    # Prepare the data for prediction
    data1 = data_cu.drop(["State_Name","Crop_Year"], axis=1)
    data_dum = pd.get_dummies(data1)
    x = data_dum.drop("Production", axis=1)
    y = data_dum[["Production"]]

    # Train the model
    model = RandomForestRegressor()
    model.fit(x, y.values.ravel())

    # Prepare input data for prediction
    ch = pd.DataFrame()
    for crop in list(data_cu["Crop"].unique()):
        t = (x[x["Crop_{}".format(crop)] == 1])[:1]
        ch = pd.concat([ch, t])

    ch["Area"] = area_in_hectares
    ch["Rainfall"] = float(rain[rain["State_Name"] == data_cu['State_Name'].iloc[0]]["Rainfall"])
    
    # Make prediction
    predict = model.predict(ch)
    
    # Prepare response
    crname = data_cu["Crop"].iloc[ch.index]
    crpro = pd.DataFrame({'Crop': crname, 'Production': predict})
    crpro = crpro.sort_values(by='Production', ascending=False)

    # Convert response to JSON
    result = crpro.head(5).to_dict(orient='records')
    

if __name__ == '__main__':
    app.run(debug=True)