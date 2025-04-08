from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline


application = Flask(__name__)

app = application


@app.route('/')
def home_page():
     return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])

def predict_datapoint():
     # render the page
     if request.method == 'GET':
          return render_template('form.html')
     
     else:
          data = CustomData(
               Amount=float(request.form.get('Amount')),
               Campaign_Name=request.form.get('Campaign_Name'),
               Sub_Id=request.form.get('Sub_Id'),
               Partner=request.form.get('Partner')
          )
          final_new_data = data.get_data_as_dataframe()
          predict_pipeline = PredictPipeline()
          pred = predict_pipeline.predict(final_new_data)
          results = round(pred[0], 2)
          return render_template('results.html', final_result = results)
          
     



if __name__ == "__main__":
     app.run(host = '0.0.0.0', debug=True)