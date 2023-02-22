from flask import Flask, render_template, request
import social_ads_model
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])

def basic():
    if request.method == 'POST':
        Gender = request.form['Gender']
        Age	 = request.form['Age']
        EstimatedSalary = request.form['EstimatedSalary']
        y_pred = [[Gender,Age,EstimatedSalary]]
        trained_model = social_ads_model.training_model()
        prediction_value = trained_model.predict(y_pred)
        Purchased= 'Customer purchased!'
        Not_purchased = 'Not purchased!'
        if prediction_value == 0:
            return render_template('index.html', Purchased=Purchased)
        elif prediction_value == 1:
            return render_template('index.html',Not_purchased=Not_purchased)
        
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)