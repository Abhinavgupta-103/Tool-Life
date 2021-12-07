from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    float_features=[float(x) for x in request.form.values()]
    final=[np.array(float_features)]
    print(float_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('index.html',pred='Change tool.\n{}'.format('High'),bhai="kuch karna hain iska ab?")
    else:
        return render_template('index.html',pred='Tool is working fine.\n {}'.format('Low'),bhai="Your Forest is Safe for now")

# @app.route('/predict',methods=['POST','GET'])
# def predict():
#     # float_features=[int(x) for x in request.form.values()]
#     float_features=[0,0,0]
#     final=[np.array(float_features)]
#     print(float_features)
#     print(final)
#     # print("Hello")
#     prediction=model.predict_proba(final)
#     output='{0:.{1}f}'.format(prediction[0][1], 2)
#     if output>str(0.5):
#         return render_template('forest_fire.html',pred='Your Tools needs to change(High).',bhai="kuch karna hain iska ab?")
#     else:
#         return render_template('forest_fire.html',pred='Your Tool is Fine(Normal).'.format(output),bhai="Your Forest is Safe for now")


if __name__ == '__main__':
    app.run(debug=True)
