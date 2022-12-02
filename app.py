import pandas as pd
from flask import Flask, render_template,request
app = Flask(__name__)
from sqlalchemy import create_engine
import pickle,joblib


impute = joblib.load('meanimpute')
winsor = joblib.load('winzor')
poly_model = pickle.load(open('poly_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template("index.html")
@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        data = pd.read_csv(f)
        user = request.form["user"]# user
        pw = request.form["password"] # passwrd
        db = request.form["Database"] #database


        engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
                        
        clean1 = pd.DataFrame(impute.transform(data), columns = data.select_dtypes(exclude = ['object']).columns)
        
        clean2 = pd.DataFrame(winsor.transform(clean1), columns = clean1.columns)
        
        prediction = pd.DataFrame(poly_model.predict(clean2), columns = ['Pred_AT'])
        
        final = pd.concat([prediction, data], axis = 1)
        final.to_sql('mpg_predictons', con = engine, if_exists = 'replace', chunksize = 1000, index= False)
        return render_template("new.html", Z = "Your results are here" , Y = final.to_html(justify='center').replace('<table border="1" class="dataframe">','<table border="1" class="dataframe" bordercolor="#000000" bgcolor="#FFCC66">'))

if __name__ == '__main__':

    app.run(debug = True)

