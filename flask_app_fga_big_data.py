
# # A very simple Flask Hello World app for you to get started with...

# from flask import Flask

# app = Flask(__name__)

# @app.route('/')
# def hello_world():
#     return 'Hello from Flask!'


from flask import Flask,render_template,flash, redirect,url_for,session,logging,request,jsonify
# from flask_ngrok import run_with_ngrok # Alternatif Ngrok => Heroku / PythonAnywhere / https://labs.play-with-docker.com / AWS / GCP / Azure/ eval_js dari Colab (agak terbatas) / etc
import sqlite3
# from flask_cors import CORS


# try:

# from flask import Flask

from flask import send_file
from io import BytesIO

from flask_wtf.file import FileField
from wtforms import SubmitField
# from flask_wtf import Form
from flask_wtf import FlaskForm
# import sqlite3
# print("All Modules Loaded .... ")
# except:
#     print (" Some Module are missing ...... ")

# from flask.ext.storage import get_default_storage_class
# from flask.ext.uploads import delete, init, save, Upload

# import subprocess
# import sys

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# try:
#     from flask_cors import CORS  # The typical way to import flask-cors
# except ImportError:

#     # import subprocess
#     # process = subprocess.Popen(['ls', '-a'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     # out, err = process.communicate()
#     # print(out)

#     install("flask_cors")

#     # Path hack allows examples to be run without installation.
#     import os
#     parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     os.sys.path.insert(0, parentdir)

#     from flask_cors import CORS



app = Flask(__name__, static_folder='static')
# run_with_ngrok(app)  # Start ngrok when app is run
# CORS(app)
# CORS(app, resources=r'/api/*')

app.debug = True
app.secret_key = 'key_big_data_app'

@app.route("/")
def index():
    return render_template("index.html")
    # return "Hello Brother"

@app.route("/login",methods=["GET", "POST"])
def login():
  conn = connect_db()
  db = conn.cursor()
  #conn = sqlite3.connect('fga_big_data_rev2.db')
  #db = conn.cursor()
  msg = ""
  if request.method == "POST":
      mail = request.form["mail"]
      passw = request.form["passw"]

      rs = db.execute("SELECT * FROM user WHERE Mail=\'"+ mail +"\'"+" AND Password=\'"+ passw+"\'" + " LIMIT 1")

      conn.commit()

      hasil = []
      for v_login in rs:
         hasil.append(v_login)

      if hasil:
        session['name'] = v_login[3]
        return redirect(url_for("bigdataApps"))
        # return redirect('/bigdataApps')
        # return render_template("bigdataApps.html")
      else:
        msg = "Masukkan Username (Email) dan Password dgn Benar!"

  return render_template("login.html", msg = msg)

@app.route("/register", methods=["GET", "POST"])
def register():
  conn = connect_db()
  db = conn.cursor()
  #conn = sqlite3.connect('fga_big_data_rev2.db')
  #db = conn.cursor()
  if request.method == "POST":
      mail = request.form['mail']
      uname = request.form['uname']
      passw = request.form['passw']

      cmd = "insert into user(Mail, Password,Name,Level) values('{}','{}','{}','{}')".format(mail,passw,uname,'1')
      conn.execute(cmd)
      conn.commit()

      # conn = db

      return redirect(url_for("login"))
  return render_template("register.html")


@app.route("/NRC", methods=["GET", "POST"])
def NRC():
  import numpy as np
  import pandas as pd
  import os.path
  import joblib
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  url = os.path.join(BASE_DIR, "reall.csv")
  dataset = pd.read_csv(url)
  X = dataset.iloc[:, :-1].values
  y = dataset.iloc[:, 1].values
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)

  from sklearn.linear_model import LinearRegression
  regressor = LinearRegression()
  regressor.fit(X_train, y_train)
  myModelReg = regressor.fit(X_train, y_train)
  with open (os.path.join(BASE_DIR, "real.joblib.pkl"), 'wb') as f: joblib.dump(myModelReg, f, compress=9)
  with open (os.path.join(BASE_DIR, "real.joblib.pkl"), 'rb') as f : myModelReg_load = joblib.load(f)
  y_pred = myModelReg_load.predict(X_test)
  y_pred2 = myModelReg_load.predict(X_train)
  aktual, predict = y_train, y_pred2
  mape = np.sum(np.abs(((aktual - predict)/aktual)*100))/len(predict)
  return render_template('NRC.html', y_aktual = list(y_train), y_prediksi = list(y_pred2), mape = mape)

@app.route("/NSC", methods=["GET", "POST"])
def NSC():
  import numpy as np
  import pandas as pd
  import os.path
  import joblib
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  url = os.path.join(BASE_DIR, "simulasii.csv")
  dataset = pd.read_csv(url)
  X = dataset.iloc[:, :-1].values
  y = dataset.iloc[:, 1].values
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)

  from sklearn.linear_model import LinearRegression
  regressor = LinearRegression()
  regressor.fit(X_train, y_train)
  myModelReg = regressor.fit(X_train, y_train)
  with open (os.path.join(BASE_DIR, "simulasi.joblib.pkl"), 'wb') as f: joblib.dump(myModelReg, f, compress=9)
  with open (os.path.join(BASE_DIR, "simulasi.joblib.pkl"), 'rb') as f : myModelReg_load = joblib.load(f)
  y_pred = myModelReg_load.predict(X_test)
  y_pred2 = myModelReg_load.predict(X_train)
  aktual, predict = y_train, y_pred2
  mape = np.sum(np.abs(((aktual - predict)/aktual)*100))/len(predict)
  return render_template('NSC.html', y_aktual = list(y_train), y_prediksi = list(y_pred2), mape = mape)

@app.route("/PSC", methods=["GET", "POST"])
def PSC():
  # MLLIB dari Pyspark Simple Linear Regression /Klasifikasi / Clustering
  # Importing the libraries
  import numpy as np
  import matplotlib.pyplot as plt
  import pandas as pd
  import os

  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  url = os.path.join(BASE_DIR, "simulasii.csv")

  import findspark
  findspark.init()
#   import findspark
#   findspark.init("/home/imamcs/spark-3.1.2-bin-hadoop3.2")

  #os.environ["SPARK_HOME"] ="/home/imamcs/mysite/FGA-Big-Data-Using-Python-Filkom-x-Mipa-UB-2021/spark-2.0.0-bin-hadoop2.7"
  #os.environ["PYTHONPATH"] ="/home/imamcs/mysite/FGA-Big-Data-Using-Python-Filkom-x-Mipa-UB-2021/spark-2.0.0-bin-hadoop2.7/python/"
  #os.environ["PYTHONPATH"] +=":/home/imamcs/mysite/FGA-Big-Data-Using-Python-Filkom-x-Mipa-UB-2021/spark-2.0.0-bin-hadoop2.7/python/lib/py4j-0.10.1-src.zip"

  #os.environ["PYTHONPATH"] = "/spark-2.4.1-bin-hadoop2.7/python/"
  #os.environ["PYTHONPATH"] += ":/spark-2.4.1-bin-hadoop2.7/python/lib/py4j-0.10.7-src.zip"

#   print(os.environ["SPARK_HOME"])

#   print(os.environ["PYTHONPATH"])

#   !echo $PYTHONPATH

  from pyspark.sql import SparkSession
  spark = SparkSession.builder.appName("Linear Regression Model").getOrCreate()

  from pyspark.ml.regression import LinearRegression
  from pyspark.ml.linalg import Vectors
  from pyspark.ml.feature import VectorAssembler
  from pyspark.ml.feature import IndexToString, StringIndexer

  from pyspark import SQLContext, SparkConf, SparkContext
  from pyspark.sql import SparkSession
  sc = SparkContext.getOrCreate()
  if (sc is None):
      sc = SparkContext(master="local[*]", appName="Linear Regression")
  spark = SparkSession(sparkContext=sc)
  # sqlcontext = SQLContext(sc)

  # Importing the dataset => ganti sesuai dengan case yg anda usulkan
  # a. Min. 30 Data dari case data simulasi dari yg Anda usulkan
  # b. Min. 30 Data dari real case, sesuai dgn yg Anda usulkan dari tugas minggu ke-3 (dari Kaggle/UCI Repository)
  # url = "./Salary_Data.csv"

  sqlcontext = SQLContext(sc)
  data = sqlcontext.read.csv(url, header = True, inferSchema = True)

  from pyspark.ml.feature import VectorAssembler
  # mendifinisikan Salary sebagai variabel label/predictor
  dataset = data.select(data.usia , data.berat,  data.tinggi, data.leher, data.dada, data.perut, data.pinggul, data.paha, data.lutut, data.pk, data.bisep, data.lb, data.pt, data.lemak.alias('label'))
  # split data menjadi 70% training and 30% testing
  training, test = dataset.randomSplit([0.7, 0.3], seed = 100)
  # mengubah fitur menjadi vektor
  assembler = VectorAssembler().setInputCols(['usia','berat','tinggi','dada', 'perut', 'pinggul', 'paha', 'lutut','pk','bisep','lb','pt',]).setOutputCol('features')
  trainingSet = assembler.transform(training)
  # memilih kolom yang akan di vektorisasi
  trainingSet = trainingSet.select("features","label")

  from pyspark.ml.regression import LinearRegression
  # fit data training ke model
  lr = LinearRegression()
  lr_Model = lr.fit(trainingSet)
  # assembler : fitur menjadi vektor
  testSet = assembler.transform(test)
  # memilih kolom fitur dan label
  testSet = testSet.select("features", "label")
  # fit testing data ke model linear regression
  testSet = lr_Model.transform(testSet)
  # testSet.show(truncate=False)

  from pyspark.ml.evaluation import RegressionEvaluator
  evaluator = RegressionEvaluator()
  # print(evaluator.evaluate(testSet, {evaluator.metricName: "r2"}))

  y_pred2 = testSet.select("prediction")
  # y_pred2.show()


  return render_template('PSC.html', y_aktual = y_pred2.rdd.flatMap(lambda x: x).collect(), y_prediksi = y_pred2.rdd.flatMap(lambda x: x).collect(), mape = evaluator.evaluate(testSet, {evaluator.metricName: "r2"}))

@app.route("/PRC", methods=["GET", "POST"])
def PRC():
  # MLLIB dari Pyspark Simple Linear Regression /Klasifikasi / Clustering
  # Importing the libraries
  import numpy as np
  import matplotlib.pyplot as plt
  import pandas as pd
  import os

  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  url = os.path.join(BASE_DIR, "reall.csv")

  import findspark
  findspark.init()
#   import findspark
#   findspark.init("/home/imamcs/spark-3.1.2-bin-hadoop3.2")

  #os.environ["SPARK_HOME"] ="/home/imamcs/mysite/FGA-Big-Data-Using-Python-Filkom-x-Mipa-UB-2021/spark-2.0.0-bin-hadoop2.7"
  #os.environ["PYTHONPATH"] ="/home/imamcs/mysite/FGA-Big-Data-Using-Python-Filkom-x-Mipa-UB-2021/spark-2.0.0-bin-hadoop2.7/python/"
  #os.environ["PYTHONPATH"] +=":/home/imamcs/mysite/FGA-Big-Data-Using-Python-Filkom-x-Mipa-UB-2021/spark-2.0.0-bin-hadoop2.7/python/lib/py4j-0.10.1-src.zip"

  #os.environ["PYTHONPATH"] = "/spark-2.4.1-bin-hadoop2.7/python/"
  #os.environ["PYTHONPATH"] += ":/spark-2.4.1-bin-hadoop2.7/python/lib/py4j-0.10.7-src.zip"

#   print(os.environ["SPARK_HOME"])

#   print(os.environ["PYTHONPATH"])

#   !echo $PYTHONPATH

  from pyspark.sql import SparkSession
  spark = SparkSession.builder.appName("Linear Regression Model").getOrCreate()

  from pyspark.ml.regression import LinearRegression
  from pyspark.ml.linalg import Vectors
  from pyspark.ml.feature import VectorAssembler
  from pyspark.ml.feature import IndexToString, StringIndexer

  from pyspark import SQLContext, SparkConf, SparkContext
  from pyspark.sql import SparkSession
  sc = SparkContext.getOrCreate()
  if (sc is None):
      sc = SparkContext(master="local[*]", appName="Linear Regression")
  spark = SparkSession(sparkContext=sc)
  # sqlcontext = SQLContext(sc)

  # Importing the dataset => ganti sesuai dengan case yg anda usulkan
  # a. Min. 30 Data dari case data simulasi dari yg Anda usulkan
  # b. Min. 30 Data dari real case, sesuai dgn yg Anda usulkan dari tugas minggu ke-3 (dari Kaggle/UCI Repository)
  # url = "./Salary_Data.csv"

  sqlcontext = SQLContext(sc)
  data = sqlcontext.read.csv(url, header = True, inferSchema = True)

  from pyspark.ml.feature import VectorAssembler
  # mendifinisikan Salary sebagai variabel label/predictor
  dataset = data.select(data.usia , data.berat,  data.tinggi, data.leher, data.dada, data.perut, data.pinggul, data.paha, data.lutut, data.pk, data.bisep, data.lb, data.pt, data.lemak.alias('label'))
  # split data menjadi 70% training and 30% testing
  training, test = dataset.randomSplit([0.7, 0.3], seed = 100)
  # mengubah fitur menjadi vektor
  assembler = VectorAssembler().setInputCols(['usia','berat','tinggi','dada', 'perut', 'pinggul', 'paha', 'lutut','pk','bisep','lb','pt',]).setOutputCol('features')
  trainingSet = assembler.transform(training)
  # memilih kolom yang akan di vektorisasi
  trainingSet = trainingSet.select("features","label")

  from pyspark.ml.regression import LinearRegression
  # fit data training ke model
  lr = LinearRegression()
  lr_Model = lr.fit(trainingSet)
  # assembler : fitur menjadi vektor
  testSet = assembler.transform(test)
  # memilih kolom fitur dan label
  testSet = testSet.select("features", "label")
  # fit testing data ke model linear regression
  testSet = lr_Model.transform(testSet)
  # testSet.show(truncate=False)

  from pyspark.ml.evaluation import RegressionEvaluator
  evaluator = RegressionEvaluator()
  # print(evaluator.evaluate(testSet, {evaluator.metricName: "r2"}))

  y_pred2 = testSet.select("prediction")
  # y_pred2.show()


  return render_template('PRC.html', y_aktual = y_pred2.rdd.flatMap(lambda x: x).collect(), y_prediksi = y_pred2.rdd.flatMap(lambda x: x).collect(), mape = evaluator.evaluate(testSet, {evaluator.metricName: "r2"}))

@app.route("/bigdataApps", methods=["GET", "POST"])
def bigdataApps():
  if request.method == 'POST':
    import pandas as pd
    import numpy as np
    import os.path

    #BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #url = os.path.join(BASE_DIR, "dataset_dump.csv")

    dataset = request.files['inputDataset']
    # url = "./dataset_dump.csv"

    persentase_data_training = 90
    banyak_fitur = int(request.form['banyakFitur'])
    banyak_hidden_neuron = int(request.form['banyakHiddenNeuron'])
    dataset = pd.read_csv(dataset, delimiter=';', names = ['Tanggal', 'Harga'], usecols=['Harga'])
    # dataset = pd.read_csv(url, delimiter=';', names = ['Tanggal', 'Harga'], usecols=['Harga'])
    # dataset = dataset.fillna(method='ffill')
    minimum = int(dataset.min()-10000)
    maksimum = int(dataset.max()+10000)
    new_banyak_fitur = banyak_fitur + 1
    hasil_fitur = []
    for i in range((len(dataset)-new_banyak_fitur)+1):
      kolom = []
      j = i
      while j < (i+new_banyak_fitur):
        kolom.append(dataset.values[j][0])
        j += 1
      hasil_fitur.append(kolom)
    hasil_fitur = np.array(hasil_fitur)
    data_normalisasi = (hasil_fitur - minimum)/(maksimum - minimum)
    data_training = data_normalisasi[:int(persentase_data_training*len(data_normalisasi)/100)]
    data_testing = data_normalisasi[int(persentase_data_training*len(data_normalisasi)/100):]

    #Training
    bobot = np.random.rand(banyak_hidden_neuron, banyak_fitur)
    bias = np.random.rand(banyak_hidden_neuron)
    h = 1/(1 + np.exp(-(np.dot(data_training[:, :banyak_fitur], np.transpose(bobot)) + bias)))
    h_plus = np.dot(np.linalg.inv(np.dot(np.transpose(h),h)),np.transpose(h))
    output_weight = np.dot(h_plus, data_training[:, banyak_fitur])

    #Testing
    h = 1/(1 + np.exp(-(np.dot(data_testing[:, :banyak_fitur], np.transpose(bobot)) + bias)))
    predict = np.dot(h, output_weight)
    predict = predict * (maksimum - minimum) + minimum

    #MAPE
    aktual = np.array(hasil_fitur[int(persentase_data_training*len(data_normalisasi)/100):, banyak_fitur])
    mape = np.sum(np.abs(((aktual - predict)/aktual)*100))/len(predict)

    print("predict = ", predict)
    print("aktual =", aktual)
    print("mape = ", mape)

    # return render_template('bigdataApps.html', data = {'y_aktual' : list(aktual),'y_prediksi' : list(predict),'mape' : mape})
    return render_template('bigdataApps.html', y_aktual = list(aktual), y_prediksi = list(predict), mape = mape)


    # return "Big Data Apps " + str(persentase_data_training) + " banyak_fitur = " + str(banyak_fitur) + " banyak_hidden_neuron = " + str(banyak_hidden_neuron) + " :D"
  else:
    return render_template('bigdataApps.html')

def connect_db():
    import os.path

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "fga_big_data_rev2.db")
    # with sqlite3.connect(db_path) as db:

    return sqlite3.connect(db_path)


# cara akses, misal: http://imamcs.pythonanywhere.com/api/fp/3.0/?a=90&b=3&c=2
@app.route("/api", methods=["GET", "POST"])
# @cross_origin()
def api():
    import os.path
    import sys

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    url = os.path.join(BASE_DIR, "reall.csv")

    # url = "../GGRM.JK.csv"
    # dataset=pd.read_csv(url)

    import pandas as pd
    import numpy as np
    import json
    # from django.http import HttpResponse
    from flask import Response


     # print(a,' ',b,' ',c)
    # bar = request.args.to_dict()
    # print(bar)

    # dataset = request.FILES['inputDataset']#'E:/Pak Imam/Digitalent/dataset_dump.csv'

    # print(persentase_data_training,banyak_fitur,banyak_hidden_neuron)
    dataset = pd.read_csv(url)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred2 = regressor.predict(X_train)

    # MAPE
    aktual, predict = y_train, y_pred2
    mape = np.sum(np.abs(((aktual - predict)/aktual)*100))/len(predict)

    response = jsonify({'y_aktual': list(y_train), 'y_prediksi': list(y_pred2), 'mape': mape})


    # Enable Access-Control-Allow-Origin
    response.headers.add("Access-Control-Allow-Origin", "*")
    # response.headers.add("access-control-allow-credentials","false")
    # response.headers.add("access-control-allow-methods","GET, POST")


    # r = Response(response, status=200, mimetype="application/json")
    # r.headers["Content-Type"] = "application/json; charset=utf-8"
    return response



# get json data from a url using flask in python
@app.route('/baca_api', methods=["GET"])
def baca_api():
    import requests
    import json
    # uri = "https://api.stackexchange.com/2.0/users?order=desc&sort=reputation&inname=fuchida&site=stackoverflow"
    uri = "http://hanifa.pythonanywhere.com/api"
    # try:
    #     uResponse = requests.get(uri)
    # except requests.ConnectionError:
    #     return "Terdapat Error Pada Koneksi Anda"
    # Jresponse = uResponse.text
    # data = json.loads(Jresponse)

    # json.loads(response.get_data().decode("utf-8"))
    data = json.loads(requests.get(uri).decode("utf-8"))
    # data = json.loads(response.get(uri).get_data().decode("utf-8"))

    # import urllib.request
    # with urllib.request.urlopen("http://imamcs.pythonanywhere.com/api/fp/3.0/?a=90&b=3&c=2") as url:
    #     data = json.loads(url.read().decode())
    #     #print(data)

    # from urllib.request import urlopen

    # import json
    # import json
    # store the URL in url as
    # parameter for urlopen
    # url = "https://api.github.com"

    # store the response of URL
    # response = urlopen(url)

    # storing the JSON response
    # from url in data
    # data_json = json.loads(response.read())

    # print the json response
    # print(data_json)

    # data = \
    #     {
    #   "items": [
    #     {
    #       "badge_counts": {
    #         "bronze": 16,
    #         "silver": 4,
    #         "gold": 0
    #       },
    #       "account_id": 258084,
    #       "is_employee": false,
    #       "last_modified_date": 1573684556,
    #       "last_access_date": 1628710576,
    #       "reputation_change_year": 0,
    #       "reputation_change_quarter": 0,
    #       "reputation_change_month": 0,
    #       "reputation_change_week": 0,
    #       "reputation_change_day": 0,
    #       "reputation": 420,
    #       "creation_date": 1292207782,
    #       "user_type": "registered",
    #       "user_id": 540028,
    #       "accept_rate": 100,
    #       "location": "Minneapolis, MN, United States",
    #       "website_url": "http://fuchida.me",
    #       "link": "https://stackoverflow.com/users/540028/fuchida",
    #       "profile_image": "https://i.stack.imgur.com/kP5GW.png?s=128&g=1",
    #       "display_name": "Fuchida"
    #     }
    #   ],
    #   "has_more": false,
    #   "quota_max": 300,
    #   "quota_remaining": 299
    # }

    # displayName = data['items'][0]['display_name']# <-- The display name
    # reputation = data['items'][0]['reputation']# <-- The reputation

    # y_train = data['y_aktual']
    # y_pred = data['y_prediksi']
    # mape = data['mape']

    return data
    # return str(mape)
    # return render_template('MybigdataAppsNonPySpark.html', y_aktual = list(y_train), y_prediksi = list(y_pred), mape = mape)



@app.route('/logout')
def logout():
   # remove the name from the session if it is there
   session.pop('name', None)
   return redirect(url_for('index'))

if __name__ == '__main__':
    #import os
    #os.environ["JAVA_HOME"] ="/usr/lib/jvm/java-8-openjdk-amd64"
    #print(os.environ["JAVA_HOME"])
    #print(os.environ["SPARK_HOME"])
    #print(os.environ["PYTHONPATH"])
    # db.create_all()
    app.run()
    # If address is in use, may need to terminate other sessions:
            # Runtime > Manage Sessions > Terminate Other Sessions
  # app.run(host='0.0.0.0', port=5004)  # If address is in use, may need to terminate other sessions:
             # Runtime > Manage Sessions > Terminate Other Sessions
