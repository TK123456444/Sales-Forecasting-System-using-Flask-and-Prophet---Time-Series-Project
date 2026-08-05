import plotly
import plotly.graph_objs as go
from flask import Flask
import pandas as pd
import prophet
from flask import Flask,render_template,redirect,request
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from random import randint
import plotly.graph_objs as go
import plotly.offline as py

import os
app = Flask("__name__")
app.config["IMAGE_UPLOADS"] = "static/img/"
@app.route('/')
def hello():
    return render_template("step1.html")




@app.route("/home")
def home():
    return redirect('/')


@app.route("/", methods=["POST"])
def submit_data():
    try:
        # ==========================
        # File Validation
        # ==========================
        f = request.files.get("userfile")

        if f is None or f.filename == "":
            return render_template(
                "step1.html",
                error="Please upload a CSV file."
            )

        os.makedirs("uploads", exist_ok=True)

        filepath = os.path.join("uploads", f.filename)
        f.save(filepath)

        # ==========================
        # Read Form Data
        # ==========================
        s1 = request.form.get("query1", "").strip()
        s2 = request.form.get("query2", "").strip()
        s3 = request.form.get("query3", "").strip()
        s4 = request.form.get("query4", "").strip()

        if s1 == "" or s2 == "" or s3 == "" or s4 == "":
            return render_template(
                "step1.html",
                error="Please fill all fields."
            )

        try:
            t = int(s3)
        except ValueError:
            return render_template(
                "step1.html",
                error="Forecast Period must be an integer."
            )

        # ==========================
        # Read CSV
        # ==========================
        df = pd.read_csv(filepath)

        if s1 not in df.columns:
            return render_template(
                "step1.html",
                error=f"Column '{s1}' not found."
            )

        if s2 not in df.columns:
            return render_template(
                "step1.html",
                error=f"Column '{s2}' not found."
            )

        # ==========================
        # Prophet Format
        # ==========================
        df = df.rename(columns={s1: "ds", s2: "y"})

        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

        df = df.dropna()

        df = df.sort_values("ds")

        if (df["y"] <= 0).any():
            return render_template(
                "step1.html",
                error="Target column contains zero or negative values."
            )

        df["y_orig"] = df["y"]

        df["y"] = np.log(df["y"])

        # ==========================
        # Train Model
        # ==========================
        model = Prophet()

        model.fit(df)

        future = model.make_future_dataframe(
            periods=t,
            freq=s4
        )

        forecast = model.predict(future)

        forecast["yhat"] = np.exp(forecast["yhat"])
        forecast["yhat_lower"] = np.exp(forecast["yhat_lower"])
        forecast["yhat_upper"] = np.exp(forecast["yhat_upper"])

        result = forecast[["ds", "yhat"]].tail(t)

        result = result.rename(
            columns={
                "ds": "Date",
                "yhat": "Forecast"
            }
        )

        # ==========================
        # Plot
        # ==========================
        fig, ax = plt.subplots(figsize=(10,5))

        ax.plot(df["y_orig"], label="Actual")

        ax.plot(forecast["yhat"][:len(df)], label="Predicted")

        ax.legend()

        os.makedirs(app.config["IMAGE_UPLOADS"], exist_ok=True)

        image_name = str(randint(10000,999999)) + "_forecast.png"

        image_path = os.path.join(
            app.config["IMAGE_UPLOADS"],
            image_name
        )

        fig.savefig(image_path)

        plt.close(fig)

        return render_template(
            "step1.html",
            user_image=image_path.replace("\\","/"),
            tables=[result.to_html(classes="forecast", index=False)],
            titles=["Forecast"],
            query1=s1,
            query2=s2,
            query3=s3,
            query4=s4
        )

    except Exception as e:
        return render_template(
            "step1.html",
            error=str(e)
        )
        
    future_data = model.make_future_dataframe(periods=t, freq = s4)

    future_data
    forecast_data = model.predict(future_data)

    
    forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
    
    model.plot(forecast_data) 
    model.plot_components(forecast_data)
    forecast_data_orig = forecast_data # make sure we save the original forecast data
    forecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])
    forecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])
    forecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])
    model.plot(forecast_data_orig)
    df['y_log']=df['y'] #copy the log-transformed data to another column
    df['y']=df['y_orig']
    final_df = pd.DataFrame(forecast_data_orig)
    
    final_df_1=final_df[['ds','yhat']].tail(t)
    final_df_1 = final_df_1.rename(columns={'yhat': 'Sales', 'ds':'Month'})
    

    #rmse = mean_squared_error(df["y_orig"].iloc[24:], final_df['yhat'].iloc[24:36])**0.5
    #print('Test MSE: %.3f' % rmse)
                
      
    fig,ax=plt.subplots(nrows=1, ncols=1)
    ax.plot(df["y_orig"],label="Actual")
    ax.plot(final_df["yhat"],label="Predicted")
    ax.legend()

    #plt.xticks(rotation=90)
    #plt.show()
    n=randint(0,1000000000000)
    n=str(n)
    fig.savefig(os.path.join(app.config["IMAGE_UPLOADS"],n+'time_series.png'))  
    full_filename= os.path.join(app.config["IMAGE_UPLOADS"],n+'time_series.png')
    
    
    return render_template('step1.html',user_image = full_filename,tables=[final_df_1.to_html(classes='forecast')],titles=['na','forecast'],query1 = request.form['query1'],query2 = request.form['query2'],query3 = request.form['query3'], query4 = request.form['query4'])
     
'''
    import plotly.graph_objs as go
    import plotly.offline as py
    #Plot predicted and actual line graph with X=dates, Y=Outbound
    actual_chart = go.Scatter(y=df["y_orig"], name= 'Actual')
    predict_chart = go.Scatter(y=final_df["yhat"], name= 'Predicted')
    predict_chart_upper = go.Scatter(y=final_df["yhat_upper"], name= 'Predicted Upper')
    predict_chart_lower = go.Scatter(y=final_df["yhat_lower"], name= 'Predicted Lower')
    #py.plot([actual_chart, predict_chart, predict_chart_upper, predict_chart_lower])
    py.plot([actual_chart, predict_chart, predict_chart_upper, predict_chart_lower], filename = 'templates/' +'filename.html', auto_open=False, image_width=200, image_height=200)
    
'''
    #return render_template('step1.html',user_image = full_filename,tables=[final_df_1.to_html(classes='forecast')],titles=['na','forecast'],query1 = request.form['query1'],query2 = request.form['query2'],query3 = request.form['query3'], query4 = request.form['query4'])
    
   
if __name__ =="__main__":
    app.run()
    
