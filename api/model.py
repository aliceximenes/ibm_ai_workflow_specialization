import time,os,re,csv,sys,uuid,joblib
from datetime import date
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from fbprophet import Prophet

from utils.logger import update_predict_log, update_train_log
from utils.ingestion_process import fetch_ts, engineer_features

## model specific variables (iterate the version and note with each change)
MODEL_DIR = "models"
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "Prophet for time-series"
DATA_DIR = os.path.join("..","data","cs-train")

def _model_train(df,tag,test=False):
    """
    example funtion to train model
    
    The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file 

    """


    ## start timer for runtime
    time_start = time.time()
    
    # X,y,dates = engineer_features(df)

    if test:
        n_samples = int(np.round(0.3 * df.shape[0]))
        subset_indices = np.random.choice(np.arange(df.shape[0]),n_samples,
                                          replace=False).astype(int)
        mask = np.in1d(np.arange(df.shape[0]),subset_indices)
        # y=y[mask]
        # X=X[mask]
        # dates=dates[mask]
        df = df[mask]
        
    ## Perform a train-test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
    #                                                     shuffle=True, random_state=42)
    # ## train a random forest model
    # param_grid_rf = {
    # 'rf__criterion': ['mse','mae'],
    # 'rf__n_estimators': [10,15,20,25]
    # }

    # pipe_rf = Pipeline(steps=[('scaler', StandardScaler()),
    #                           ('rf', RandomForestRegressor())])
    
    # grid = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=5, iid=False, n_jobs=-1)
    # grid.fit(X_train, y_train)
    # y_pred = grid.predict(X_test)
    
    ## retrain using all data
    # grid.fit(X, y)

    #Train prophet
    df.rename(columns={'date':'ds', 'revenue': 'y'},inplace=True)
    mod = Prophet(weekly_seasonality=True)  
    mod.fit(df)
    y_pred = mod.predict(df)
    eval_rmse =  mean_squared_error(df.y,y_pred.yhat)
    
    model_name = re.sub("\.","_",str(MODEL_VERSION))
    if test:
        saved_model = os.path.join(MODEL_DIR,
                                   "test-{}-{}.joblib".format(tag,model_name))
        print("... saving test version of model: {}".format(saved_model))
    else:
        saved_model = os.path.join(MODEL_DIR,
                                   "sl-{}-{}.joblib".format(tag,model_name))
        print("... saving model: {}".format(saved_model))
        
    joblib.dump(mod,saved_model)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update log
    update_train_log(tag,(str(df.ds.min()),str(df.ds.max())),{'rmse':eval_rmse},runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE,test=True)
  

def model_train(data_dir,test=False):
    """
    funtion to train model given a df
    
    'mode' -  can be used to subset data essentially simulating a train
    """
    
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if test:
        print("... test flag on")
        print("...... subseting data")
        print("...... subseting countries")
        
    ## fetch time-series formatted data
    ts_data = fetch_ts(data_dir)

    ## train a different model for each data sets
    for country,df in ts_data.items():
        
        if test and country not in ['all']:
            continue
        
        _model_train(df,country,test=test)
    
def model_load(prefix='sl',data_dir=None,training=True):
    """
    example funtion to load model
    
    The prefix allows the loading of different models
    """

    if not data_dir:
        data_dir = os.path.join("..","data","cs-train")
    
    models = [f for f in os.listdir(os.path.join(".","models")) if re.search(prefix,f)]

    if len(models) == 0:
        raise Exception("Models with prefix '{}' cannot be found did you train?".format(prefix))

    all_models = {}
    for model in models:
        all_models[re.split("-",model)[1]] = joblib.load(os.path.join(".","models",model))

    ## load data
    ts_data = fetch_ts(data_dir)
    all_data = {}
    for country, df in ts_data.items():
        # X,y,dates = engineer_features(df,training=training)
        # dates = np.array([str(d) for d in dates])
        # all_data[country] = {"X":X,"y":y,"dates": dates}
        all_data[country] = df.rename(columns={'date':'ds', 'revenue': 'y'})
        
    return(all_data, all_models)

def model_predict(country,year,month,day,all_models=None,test=False, model_prefix="sl"):
    """
    example funtion to predict from model
    """

    ## start timer for runtime
    time_start = time.time()

    ## load model if needed
    if not all_models:
        all_data,all_models = model_load(prefix=model_prefix,training=False)
    
    ## input checks
    if country not in all_models.keys():
        raise Exception("ERROR (model_predict) - model for country '{}' could not be found".format(country))

    for d in [year,month,day]:
        if re.search("\D",d):
            raise Exception("ERROR (model_predict) - invalid year, month or day")
    
    ## load data
    model = all_models[country]
    data = all_data[country]

    ## check date
    target_date = "{}-{}-{}".format(year,str(month).zfill(2),str(day).zfill(2))
    print(target_date)

    if target_date not in data.ds.values:
        raise Exception("ERROR (model_predict) - date {} not in range {} - {}".format(target_date,
                                                                                    data.ds.values[0],
                                                                                    data.ds.values[-1]))
    date_indx = np.where(data.ds.values == target_date)[0][0]
    query = data.iloc[[date_indx]]

    ## make prediction and gather data for log entry
    y_pred = model.predict(query).yhat[0]
    y_proba = None
    if 'predict_proba' in dir(model) and 'probability' in dir(model):
        if model.probability == True:
            y_proba = model.predict_proba(query)


    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update predict log
    update_predict_log(country,y_pred,y_proba,y_proba,target_date,
                       runtime, MODEL_VERSION, test=test)
    
    return({'y_pred':y_pred,'y_proba':y_proba})

if __name__ == "__main__":

    """
    basic test procedure for model.py
    """

    ## train the model
    print("TRAINING MODELS")
    data_dir = DATA_DIR
    model_train(data_dir,test=True)

    ## load the model
    print("LOADING MODELS")
    all_data, all_models = model_load(prefix="test")
    print("... models loaded: ",",".join(all_models.keys()))

    ## test predict
    country='all'
    year='2018'
    month='01'
    day='05'
    result = model_predict(country,year,month,day, model_prefix="test")
    print(result)
