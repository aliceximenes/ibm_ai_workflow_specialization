import time,os,csv,uuid
from datetime import date

if not os.path.exists(os.path.join(".","logs")):
    os.mkdir("logs")

def update_train_log(country, data_range,eval_metric,runtime,MODEL_VERSION,MODEL_VERSION_NOTE,test=False):
    """
    update train log file
    """

    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    if test:
        logfile = os.path.join("logs","train-test.log")
    else:
        logfile = os.path.join("logs","train-{}-{}.log".format(today.year, today.month))
        
    ## write the data to a csv file    
    header = ['timestamp','unique_id','country','data_range','eval_metric','model_version',
              'model_version_note','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[time.time(),uuid.uuid4(), country,
                            data_range,eval_metric,
                            MODEL_VERSION,MODEL_VERSION_NOTE,runtime])
        writer.writerow(to_write)

def update_predict_log(country, y_pred, y_lower, y_upper, query,runtime,MODEL_VERSION,test=False):
    """
    update predict log file
    """

    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    if test:
        logfile = os.path.join("logs","predict-test.log")
    else:
        logfile = os.path.join("logs","predict-{}-{}.log".format(today.year, today.month))
        
    ## write the data to a csv file    
    header = ['timestamp','unique_id','country','y_pred','y_lower', 'y_upper','query','model_version','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[time.time(),uuid.uuid4(), country,
                            y_pred,y_lower,y_upper,query,
                            MODEL_VERSION,runtime])
        writer.writerow(to_write)

if __name__ == "__main__":

    """
    basic test procedure for logger.py
    """

    from application.model import MODEL_VERSION, MODEL_VERSION_NOTE
    
    ## train logger
    update_train_log("Brazil","('2000-01-01', '2019-01-01')","{'rmse':4.2}","00:00:01",
                     MODEL_VERSION, MODEL_VERSION_NOTE,test=True)
    ## predict logger
    update_predict_log("Brazil", 1, -1, 3, "('2000-01-01', '2000-01-01')",
                       "00:00:01",MODEL_VERSION, test=True)