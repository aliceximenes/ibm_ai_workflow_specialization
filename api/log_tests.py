import numpy as pd
import unittest2 as unittest
import os

from utils.logger import update_train_log, update_predict_log

class TestLog(unittest.TestCase):
    def test_train_log(self):
        log_file = os.path.join("logs","train-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)

        ## update the log
        tag = 'united_kingdom'
        period = ('2017-12-01', '2019-05-31')
        rmse = {'rmse':0.5}
        runtime = "00:00:01"
        MODEL_VERSION = 0.1
        MODEL_VERSION_NOTE = "Prophet for time-series"
        
        update_train_log(tag,period,rmse,runtime,
                         MODEL_VERSION, MODEL_VERSION_NOTE,test=True)

        self.assertTrue(os.path.exists(log_file))

### Run the tests
if __name__ == '__main__':
    unittest.main()