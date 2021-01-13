import numpy as pd
import unittest2 as unittest

from model import *

class TestModel(unittest.TestCase):
    def test_train1(self):

        MODEL_DIR = os.path.join("..","data","cs-train")

        model_train(MODEL_DIR,test=True)
        self.assertTrue(os.path.exists(MODEL_DIR))
    
    def test_load(self):

        _ , all_models = model_load(prefix='sl')

        for i, model in all_models.items():
            self.assertTrue('predict' in dir(model))
            self.assertTrue('fit' in dir(model))




### Run the tests
if __name__ == '__main__':
    unittest.main()