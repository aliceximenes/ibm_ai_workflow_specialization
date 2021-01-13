import numpy as pd
import unittest2 as unittest
import requests
import re

from api import *

port = 8050

try:
    requests.post('http://localhost:{}/'.format(port))
    server_available = True
except:
    server_available = False

class TestApi(unittest.TestCase):

    @unittest.skipUnless(server_available,"local server is not running")
    def test_train_ok(self):
        r = requests.post('http://127.0.0.1:{}/train'.format(port),json={"mode":"test"})
        train_complete = re.sub("\W+","",r.text)
        self.assertEqual(train_complete,'true')



### Run the tests
if __name__ == '__main__':
    unittest.main()