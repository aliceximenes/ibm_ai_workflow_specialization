import unittest2 as unittest

from api_tests import *
ApiTestSuite = unittest.TestLoader().loadTestsFromTestCase(TestApi)

from model_tests import *
ModelTestSuite = unittest.TestLoader().loadTestsFromTestCase(TestModel)

from log_tests import *
LogTestSuite = unittest.TestLoader().loadTestsFromTestCase(TestLog)

MainSuite = unittest.TestSuite([LogTestSuite,ModelTestSuite, ApiTestSuite])

### Run the tests
if __name__ == '__main__':
    unittest.main()