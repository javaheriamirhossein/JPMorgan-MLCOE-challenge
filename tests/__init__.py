# tests/__init__.py

from .DeepHalo_Tests import TestDeepHaloChoiceModel

# Or define test suites
import unittest

def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestDeepHaloChoiceModel))
    return test_suite

# Run tests if module is executed directly
if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())