#!/bin/python
# coding=utf8
import unittest
import tensorflow as tf
from string import Template
import sys
sys.path.insert(1, '../utils')
import pgnToTensor

class PgnToFenTester(unittest.TestCase):
    def test_fill_empty_squares(self):
       expectedResult = [2, 0, 0, 6, 5, 0, 0, 2]
       pgnConverter = pgnToTensor.PgnToTensor()
       actualResult = pgnConverter.fill_empty_squares('r2kq2r')
       self.assertEqual(expectedResult, actualResult)


if __name__ == '__main__':
    unittest.main()
