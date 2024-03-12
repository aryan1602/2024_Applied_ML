#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:32:42 2024

@author: aryan
"""

import requests
import time
import unittest
import subprocess

import pickle
from score import *

class TestScoreFunction(unittest.TestCase):

    def test_smoke_test(self):
        # Ensure the function does not crash
        model = pickle.load(open('model.pkl','rb'))
        text = "This is a sample text for testing."
        threshold = 0.5
        result = score(text, model, threshold)
        self.assertIsNotNone(result)

    def test_format_test(self):
        # Ensure the input/output formats/types are as expected
        model = pickle.load(open('model.pkl','rb'))

        text = "This is a sample text for testing."
        threshold = 0.5
        prediction, propensity = score(text, model, threshold)
        self.assertIsInstance(prediction, bool)
        self.assertIsInstance(propensity, float)

    def test_prediction_values(self):
        # Ensure prediction value is 0 or 1
        model = pickle.load(open('model.pkl','rb'))
 
        text = "This is a sample text for testing."
        threshold = 0.5
        prediction, _ = score(text, model, threshold)
        self.assertIn(prediction, [0, 1])

    def test_propensity_range(self):
        # Ensure propensity score is between 0 and 1
        model = pickle.load(open('model.pkl','rb'))

        text = "This is a sample text for testing."
        threshold = 0.5
        _, propensity = score(text, model, threshold)
        self.assertGreaterEqual(propensity, 0)
        self.assertLessEqual(propensity, 1)

    def test_threshold_zero(self):
        # If the threshold is set to 0, prediction should always be 1
        model = pickle.load(open('model.pkl','rb'))

        text = "This is a sample text for testing."
        threshold = 0
        prediction, _ = score(text, model, threshold)
        self.assertEqual(prediction, 1)

    def test_threshold_one(self):
        # If the threshold is set to 1, prediction should always be 0
        model = pickle.load(open('model.pkl','rb'))

        text = "This is a sample text for testing."
        threshold = 1
        prediction, _ = score(text, model, threshold)
        self.assertEqual(prediction, 0)

    def test_spam_input(self):
        # On an obvious spam input text, the prediction should be 1
        model = pickle.load(open('model.pkl','rb'))

        text = "WINNER!! As a valued network customer you have been selected to receivea �900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."
        threshold = 0.5
        prediction, _ = score(text, model, threshold)
     
        self.assertEqual(prediction, True)

    def test_non_spam_input(self):
        # On an obvious non-spam input text, the prediction should be 0
        model = pickle.load(open('model.pkl','rb'))

        text = "Subject: re : meeting w kevin hannon  vince and stinson :  carol brown called and we have scheduled the meeting for 4 : 00 pm on  thursday , may 11 .  it will be in kevin ' s office at eb 4508 .  thanks !  shirley  stinson gibner  05 / 08 / 2000 04 : 42 pm  to : shirley crenshaw / hou / ect @ ect  cc :  subject : meeting w kevin hannon  shirley ,  could you call carol brown and set up a time for vince and i to meet with  kevin hannon later this week ?  thanks ,  stinson"
        threshold = 0.5
        prediction, _ = score(text, model, threshold)
     
        self.assertEqual(prediction, False)
        
        
        
class FlaskIntegrationTest(unittest.TestCase):
    def setUp(self):
        # Launch the Flask app using the subprocess module
        self.flask_process = subprocess.Popen(['python', 'app.py'])

        # Allow some time for the app to start
        time.sleep(2)

    def tearDown(self):
        # Close the Flask app by sending a request to the /shutdown endpoint
        shutdown_url = 'http://localhost:5000/shutdown'
        requests.post(shutdown_url)

        # Wait for the server to shut down
        self.flask_process.wait()

    def test_flask_endpoint(self):
        # Test the response from the localhost endpoint
        url = 'http://localhost:5000/score'
        text_data = {'text': 'This is a test text.'}
        response = requests.post(url, json=text_data)

        # Check if the response is successful (status code 200)
        self.assertEqual(response.status_code, 200)

        # Check if the response has the expected keys
        json_response = response.json()
        self.assertIn('prediction', json_response)
        self.assertIn('propensity', json_response)

if __name__ == '__main__':
    unittest.main()

