import unittest
import os
import requests
import subprocess
import time

class TestDocker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Build the Docker image
        subprocess.run(["docker", "build", "-t", "my-flask-app", "."])

        # Run the Docker container in detached mode
        subprocess.run(["docker", "run", "--name", "flask-app", "-d", "-p", "5000:5000", "my-flask-app"])

        # Wait for the container to start
        time.sleep(2)

    @classmethod
    def tearDownClass(cls):
        # Close the Docker container
        subprocess.run(["docker", "stop", "flask-app"])
        subprocess.run(["docker", "rm", "flask-app"])

    def test_request_response(self):
        # Send a sample request to the Flask app
        sample_text = "This is a sample text."
        response = requests.post("http://localhost:5000/score", json={"text": sample_text})

        # Check if the response is successful (status code 200) and has the expected content
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertIn('prediction', json_response)
        self.assertIn('propensity', json_response)
 

if __name__ == "__main__":
    unittest.main()
