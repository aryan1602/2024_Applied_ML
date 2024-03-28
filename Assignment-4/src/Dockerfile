# Use the official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install Flask and other dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords
# Copy the Flask app files to the working directory
COPY app.py .
COPY score.py .
COPY models .

# Expose the port on which your Flask app runs
EXPOSE 5000
ENV FLASK_APP=app.py
# Command to run the Flask app
CMD ["flask", "run", "--host", "0.0.0.0"]
