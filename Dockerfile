# # Use an official Python runtime as a parent image
# FROM python:3.9.17-slim-buster

# # Set environment variables to avoid buffering and allow instant log output
# ENV PYTHONUNBUFFERED=1

# # Set the working directory in the container
# WORKDIR /app

# # Copy the current directory contents into the container at /app
# COPY . /app

# # Install the dependencies from the requirements.txt file
# RUN pip install --no-cache-dir -r requirements.txt

# # Expose the port Flask is running on
# EXPOSE 8080

# # Command to run the Flask app with Gunicorn
# CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]



# # Dockerfile
# FROM python:3.9.17-bookworm
# # Allow statements and log messages to immediately appear in the logs
# ENV PYTHONUNBUFFERED True
# # Copy local code to the container image.
# ENV APP_HOME /back-end
# WORKDIR $APP_HOME
# COPY . ./

# RUN pip install --no-cache-dir --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt

# # Run the web service on container startup. Here we use the gunicorn
# # webserver, with one worker process and 8 threads.
# # For environments with multiple CPU cores, increase the number of workers
# # to be equal to the cores available.
# # Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app





# Use Python 3.11 instead of Python 3.9
FROM python:3.11.5-bookworm

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /back-end
WORKDIR $APP_HOME
COPY . ./

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Run the web service on container startup.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app