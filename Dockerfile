# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app code into the container at /app
COPY app.py .

# Make port 8502 available to the world outside this container (Streamlit's default port)
EXPOSE 8502

# Define environment variable to maybe configure Streamlit if needed (optional)
# ENV STREAMLIT_SERVER_PORT=8501
# ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Run app.py when the container launches
# Use healthcheck for better orchestration (optional but recommended)
HEALTHCHECK CMD streamlit hello
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]
