# Use the official Python 3.11 image
FROM python:3.11

# Set environment variables for git credentials
ENV GIT_USERNAME peterspat
ENV GIT_PASSWORD 29.7_p91+(bw)

# Install git
RUN apt-get update && \
    apt-get install -y git

# Set the working directory
WORKDIR /src

# Clone the git repository
RUN git clone https://${GIT_USERNAME}:${GIT_PASSWORD}@github.com/peterspat/ai_case_study.git .

# Install any dependencies required by your Python script
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Run the Python script
CMD ["python", "your_script.py"]
