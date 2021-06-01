FROM pytorch/pytorch:latest

# Install linux packages 
RUN apt-get update && apt-get -y install libglib2.0-0 libgl1-mesa-glx htop screen; apt-get clean

# Install python dependencies 
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache -r requirements.txt

# Create woking directory
WORKDIR /app

# Copy contents
COPY . .

# Run command
ENTRYPOINT ["python"]
CMD ["app.py"]
