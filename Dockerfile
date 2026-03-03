FROM python:3.13

# Install UV
RUN pip install --upgrade pip
RUN pip install uv

# Copy the whole project
WORKDIR /app
COPY . /app

# Install dependencies
RUN uv sync

# Run the application through the virtual env
CMD [".venv/bin/python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
