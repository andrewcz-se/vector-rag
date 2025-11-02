# 1. Start from a base Python image
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# --- OPTIMIZATION ---
# 3. Copy *only* the requirements file first
COPY requirements.txt .

# 4. Install the requirements. This layer will now be cached
# as long as requirements.txt doesn't change.
RUN pip install -r requirements.txt

# 5. Now, copy the rest of your application code
COPY . /app
# --- END OPTIMIZATION ---

# 6. Expose the port the app runs on
EXPOSE 5000

# 7. Define the command to run your app
CMD ["python", "app.py"]

