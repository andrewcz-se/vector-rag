# **Local Business RAG Search**

This is a simple web application that demonstrates a full Retrieval-Augmented Generation (RAG) pipeline. It uses a local vector database (ChromaDB) to find relevant local businesses and then sends that context to the Gemini API to generate a natural language summary. In this simple example to simulate the sanitized data that would be taken from a database or information source, ChromaDB is populated with entries we have added to the Python script

## **Features**

* **Vector Search:** Uses ChromaDB and sentence-transformers to perform semantic searches on a list of local businesses.  
* **Hybrid Search:** Automatically detects categories (auto repair, restaurant, etc.) from the user's query to apply metadata filters for more accurate results.  
* **AI Summaries:** Sends the retrieved business data as context to the Gemini API, which provides a helpful, conversational answer.  
* **Web Interface:** A simple HTML and JavaScript frontend for searching and viewing results.  
* **Fully Dockerized:** The entire application (Python backend, database, and models) is managed with Docker Compose for easy setup and persistent data.

## **How It Works**

1. A user enters a query (e.g., "where can I fix my car?") into the index.html frontend.  
2. The request is sent to the Flask server (app.py).  
3. The server automatically scans the query for known categories (e.g., "auto repair").  
4. The server queries ChromaDB, applying both the semantic vector search and any detected category filters.  
5. ChromaDB returns the most relevant local businesses.  
6. The server bundles the full details of these businesses (name, address, phone, etc.) into a "context" string.  
7. This context, along with the original query, is sent to the Gemini API.  
8. Gemini generates a natural language summary (e.g., "QuickCar Repair at 45 Sunset Blvd can help fix your car...").  
9. The Flask server sends both the AI summary and the raw search results back to the frontend to be displayed.

## **Project Structure**

.  
├── app.py               # The Flask web server, API endpoints, and all RAG logic.  
├── index.html           # The HTML/JS frontend for the user.  
├── docker-compose.yml   # Defines the app service and persistent volumes.  
├── Dockerfile           # Instructions to build the Docker image with Python & dependencies.  
├── requirements.txt     # List of all Python libraries (Flask, chromadb, etc.).  
└── README.md            # This file.

## **How to Use**

### **Prerequisites**

* [Docker](https://www.docker.com/get-started) and [Docker Compose](https://docs.docker.com/compose/install/) installed.  
* A Gemini API Key. You can get one from the [Google AI Studio](https://aistudio.google.com/app/apikey).
* The application is currently set to use the **gemini-2.5-flash-lite** Model as this has the most generous free API access limits. You may change the model by adjusting the URL in apiUrl.

### **1. Set Up Your API Key**

1. Create a new file in this directory named .env (just .env, no name before it).  
2. Add the GEMINI_API_KEY variable and add your key. It should look like this:

	> GEMINI_API_KEY=YOUR_ACTUAL_API_KEY_GOES_HERE

3. Replace YOUR_ACTUAL_API_KEY_GOES_HERE with your real Gemini API key.

### **2. Build and Run the Application**

Open your terminal in the project directory and run:

	> docker-compose up --build -d

* --build: This tells Docker Compose to build the image from the Dockerfile (it will install all the requirements.txt).  
* -d: This runs the container in "detached" mode in the background.

	> ! The first time you run this it will take some time to download the Python base image, install the libraries, and download the all-MiniLM-L6-v2 embedding model into your volume. The sentence-transformers requirement and dependencies is around 12Gb, and creating the image layers may take some time. The whole process may take around 10 minutes depending on your connection speed and device performance.

### **3. Access the Application**

Open your web browser and go to:

[**http://localhost:5000**](http://localhost:5000)

You should see the search interface. Try queries like:

* "where can I get my car fixed"  
* "I'm hungry late at night"  
* "find a quiet bookstore"  
* "where can I buy tools"

### **4. Stopping the Application**

To stop the server, run:

	> docker-compose down

### **5. Starting the Application after 1st Use**

To start the application after 1st use the following command:

	> docker-compose up -d
	
Changes to app.py and index.html should not require a **--build** as the Python application is called each time the container is brought up.

### **6. Cleaning Up (Optional)**

If you want to completely remove the persistent database (the my-chroma-data volume) and start fresh, run:

	> docker-compose down -v  
