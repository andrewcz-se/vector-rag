from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import sys
import os # <-- Import os
import json
import requests # <-- Import requests at the top

# --- FLASK APP SETUP ---
# We tell Flask to look for the 'index.html' file in the same directory
app = Flask(__name__, template_folder='.')


# --- CONFIGURATION ---
KNOWN_CATEGORIES = ['auto repair', 'restaurant', 'cafe', 'bookstore', 'hardware store', 'electronics']
MODEL_CACHE_PATH = "./chroma_data/models"

# --- HELPER FUNCTION ---
def find_category_in_query(query_text):
    """Scans query for a known category."""
    query_lower = query_text.lower()
    for category in KNOWN_CATEGORIES:
        if category in query_lower:
            return category
    return None

# --- CHROMA DB SETUP (Runs once on startup) ---
print(f"Loading embedding model (cache: {MODEL_CACHE_PATH})...")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    cache_folder=MODEL_CACHE_PATH 
)

print("Initializing persistent Chroma client...")
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection("businesses", embedding_function=embed_fn)

# 4. Example business entries
businesses = [
    {
        "id": "1",
        "name": "Bluebird Café",
        "desc": "Cozy café serving breakfast, coffee, and pastries.",
        "address": "123 Main St",
        "type": "cafe",
        "phone": "555-1234",
        "hours": "8am–5pm"
    },
    {
        "id": "2",
        "name": "QuickCar Repair",
        "desc": "Auto repair shop offering 24-hour emergency service.",
        "address": "45 Sunset Blvd",
        "type": "auto repair",
        "phone": "555-5678",
        "hours": "9am-5pm"
    },
    {
        "id": "3",
        "name": "NightOwl Diner",
        "desc": "Late-night diner open until 3am serving comfort food.",
        "address": "89 High Street",
        "type": "restaurant",
        "phone": "555-7865",
        "hours": "24/7"
    },
    {
        "id": "4",
        "name": "Green Garden",
        "desc": "Vegan restaurant open for lunch and dinner.",
        "address": "78 Oak Ave",
        "type": "restaurant",
        "phone": "555-8765",
        "hours": "11am–10pm"
    },
    {
        "id": "5",
        "name": "The Reading Nook",
        "desc": "Independent bookstore with a curated selection of fiction and non-fiction.",
        "address": "21B Baker St",
        "type": "bookstore",
        "phone": "555-9000",
        "hours": "10am–7pm"
    },
    {
        "id": "6",
        "name": "Main St Hardware",
        "desc": "Local hardware store for all your home improvement and repair needs. Sells tools, paint, and lumber.",
        "address": "129 Main St",
        "type": "hardware store",
        "phone": "555-1212",
        "hours": "7am–8pm"
    },
    {
        "id": "7",
        "name": "The Daily Grind",
        "desc": "Modern coffee shop with fast wifi, artisan espresso, and light snacks. Good for working.",
        "address": "500 Financial Plaza",
        "type": "cafe",
        "phone": "555-3344",
        "hours": "6am–6pm"
    },
    {
        "id": "8",
        "name": "L'Artiste",
        "desc": "Fine dining French restaurant. Perfect for a fancy dinner or anniversary. Reservations required.",
        "address": "99 Riverfront",
        "type": "restaurant",
        "phone": "555-9876",
        "hours": "5pm–11pm"
    },
	{
        "id": "9",
        "name": "The Electric Giant",
        "desc": "Modern large retailer of electronics. TVs, Phones, Laptops and more",
        "address": "600 Financial Plaza",
        "type": "electronics",
        "phone": "555-3233",
        "hours": "9am–7pm"
    },
	{
        "id": "10",
        "name": "Power Store",
        "desc": "Modern large retailer of electronics. TVs, Phones, Laptops and more",
        "address": "600 Financial Plaza",
        "type": "electronics",
        "phone": "555-3233",
        "hours": "9am–7pm"
    },
	{
        "id": "11",
        "name": "Media Market",
        "desc": "All new media. Selling Blurays, CDs also TVs, Phones, Laptops and more",
        "address": "600 Web Plaza",
        "type": "electronics",
        "phone": "555-4356",
        "hours": "9am–5pm"
    }
]

# 5. Add to collection
print("Syncing documents to collection...")
rich_documents = [
    f"Name: {b['name']}. Type: {b['type']}. Description: {b['desc']}"
    for b in businesses
]
# Create a mapping of ID to rich document
rich_doc_map = {b["id"]: doc for b, doc in zip(businesses, rich_documents)}

collection.add(
    ids=[b["id"] for b in businesses],
    documents=rich_documents,
    metadatas=[{
        "name": b["name"],
        "address": b["address"],
        "type": b["type"],
        "phone": b["phone"],
        "hours": b["hours"]
    } for b in businesses]
)
print(f"Collection is ready with {collection.count()} items.")


# --- NEW GEMINI API FUNCTION ---
def call_gemini_api(query, context):
    """Calls the Gemini API to generate a natural language summary."""
    
    # --- CHANGE 1: Read API key from environment variable ---
    apiKey = os.environ.get("GEMINI_API_KEY")
    
    # --- CHANGE 2: Add error handling if key is missing ---
    if not apiKey:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return ("Error: The server is not configured with an API key. "
                "Please set the GEMINI_API_KEY environment variable.")
    
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={apiKey}"
    
    # --- UPDATED SYSTEM PROMPT ---
    # Give the model permission to make common-sense connections.
    system_prompt = (
        "You are a helpful local assistant. Based on the provided context "
        "of local businesses, answer the user's question. "
        "It is OK to make common-sense inferences (e.g., 'car repair' or 'auto shop' "
        "can help a user 'fix their car'). "
        "If the provided context does not contain a relevant answer, "
        "simply state that you cannot find a relevant business in the provided information."
    )
    # --- END OF UPDATE ---
    
    # Format the context for the model
    context_string = "\n".join(context)
    prompt = f"User Question: {query}\n\nContext:\n{context_string}"
    
    payload = {
        "contents": [{ "parts": [{ "text": prompt }] }],
        "systemInstruction": {
            "parts": [{ "text": system_prompt }]
        },
    }
    
    try:
        response = requests.post(
            apiUrl, 
            headers={'Content-Type': 'application/json'}, 
            data=json.dumps(payload)
        )
        
        response.raise_for_status() # Raise an error for bad HTTP responses
        result = response.json()
        
        # --- Add logging to see the full API response ---
        print(f"Gemini API Response: {json.dumps(result, indent=2)}")

        # --- Safely check for 'candidates' ---
        if 'candidates' in result and result['candidates']:
            candidate = result['candidates'][0]
            if candidate.get('content') and candidate['content']['parts'][0].get('text'):
                return candidate['content']['parts'][0]['text']
            else:
                print("Error: Could not parse Gemini response content.")
                return "Error: Could not parse Gemini response content."
        
        # --- Handle API error messages gracefully ---
        elif 'error' in result:
            error_message = result['error'].get('message', 'Unknown API error')
            print(f"Gemini API Error: {error_message}")
            return f"Error from AI: {error_message}"
        
        else:
            print("Error: Unknown response structure from Gemini.")
            return "Error: Unknown response structure from Gemini."

    except Exception as e:
        print(f"Error calling Gemini: {e}")
        # Return the error message to the frontend
        return f"Error connecting to AI: {str(e)}" 


# --- WEB ROUTES ---

@app.route('/')
def home():
    """Serves the main index.html page."""
    return render_template('index.html')

@app.route('/api/all', methods=['GET'])
def api_get_all():
    try:
        print("Received request for /api/all")
        all_items = collection.get(include=["metadatas", "documents"])
        
        formatted_results = []
        if all_items["ids"]:
            # --- BUG FIX: Indentation corrected ---
            for i in range(len(all_items["ids"])):
                metadata = all_items["metadatas"][i]
                document = all_items["documents"][i]
                metadata["rich_document"] = document
                formatted_results.append(metadata)
            
        return jsonify({"results": formatted_results})

    except Exception as e:
        print(f"\nAn error occurred in /api/all: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/query', methods=['POST'])
def api_query():
    try:
        data = request.get_json()
        query = data.get('query')
        if not query:
            return jsonify({"error": "No query provided"}), 400

        print(f"Received query: '{query}'")

        # --- Dynamic Query Logic ---
        detected_category = find_category_in_query(query)
        
        query_args = {
            "query_texts": [query],
            "n_results": 3,
            "include": ["metadatas", "documents"] # Ask for documents
        }
        
        # --- BUG FIX: Define where_filter before using it ---
        if detected_category:
            where_filter = {"type": detected_category} # Define it here
            query_args["where"] = where_filter
            print(f"Querying with filter: {where_filter}")
        else:
            print("Querying without category filter.")
        
        # 7. Query example
        results = collection.query(**query_args)

        # 8. Format results for the frontend
        formatted_results = []
        context_for_gemini = [] # <-- New list for Gemini
        
        if results["ids"][0]:
            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i]
                document = results["documents"][0][i]
                
                metadata["rich_document"] = document
                formatted_results.append(metadata)
                
                # --- THIS IS THE FIX ---
                # Create a complete context string for this one result
                # The 'document' already has Name, Type, and Desc.
                # We just need to add the other metadata fields.
                full_context_string = (
                    f"{document} " # This has Name, Type, Desc
                    f"Address: {metadata['address']}. "
                    f"Phone: {metadata['phone']}. "
                    f"Hours: {metadata['hours']}."
                )
                context_for_gemini.append(full_context_string)
                # --- END OF FIX ---
        
        # 9. --- NEW: Call Gemini API ---
        ai_summary = None
        if formatted_results: # Only call Gemini if we have results
            print("Sending context to Gemini...")
            # --- BUG FIX: Corrected function name ---
            ai_summary = call_gemini_api(query, context_for_gemini)
        
        # 10. Return results + AI summary
        return jsonify({
            "results": formatted_results,
            "ai_summary": ai_summary  # <-- Send summary to frontend
        })

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        return jsonify({"error": str(e)}), 500

# --- RUN THE APP ---
if __name__ == '__main__':
    # Set debug=True for auto-reloading during development
    app.run(host='0.0.0.0', port=5000, debug=True)

