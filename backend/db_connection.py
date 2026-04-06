from pymongo import MongoClient
import os
from dotenv import load_dotenv
import certifi

# 🔹 Load environment variables
load_dotenv(dotenv_path="backend/.env")

MONGO_URI = os.getenv("MONGO_URI")

# 🔹 Correct MongoDB Atlas connection
client = MongoClient(
    MONGO_URI,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=5000
)

# 🔹 Database
db = client["nlp_dashboard"]

# 🔹 Collections
queries_collection = db["queries"]           # existing
predictions_collection = db["predictions"]   # ⭐ REQUIRED
labels_collection = db["labels"]             # ⭐ REQUIRED

# 🔹 Optional: keep old variable for compatibility
collection = queries_collection

# 🔹 Test connection
try:
    client.admin.command('ping')
    print("✅ Connected to MongoDB")
except Exception as e:
    print("❌ MongoDB connection error:", e)

# 🔹 Debug (optional)
#print("MONGO URI:", MONGO_URI)