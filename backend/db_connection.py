from pymongo import MongoClient
import os
from dotenv import load_dotenv
import certifi

load_dotenv(dotenv_path="backend/.env")
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(
    MONGO_URI,
    tlsCAFile=certifi.where(),
    tls=True,
    tlsAllowInvalidCertificates=True,
    serverSelectionTimeoutMS=5000
)

db = client["nlp_dashboard"]
queries_collection = db["queries"]
predictions_collection = db["predictions"]
labels_collection = db["labels"]

collection = queries_collection

try:
    client.admin.command('ping')
    print("✅ Connected to MongoDB")
except Exception as e:
    print("❌ MongoDB connection error:", e)

print("Connected to MongoDB")
#print("Documents:", list(collection.find()))
print("MONGO URI:", MONGO_URI)
