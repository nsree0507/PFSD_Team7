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

# ✅ ERP database
db = client["erp_db"]

# ✅ Collections
users_collection = db["users"]
tickets_collection = db["tickets"]

# ✅ Connection check
try:
    client.admin.command('ping')
    print("✅ Connected to MongoDB")
except Exception as e:
    print("❌ MongoDB connection error:", e)

print("Connected to MongoDB")
