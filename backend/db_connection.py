from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)

db = client["nlp_dashboard"]

collection = db["queries"]

print("Connected to MongoDB")
print("Documents:", list(collection.find()))