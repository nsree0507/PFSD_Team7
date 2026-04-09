from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.db_connection import collection, collection_users
from pydantic import BaseModel
from bson import ObjectId

app = FastAPI()

class User(BaseModel):
    name: str = ""
    email: str
    password: str

class Ticket(BaseModel):
    text: str
    user_id: str
    label: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Backend running"}

@app.post("/tickets")
def create_ticket(ticket: Ticket):
    collection.insert_one(ticket.dict())
    return {"message": "Ticket created"}

@app.get("/tickets/{user_id}")
def get_tickets(user_id: str):
    tickets = list(collection.find({"user_id": user_id}))
    for t in tickets:
        t["_id"] = str(t["_id"])
    return {"data": tickets}

@app.post("/signup")
def signup(user: User):
    existing = collection_users.find_one({"email": user.email})
    if existing:
        return {"message": "User already exists"}
    collection_users.insert_one(user.dict())
    return {"message": "User created successfully"}

@app.post("/signin")
def signin(user: User):
    existing = collection_users.find_one({
        "email": user.email,
        "password": user.password
    })
    if existing:
        return {
            "message": "Login successful",
            "user_id": str(existing["_id"]),
            "name": existing["name"],
            "email": existing["email"]
        }
    else:
        return {"message": "Invalid credentials"}

@app.put("/update-profile/{user_id}")
def update_profile(user_id: str, user: User):
    result = collection_users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {
            "name": user.name,
            "email": user.email
        }}
    )
    if result.modified_count == 1:
        return {"message": "Profile updated successfully"}
    else:
        raise HTTPException(status_code=404, detail="User not found")
