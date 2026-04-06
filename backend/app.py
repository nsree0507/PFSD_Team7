from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.db_connection import collection

app = FastAPI()

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

@app.get("/tickets")
def get_tickets():

    tickets = list(collection.find())

    for t in tickets:
        t["_id"] = str(t["_id"])

    return {"data": tickets}
