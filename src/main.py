from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routes import user_control, recommend

app = FastAPI(title="Cinefuse API", description="Backend for Cinefuse Group Recommendation App")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_control.router)
app.include_router(recommend.router)

@app.get("/")
async def root():
    return {"message": "Welcome to Cinefuse API"}
