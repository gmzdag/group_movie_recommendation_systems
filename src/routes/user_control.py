from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile
import zipfile

# Import user manager functions
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from user_manager import (
    get_userid_if_exists,
    user_has_ratings,
    import_letterboxd_export,
    update_user_recommender_cache
)

router = APIRouter(
    prefix="/user_control",
    tags=["User Control"]
)

class UsernameCheck(BaseModel):
    username: str

class UserDataResponse(BaseModel):
    user_id: int | None
    username: str
    has_data: bool
    message: str

@router.post("/check_user")
async def check_user(data: UsernameCheck) -> UserDataResponse:
    """
    Check if a Letterboxd username exists in the system.
    Returns user status and whether they have data.
    Does NOT create the user if they don't exist.
    """
    try:
        username = data.username.strip()
        if not username:
            raise HTTPException(status_code=400, detail="Username cannot be empty")
        
        user_id = get_userid_if_exists(username)
        
        if user_id is None:
            # User doesn't exist
            return UserDataResponse(
                user_id=None,
                username=username,
                has_data=False,
                message="We don't have your films yet"
            )
        
        has_data = user_has_ratings(user_id)
        
        if has_data:
            message = "User found with existing data"
        else:
            message = "We don't have your films yet"
        
        return UserDataResponse(
            user_id=user_id,
            username=username,
            has_data=has_data,
            message=message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload_letterboxd")
async def upload_letterboxd(
    file: UploadFile = File(...),
    username: str = Form(...),
    update: str = Form("false")
):
    """
    Upload Letterboxd export (ZIP or CSV) for a user.
    Supports:
    - Full ZIP export (ratings.csv + watchlist.csv)
    - Only ratings.csv (ZIP or single CSV)
    - Only watchlist.csv (ZIP or single CSV)
    """
    tmp_path = None
    # Convert string to boolean
    is_update = update.lower() == "true"
    try:
        # Validate file type
        if not (file.filename.endswith('.zip') or file.filename.endswith('.csv')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Please upload a ZIP or CSV file."
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        print(f"[API] Received file: {file.filename}, size: {len(content)} bytes")
        
        # Track what we're importing
        import_type = None
        
        # Validate ZIP contains data
        if file.filename.endswith('.zip'):
            try:
                with zipfile.ZipFile(tmp_path, 'r') as z:
                    files = z.namelist()
                    print(f"[API] ZIP contains: {files}")
                    
                    has_ratings = any('ratings.csv' in name.lower() for name in files)
                    has_watchlist = any('watchlist.csv' in name.lower() for name in files)
                    
                    if not has_ratings and not has_watchlist:
                        os.unlink(tmp_path)
                        raise HTTPException(
                            status_code=400,
                            detail="We opened the file, but couldn't find any films. Make sure your export includes ratings.csv or watchlist.csv"
                        )
                    
                    # Determine import type
                    if has_ratings and has_watchlist:
                        import_type = "both"
                    elif has_ratings:
                        import_type = "ratings"
                    else:
                        import_type = "watchlist"
                        
            except zipfile.BadZipFile:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise HTTPException(status_code=400, detail="Invalid ZIP file")
        else:
            # For single CSV, we'll detect type during import
            import_type = "csv"
        
        # Import the data
        print(f"[API] Importing data for user: {username}, update: {is_update}")
        user_id = import_letterboxd_export(username, tmp_path, update=is_update)
        print(f"[API] Import successful for user_id: {user_id}")
        
        # Update cache if updating existing user
        if is_update:
            print(f"[API] Updating cache for user_id: {user_id}")
            update_user_recommender_cache(user_id)
        
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        # Generate appropriate success message
        if import_type == "both":
            message = "Your films and watchlist are in!"
        elif import_type == "ratings":
            message = "Your rated films are in!"
        elif import_type == "watchlist":
            message = "Your watchlist is in!"
        else:
            message = "Your films are in!"
        
        return {
            "success": True,
            "message": message,
            "user_id": user_id,
            "username": username,
            "import_type": import_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        print(f"[API ERROR] Upload failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/user/{username}/status")
async def get_user_status(username: str):
    """
    Get detailed status for a user including last sync info.
    """
    try:
        user_id = get_userid_if_exists(username)
        
        if user_id is None:
            return {
                "user_id": None,
                "username": username,
                "has_data": False,
                "last_synced": None
            }
        
        has_data = user_has_ratings(user_id)
        
        return {
            "user_id": user_id,
            "username": username,
            "has_data": has_data,
            "last_synced": "14 days ago"  # TODO: Implement actual tracking
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create_group")
async def create_group_recommendation():
    """
    Initiates a new group recommendation session.
    """
    return {
        "message": "Group recommendation session initiated",
        "status": "active",
        "group_id": "temp_group_001"  # TODO: Implement actual group management
    }

@router.get("/status")
async def get_status():
    return {"status": "User Control Service Ready"}
