from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2
import numpy as np
import shutil
import os
from face_recognition_system import FaceRecognitionSystem

app = FastAPI()
face_system = FaceRecognitionSystem()
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "File uploaded successfully", "file_path": file_path}

@app.post("/process/")
async def process_video(file_name: str):
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    
    unique_faces_count = face_system.process_video(file_path, skip_frames=10, display=False, verbose=False)
    return {"message": "Processing complete", "unique_faces": unique_faces_count}

@app.get("/unique_faces/")
def get_unique_faces():
    return {"unique_faces": len(face_system.unique_faces)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
