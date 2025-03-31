Unique Face Detection API

Overview

This is a FastAPI-based face recognition system using YOLOv5. 
The API allows users to process videos and detect unique faces and count the number of people in a video or recording. 
This project is created for Ola to enhance security and analytics.

Features

Face detection using YOLOv5

Unique face identification and counting

Process video files for face recognition

FastAPI-based RESTful API

Deployable with ngrok for easy sharing



Requirements

Ensure you have the following installed:

Python 3.9+

Anaconda (Optional but recommended)

FastAPI

Uvicorn

OpenCV (cv2)

Torch (PyTorch)

Ngrok (for deployment)


Installation

1️⃣ Clone the Repository

git clone https://github.com/himanshudhingra2910/backend-u.git

2️⃣ Create a Virtual Environment (Recommended)

conda create --name face-api python=3.9 -y
conda activate face-api

3️⃣ Install Dependencies
pip install -r requirements.txt

Running the API

1️⃣ Start FastAPI Server

uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Run the interface locally on http://127.0.0.1:8000/docs

2️⃣ Deploy Using Ngrok

ngrok http 8000

License

This project is licensed under the MIT License.

Developed by Himanshu Dhingra



