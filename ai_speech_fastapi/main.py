import os
import ujson as json
import redis
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, status, Request, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from dotenv import load_dotenv
from urllib.parse import parse_qs, unquote
from concurrent.futures import ThreadPoolExecutor
from fastapi.responses import FileResponse
import asyncio
import shutil
import time
from tasks import process_chunk

from firebase_admin import credentials, firestore, initialize_app

cred = credentials.Certificate("firebase_key.json")
initialize_app(cred)
db = firestore.client()

from ai_speech_module import Topic
from auth import hash_password, verify_password, create_access_token
from schemas import UserCreate, UserOut, UserUpdate, Token, LoginRequest, ForgotPasswordRequest, GeminiRequest

load_dotenv()
app = FastAPI(title="FastAPI‑Firebase")

redis_client = redis.StrictRedis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    username=os.getenv("REDIS_USERNAME"),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)

origins = ["http://13.127.239.8:3005"]

app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def get_user_from_redis_session(request: Request):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = token.split(" ")[1]
    session_data = redis_client.get(f"session:{token}")
    if not session_data:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    return json.loads(session_data)

@app.post("/register", response_model=UserOut)
def register(user: UserCreate):
    user_ref = db.collection("users").where("username", "==", user.username).stream()
    if any(user_ref):
        raise HTTPException(400, "Username or email already exists")

    doc_ref = db.collection("users").add({
        "username": user.username,
        "email": user.email,
        "password": hash_password(user.password)
    })
    user_id = doc_ref[1].id
    return UserOut(id=user_id, username=user.username, email=user.email)

@app.post("/login", response_model=Token)
def login(data: LoginRequest):
    docs = db.collection("users").where("username", "==", data.username).stream()
    user_doc = next(docs, None)
    if not user_doc or not verify_password(data.password, user_doc.to_dict()["password"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Bad credentials")

    token = create_access_token({"sub": user_doc.id})
    redis_client.setex(f"session:{token}", timedelta(hours=1), json.dumps({"user_id": user_doc.id, "username": data.username}))
    return Token(access_token=token, username=data.username)

@app.get("/logout")
def logout(request: Request):
    token = request.headers.get("Authorization")
    if token and token.startswith("Bearer "):
        redis_client.delete(f"session:{token.split(' ')[1]}")
    return {"detail": "Logged out"}

@app.get("/me", response_model=UserOut)
def me(user=Depends(get_user_from_redis_session)):
    doc = db.collection("users").document(user["user_id"]).get()
    if not doc.exists:
        raise HTTPException(404, "User not found")
    data = doc.to_dict()
    return UserOut(id=doc.id, username=data["username"], email=data["email"])

@app.post("/generate-prompt")
def generate_prompt(data: GeminiRequest, user=Depends(get_user_from_redis_session)):
    prompt = f"Generate a essay for a student in class {data.student_class} with a {data.accent} accent, on the topic '{data.topic}', and the mood is '{data.mood}' and give me essay less than 800 words and in response did not want \n\n or \n and also not want word count thanks you this type of stuff."
    username = user.get("username")
    topic = Topic()
    response_text = topic.topic_data_model_for_Qwen(username, prompt)

    db.collection("essays").add({
        "student_class": data.student_class,
        "accent": data.accent,
        "topic": data.topic,
        "mood": data.mood,
        "content": response_text,
        "user_id": user["user_id"]
    })
    return {"response": response_text}

TEMP_DIR = os.path.abspath("audio_folder")
os.makedirs(TEMP_DIR, exist_ok=True)

@app.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket):
    await websocket.accept()
    query_params = parse_qs(websocket.url.query)
    username = query_params.get("username", [None])[0]
    raw_token = query_params.get("token", [None])[0]

    token = None
    if raw_token:
        decoded_once = unquote(raw_token)
        decoded_twice = unquote(decoded_once)
        if decoded_twice.startswith("Bearer "):
            token = decoded_twice
        elif decoded_once.startswith("Bearer "):
            token = decoded_once

    if not username or not token or not token.startswith("Bearer "):
        await websocket.close(code=4001)
        return

    chunk_index = 0
    chunk_files = []
    text_output = []
    topic = Topic()

    date_str = datetime.now().strftime("%Y-%m-%d")
    user_dir = os.path.join(TEMP_DIR, username, date_str)
    os.makedirs(user_dir, exist_ok=True)

    final_output = os.path.join(user_dir, f"{username}_output.wav")
    transcript_path = os.path.join(user_dir, f"{username}_transcript.txt")

    if os.path.exists(final_output): os.remove(final_output)
    if os.path.exists(transcript_path): os.remove(transcript_path)

    loop = asyncio.get_event_loop()

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message["type"] == "websocket.receive" and "bytes" in message:
                chunk_filename = os.path.join(user_dir, f"chunk_{chunk_index}.wav")
                audio = AudioSegment(
                    data=message["bytes"],
                    sample_width=2,
                    frame_rate=16000,
                    channels=1
                )
                audio.export(chunk_filename, format="wav")
                chunk_files.append(chunk_filename)
                task = process_chunk.delay(chunk_filename)
                while not task.ready():
                    await asyncio.sleep(0.5)

                results = task.get()
                transcribed_text, emotion, fluency, pronunciation, vad_segments = results
                text_output.append(transcribed_text)
                chunk_index += 1
    except WebSocketDisconnect:
        pass
    finally:
        await loop.run_in_executor(None, merge_chunks, chunk_files, final_output)
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(" ".join(text_output).strip())
        for file in chunk_files:
            try:
                os.remove(file)
            except:
                pass

async def merge_chunks(chunk_files, final_output):
    combined = AudioSegment.empty()
    for file in chunk_files:
        audio = AudioSegment.from_file(file, format="wav")
        combined += audio
    combined.export(final_output, format="wav")

@app.get("/get-tts-audio")
def get_tts_audio(username: str):
    folder = os.path.join("text_to_speech_audio_folder", username)
    file_path = os.path.join(folder, f"{username}_output.wav")

    timeout = 60
    poll_interval = 2
    waited = 0

    while waited < timeout:
        if os.path.exists(file_path):
            return FileResponse(file_path, media_type="audio/wav", filename=f"{username}_output.wav")
        time.sleep(poll_interval)
        waited += poll_interval

    raise HTTPException(status_code=408, detail="Audio file not generated within 1 minute.")

@app.get("/task-status/{task_id}")
def get_task_status(task_id: str):
    from celery_app import celery
    task = celery.AsyncResult(task_id)
    return {
        "task_id": task.id,
        "status": task.status,
        "result": task.result if task.ready() else None,
    }


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), student_class: str = Form(...), subject: str = Form(...)):
    folder = f"uploads/{student_class}/{subject}"
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, file.filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filepath": path}

@app.get("/test")
def welcome_page():
    return {"Message": "Welcome the ai speech module page."}

@app.get("/")
def home():
    return {"message": "API is working"}
