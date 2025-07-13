# # # import os
# # # import requests
# # # from dotenv import load_dotenv

# # # load_dotenv()

# # # def speech_to_text() -> str:
# # #         API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
# # #         headers = {
# # #             "Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}",
# # #             "Content-Type": "audio/wav"
# # #         }

# # #         if not os.path.exists("harvard.wav"):
# # #             print(f"[Error] Audio file not found: ")
# # #             return ""

# # #         try:
# # #             with open("harvard.wav", "rb") as f:
# # #                 data = f.read()

# # #             response = requests.post(API_URL, headers=headers, data=data)
# # #             response.raise_for_status()

# # #             result = response.json()
# # #             text = result.get("text", "").strip()
# # #             print(f"Transcribed [{os.path.basename("harvard.wav")}]: {text}")
# # #             return text

# # #         except Exception as e:
# # #             print(f"[Error] Failed to transcribe harvard.wav: {e}")
# # #             return ""
        
# # # speech_to_text()


# # from langchain_openai import ChatOpenAI
# # from langchain.schema import SystemMessage, HumanMessage
# # import os
# # import threading
# # from dotenv import load_dotenv

# # load_dotenv()

# # class TinyLlamaChat:
# #     def __init__(self):
# #         self.llm = ChatOpenAI(
# #             model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
# #             openai_api_key=os.getenv("OPENROUTER_API_KEY"),
# #             openai_api_base="https://openrouter.ai/api/v1",
# #             temperature=0.7
# #         )


# #     def topic_data_model(self, username: str, prompt: str) -> str:
# #         try:
# #             messages = [
# #                 SystemMessage(content="You are a helpful assistant."),
# #                 HumanMessage(content=prompt)
# #             ]
# #             response = self.llm.invoke(messages)
# #             text_output = response.content
# #             return text_output
# #         except Exception as e:
# #             print(f"[LangChain TinyLlama API Error] {e}")
# #             return "TinyLlama model failed to generate a response."


# # test = TinyLlamaChat()
# # data = test.topic_data_model("Prince","tell me the capial of india?")


# # import torch
# # from transformers import pipeline

# # pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# # messages = [
# #     {
# #         "role": "system",
# #         "content": "You are a for help in text generation.",
# #     },
# #     {"role": "user", "content": "tell me the capial of india"},
# # ]
# # prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# # outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
# # print(outputs[0]["generated_text"])


# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "Qwen/Qwen3-0.6B"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )

# prompt = "Give me a short introduction to large language model."
# messages = [
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=False
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=32768
# )
# output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# try:
#     index = len(output_ids) - output_ids[::-1].index(151668)
# except ValueError:
#     index = 0

# content = tokenizer.decode(output_ids[index:], skip_special_tokens=True)

# print("content:", content)


# # def topic_data_model_for_Qwen(prompt: str) -> str:
# #         try:
# #             api_url = "https://api-inference.huggingface.co/models/Qwen/Qwen1.5-7B-Chat"
# #             headers = {
# #                 "Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}",
# #                 "Content-Type": "application/json"
# #             }

# #             payload = {
# #                 "inputs": [
# #                     {"role": "user", "content": prompt}
# #                 ],
# #                 "parameters": {
# #                     "do_sample": True,
# #                     "max_new_tokens": 512,
# #                     "return_full_text": False
# #                 }
# #             }

# #             response = requests.post(api_url, headers=headers, json=payload)
# #             response.raise_for_status()
# #             result = response.json()

# #             if isinstance(result, dict) and "error" in result:
# #                 print(f"[HuggingFace Qwen Error] {result['error']}")
# #                 return "Qwen API returned an error."

# #             if isinstance(result, list):
# #                 output_text = result[0]["generated_text"]
# #             else:
# #                 output_text = str(result)

# #             cleaned_text = re.sub(r'\*.*?\*', '', output_text)
# #             cleaned_text = cleaned_text.replace('\n', ' ').replace('\r', ' ')
# #             cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

# #             return cleaned_text

# #         except Exception as e:
# #             print(f"[Qwen API Error] {e}")
# #             return "Qwen model failed to generate a response."
        
        
# # result = topic_data_model_for_Qwen("tell me the capital of india")
# # print(result)



import os
import ujson as json
import redis
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, status, Request, WebSocket, WebSocketDisconnect,UploadFile, File, Form
from fastapi_sqlalchemy import DBSessionMiddleware, db
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from models import User, Essay
from schemas import UserCreate, UserOut, UserUpdate, Token, LoginRequest, ForgotPasswordRequest, GeminiRequest, TextToSpeechRequest, ChatRequest
from ai_speech_module import Topic as Topic
from auth import hash_password, verify_password, create_access_token
from dotenv import load_dotenv
from urllib.parse import parse_qs
from concurrent.futures import ThreadPoolExecutor
import asyncio
from fastapi.responses import FileResponse
import time
import re
from fastapi import FastAPI, UploadFile, File, Form
# from google.cloud import vision
from pdf2image import convert_from_path
from typing import List
import os, shutil, io, base64, requests, uuid
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from fastapi import FastAPI, UploadFile, File, Form, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from urllib.parse import parse_qs, unquote


executor = ThreadPoolExecutor(max_workers=4)

load_dotenv()

app = FastAPI(title="FastAPI​‑SQLAlchemy​‑MySQL")

redis_client = redis.StrictRedis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    username=os.getenv("REDIS_USERNAME"),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)

origins = [
    "http://localhost:3000",
    "http://43.205.138.222/",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(DBSessionMiddleware, db_url=os.environ["DATABASE_URL"])

qwen_model_name = "Qwen/Qwen1.5-0.6B"
tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(qwen_model_name, trust_remote_code=True).eval()

if torch.cuda.is_available():
    qwen_model = qwen_model.to("cuda")


def get_user_from_redis_session(request: Request):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = token.split(" ")[1]
    session_data = redis_client.get(f"session:{token}")
    if not session_data:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    return json.loads(session_data)



@app.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(user: UserCreate):
    if db.session.query(User).filter(
        (User.username == user.username) | (User.email == user.email)
    ).first():
        raise HTTPException(400, "Username or email already exists")

    new_user = User(
        username=user.username,
        email=user.email,
        password=hash_password(user.password)
    )
    db.session.add(new_user)
    db.session.commit()
    db.session.refresh(new_user)
    return new_user

@app.post("/login", response_model=Token)
def login(data: LoginRequest):
    user = db.session.query(User).filter(User.username == data.username).first()
    if not user or not verify_password(data.password, user.password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Bad credentials")

    token = create_access_token({"sub": str(user.id)})
    session_payload = json.dumps({"user_id": user.id, "username": user.username})
    redis_client.setex(f"session:{token}", timedelta(hours=1), session_payload)

    return Token(access_token=token, username=user.username)

@app.get("/logout")
def logout(request: Request):
    token = request.headers.get("Authorization")
    if token and token.startswith("Bearer "):
        redis_client.delete(f"session:{token.split(' ')[1]}")
    return {"detail": "Logged out"}

@app.get("/users/{user_id}", response_model=UserOut)
def get_user(user_id: int, user=Depends(get_user_from_redis_session)):
    user = db.session.get(User, user_id)
    if not user:
        raise HTTPException(404, "User not found")
    return user

@app.get("/me", response_model=UserOut)
def me(user=Depends(get_user_from_redis_session)):
    return user

@app.put("/users/{user_id}", response_model=UserOut)
def update_user(user_id: int, payload: UserUpdate, user=Depends(get_user_from_redis_session)):
    user_obj = db.session.get(User, user_id)
    if not user_obj:
        raise HTTPException(404, "User not found")
    if payload.username: user_obj.username = payload.username
    if payload.email: user_obj.email = payload.email
    if payload.password: user_obj.password = hash_password(payload.password)
    db.session.commit()
    return user_obj

@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, user=Depends(get_user_from_redis_session)):
    user = db.session.get(User, user_id)
    if not user:
        raise HTTPException(404, "User not found")
    db.session.delete(user)
    db.session.commit()
    return None

@app.post("/forgot-password", status_code=status.HTTP_200_OK)
def forgot_password(request: ForgotPasswordRequest):
    user = db.session.query(User).filter(User.username == request.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.password = hash_password(request.new_password)
    db.session.commit()
    return {"detail": "Password reset successfully"}

@app.post("/generate-prompt")
def generate_prompt(data: GeminiRequest, user=Depends(get_user_from_redis_session)):
    prompt = (
        f"Generate a essay for a student in class {data.student_class} "
        f"with a {data.accent} accent, on the topic '{data.topic}', and the mood is '{data.mood}' and give me essay less than 800 words and in response did not want \n\n or \n and also not want word count thanks you this type of stuff."
    )
    username = user.get("username")
    topic = Topic()
    response_text = topic.topic_data_model_for_Qwen(username, prompt)

    essay = Essay(
        student_class=data.student_class,
        accent=data.accent,
        topic=data.topic,
        mood=data.mood,
        content=response_text,
        user_id=user["user_id"]
    )
    db.session.add(essay)
    db.session.commit()
    db.session.refresh(essay)
    return {"response": response_text, "essay_id": essay.id}

TEMP_DIR = os.path.abspath("audio_folder")
os.makedirs(TEMP_DIR, exist_ok=True)

@app.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket):
    await websocket.accept()
    query_params = parse_qs(websocket.url.query)
    username = query_params.get("username", [None])[0]
    raw_token = query_params.get("token", [None])[0]
    token = unquote(raw_token) if raw_token else None

    if not username or not token or not token.startswith("Bearer "):
        await websocket.close(code=4001)
        print("Missing or invalid username/token in WebSocket connection.")
        return

    print(f"[WS] Client connected: {username} | Token: {token}")

    chunk_index = 0
    chunk_files = []
    text_output = []
    topic = Topic()

    date_str = datetime.now().strftime("%Y-%m-%d")
    user_dir = os.path.join(TEMP_DIR, username, date_str)
    os.makedirs(user_dir, exist_ok=True)

    final_output = os.path.join(user_dir, f"{username}_output.wav")
    transcript_path = os.path.join(user_dir, f"{username}_transcript.txt")

    if os.path.exists(final_output):
        os.remove(final_output)
    if os.path.exists(transcript_path):
        os.remove(transcript_path)

    loop = asyncio.get_event_loop()

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                print(f"[WS] {username} disconnected.")
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

                results = await asyncio.gather(
                    asyncio.to_thread(topic.speech_to_text, chunk_filename),
                    topic.detect_emotion(chunk_filename),
                    topic.fluency_scoring(chunk_filename),
                    topic.pronunciation_scoring(chunk_filename),
                    topic.silvero_vad(chunk_filename)
                )

                transcribed_text, emotion, fluency, pronunciation, vad_segments = results

                text_output.append(transcribed_text)
                print(f"[Chunk {chunk_index}] Transcribed: {transcribed_text.strip()}")
                print(f"Emotion: {emotion} | Fluency: {fluency} | Pronunciation: {pronunciation}")
                print(f"Silero VAD Segments: {len(vad_segments)} detected speech windows")
                chunk_index += 1

    except WebSocketDisconnect:
        print(f"[WS] {username} forcibly disconnected.")

    finally:
        await loop.run_in_executor(None, merge_chunks, chunk_files, final_output)
        print(f"[Output] Final audio saved: {final_output}")

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(" ".join(text_output).strip())
        print(f"[Output] Transcript saved: {transcript_path}")

        for file in chunk_files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"[Warning] Failed to remove {file}: {e}")

async def merge_chunks(chunk_files, final_output):
    print("[Merge] Merging audio chunks...")
    combined = AudioSegment.empty()
    for file in chunk_files:
        audio = AudioSegment.from_file(file, format="wav")
        combined += audio
    combined.export(final_output, format="wav")
    print("[Merge] Merged audio file saved.")

@app.get("/grammar-score")
def grammar_score(username: str):
    try:
        service = Topic()
        result = service.overall_scoring(username=username)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-tts-audio")
def get_tts_audio(username: str):
    folder = os.path.join("text_to_speech_audio_folder", username)
    file_path = os.path.join(folder, f"{username}_output.wav")

    timeout = 60
    poll_interval = 2
    waited = 0

    while waited < timeout:
        if os.path.exists(file_path):
            return FileResponse(
                file_path,
                media_type="audio/wav",
                filename=f"{username}_output.wav"
            )
        time.sleep(poll_interval)
        waited += poll_interval

    raise HTTPException(status_code=408, detail="Audio file not generated within 1 minute.")


# GOOGLE_VISION_API_KEY = os.getenv("YOUR_GOOGLE_VISION_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
# PINECONE_INDEX_NAME = "rag-index"

# pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
# if PINECONE_INDEX_NAME not in pinecone.list_indexes():
#     pinecone.create_index(PINECONE_INDEX_NAME, dimension=384)
    
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# @app.post("/upload/")
# async def upload_file(
#     file: UploadFile = File(...),
#     student_class: str = Form(...),
#     subject: str = Form(...)
# ):
#     folder = f"uploads/{student_class}/{subject}"
#     os.makedirs(folder, exist_ok=True)
#     file_path = os.path.join(folder, file.filename)

#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     extracted_text = extract_text_from_pdf_using_api_key(file_path)

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50
#     )
#     chunks = text_splitter.split_text(extracted_text)

#     vectorstore = PineconeVectorStore.from_texts(
#         texts=chunks,
#         embedding=embedding_model,
#         index_name=PINECONE_INDEX_NAME,
#         metadatas=[{
#             "student_class": student_class,
#             "subject": subject,
#             "filename": file.filename,
#             "id": str(uuid.uuid4())
#         }] * len(chunks)
#     )

#     return {
#         "filepath": file_path,
#         "chunks_stored": len(chunks),
#         "pinecone_index": PINECONE_INDEX_NAME
#     }


# def extract_text_from_pdf_using_api_key(pdf_path: str) -> str:
#     images: List = convert_from_path(pdf_path, dpi=300)
#     all_text = ""

#     for page_number, image in enumerate(images):
#         img_byte_arr = io.BytesIO()
#         image.save(img_byte_arr, format='JPEG')
#         encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode()

#         url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"

#         request_body = {
#             "requests": [
#                 {
#                     "image": {"content": encoded_image},
#                     "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]
#                 }
#             ]
#         }

#         response = requests.post(url, json=request_body)
#         result = response.json()

#         if "error" in result["responses"][0]:
#             raise Exception(f"Error on page {page_number + 1}: {result['responses'][0]['error']['message']}")

#         text = result["responses"][0].get("fullTextAnnotation", {}).get("text", "")
#         all_text += f"\n\n--- Page {page_number + 1} ---\n{text}"

#     return all_text




# @app.post("/chat/")
# async def chat(request: ChatRequest):
#     question = request.question
#     student_class = request.student_class
#     subject = request.subject

#     query_embedding = embedding_model.embed_query(question)

#     vectorstore = PineconeVectorStore.from_existing_index(
#         index_name=PINECONE_INDEX_NAME,
#         embedding=embedding_model
#     )

#     docs = vectorstore.similarity_search_by_vector(
#         embedding=query_embedding,
#         k=4,
#         filter={
#             "student_class": student_class,
#             "subject": subject
#         }
#     )

#     context_text = "\n\n".join([doc.page_content for doc in docs])

#     prompt = f"""You are a helpful assistant. Use the context below to answer the question.

#             Context:
#             {context_text}

#             Question: {question}
#             Answer:"""

#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
#     if torch.cuda.is_available():
#         inputs = {k: v.to("cuda") for k, v in inputs.items()}

#     output_ids = qwen_model.generate(
#         **inputs,
#         max_new_tokens=256,
#         do_sample=True,
#         temperature=0.7,
#         top_p=0.9
#     )

#     answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     answer_only = answer.split("Answer:")[-1].strip()

#     return {
#         "question": question,
#         "answer": answer_only,
#         "context_used": context_text[:300]
#     }


@app.get("/test")
def welcome_page():
    return {"Message": "Welcome the ai speech module page."}

@app.get("/")
def home():
    return {"message": "API is working"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)