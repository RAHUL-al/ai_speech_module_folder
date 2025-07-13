# tasks.py
from celery_app import celery
from ai_speech_module import Topic

@celery.task
def process_chunk(chunk_filename):
    topic = Topic()

    transcribed_text = topic.speech_to_text(chunk_filename)
    emotion = topic.detect_emotion(chunk_filename)
    fluency = topic.fluency_scoring(chunk_filename)
    pronunciation = topic.pronunciation_scoring(chunk_filename)
    vad_segments = topic.silvero_vad(chunk_filename)

    return {
        "text": transcribed_text,
        "emotion": emotion,
        "fluency": fluency,
        "pronunciation": pronunciation,
        "vad_segments": vad_segments,
    }
