# firestore_models.py
from firebase_config import db

class FirestoreUser:
    def __init__(self, username, email, password, id=None):
        self.id = id  # Firestore doc ID (optional)
        self.username = username
        self.email = email
        self.password = password

    def to_dict(self):
        return {
            "username": self.username,
            "email": self.email,
            "password": self.password
        }

    @staticmethod
    def from_dict(doc_id, data):
        return FirestoreUser(
            id=doc_id,
            username=data.get("username"),
            email=data.get("email"),
            password=data.get("password")
        )


class FirestoreEssay:
    def __init__(self, user_id, student_class, accent, topic, mood, content, id=None):
        self.id = id
        self.user_id = user_id
        self.student_class = student_class
        self.accent = accent
        self.topic = topic
        self.mood = mood
        self.content = content

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "student_class": self.student_class,
            "accent": self.accent,
            "topic": self.topic,
            "mood": self.mood,
            "content": self.content
        }

    @staticmethod
    def from_dict(doc_id, data):
        return FirestoreEssay(
            id=doc_id,
            user_id=data.get("user_id"),
            student_class=data.get("student_class"),
            accent=data.get("accent"),
            topic=data.get("topic"),
            mood=data.get("mood"),
            content=data.get("content")
        )
