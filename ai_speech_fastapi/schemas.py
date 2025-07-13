from pydantic import BaseModel, EmailStr, Field, model_validator

class UserCreate(BaseModel):
    username: str = Field(..., max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    confirm_password: str = Field(..., min_length=8)

    @model_validator(mode="after")
    def validate_password_match(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class UserOut(BaseModel):
    id: str
    username: str
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    username: str | None = Field(None, max_length=50)
    email:    EmailStr | None
    password: str | None = Field(None, min_length=8)

class Token(BaseModel):
    access_token: str
    username: str

class LoginRequest(BaseModel):
    username: str
    password: str

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordReset(BaseModel):
    token: str
    new_password: str


class ForgotPasswordRequest(BaseModel):
    username: str
    new_password: str

class GeminiRequest(BaseModel):
    student_class: str
    accent: str
    topic: str
    mood: str

class TextToSpeechRequest(BaseModel):
    text: str

class EssayOut(BaseModel):
    id: int
    student_class: str
    accent: str
    topic: str
    mood: str
    content: str

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    question: str
    student_class: str
    subject: str