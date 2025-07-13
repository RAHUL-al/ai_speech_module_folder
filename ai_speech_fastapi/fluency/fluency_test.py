from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load the model and tokenizer
model_name = "prithivida/parrot_fluency_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example sentence
sentence = """Diwali, also known as Deepavali, is one of the most significant and widely celebrated festivals in India. It symbolizes the victory of light over darkness and good over evil. The word "Deepavali" means a row of lamps, and on this day, homes, streets, and temples are beautifully illuminated with diyas, candles, and electric lights.
Diwali is celebrated across various religions, including Hinduism, Jainism, and Sikhism, but with slightly different historical backgrounds. In Hindu tradition, Diwali marks the return of Lord Rama to Ayodhya after 14 years of exile and defeating the demon king Ravana. To welcome him, the people of Ayodhya lit lamps across the kingdom, which is reenacted every year through the lighting of diyas.
Preparations for Diwali begin weeks in advance. People clean and decorate their homes, buy new clothes, and prepare a variety of sweets and snacks. On the night of Diwali, families perform Lakshmi Puja, worshipping the Goddess of wealth and prosperity. After the puja, fireworks light up the sky, and children and adults alike enjoy bursting crackers and sharing sweets with friends and neighbors.
One of the key messages of Diwali is spiritual renewal. It encourages people to let go of past negativity and start afresh with hope, light, and positivity. It also promotes generosity, togetherness, and community bonding.
However, in recent years, there has been rising awareness about the harmful effects of firecrackers on health and the environment. Many people now opt for a green Diwali, using eco-friendly decorations and avoiding noisy crackers.
In conclusion, Diwali is a festival that brings joy, unity, and a sense of renewal. It is not just about lights and sweets, but also about spreading love, removing darkness from our lives, and embracing new beginnings with happiness and hope.
"""

# Tokenize and prepare input
inputs = tokenizer(sentence, return_tensors="pt", truncation=True)

# Run through model
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    score = probs[0][1].item()  # Assuming class 1 = fluent, class 0 = not fluent

print(f"🧠 Fluency Score (1 = fluent): {score:.2f}")
