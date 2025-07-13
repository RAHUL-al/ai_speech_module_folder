from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch
import re

# Create pipeline
pipeline = KPipeline(lang_code='a')

# Original long text
text = '''
Diwali, also known as Deepavali, is one of the most significant and widely celebrated festivals in India. It symbolizes the victory of light over darkness and good over evil. The word "Deepavali" means a row of lamps, and on this day, homes, streets, and temples are beautifully illuminated with diyas, candles, and electric lights.
Diwali is celebrated across various religions, including Hinduism, Jainism, and Sikhism, but with slightly different historical backgrounds. In Hindu tradition, Diwali marks the return of Lord Rama to Ayodhya after 14 years of exile and defeating the demon king Ravana. To welcome him, the people of Ayodhya lit lamps across the kingdom, which is reenacted every year through the lighting of diyas.
Preparations for Diwali begin weeks in advance. People clean and decorate their homes, buy new clothes, and prepare a variety of sweets and snacks. On the night of Diwali, families perform Lakshmi Puja, worshipping the Goddess of wealth and prosperity. After the puja, fireworks light up the sky, and children and adults alike enjoy bursting crackers and sharing sweets with friends and neighbors.
One of the key messages of Diwali is spiritual renewal. It encourages people to let go of past negativity and start afresh with hope, light, and positivity. It also promotes generosity, togetherness, and community bonding.
However, in recent years, there has been rising awareness about the harmful effects of firecrackers on health and the environment. Many people now opt for a green Diwali, using eco-friendly decorations and avoiding noisy crackers.
In conclusion, Diwali is a festival that brings joy, unity, and a sense of renewal. It is not just about lights and sweets, but also about spreading love, removing darkness from our lives, and embracing new beginnings with happiness and hope.
'''

# Step 1: Split into clean sentence-level chunks
def split_into_chunks(text, max_chars=400):
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

chunks = split_into_chunks(text, max_chars=400)

# Step 2: Process each chunk with Kokoro and export
for i, chunk in enumerate(chunks):
    print(f"\n🔊 Chunk {i+1}: {chunk}\n")
    generator = pipeline(chunk, voice='af_heart')
    for j, (gs, ps, audio) in enumerate(generator):
        display(Audio(data=audio, rate=24000, autoplay=i == 0 and j == 0))
        sf.write(f'chunk_{i+1}_part_{j+1}.wav', audio, 24000)
