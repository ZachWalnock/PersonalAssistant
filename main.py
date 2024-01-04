from openai import OpenAI
import json
import numpy as np

information = ''' My name is John Dunk, I am a third year computer science major at The Pennslyvania State University. Classes I have taken: Some of the relevant 
classes I have taken include CMPSC311 Operating Systems, CMPSC465 Data Structures and Algorithms, and CMPSC360 Discrete Math. My passion and expertise lies within Artificial 
Intelligence. Some relevant experiences I have include: graduating from the machine learning bootcamp, participating in the Nittany AI Challenge, and using ChatGPT to help me
with my math homework. My GPA is a 3.3.
'''
apiKey = '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'
client = OpenAI(api_key=apiKey)


data = {'embeddings': [], 'associatedInfo': []}
CHUNK_SIZE = 150
current = 0
i = 1
while current < len(information):
    chunk = information[current:current+CHUNK_SIZE]
    response = client.embeddings.create(
        input=chunk,
        model = "text-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    data['embeddings'].append(embedding)
    data['associatedInfo'].append(chunk)
    current += CHUNK_SIZE
with open('data.json', 'w') as f:
    json.dump(data, f)