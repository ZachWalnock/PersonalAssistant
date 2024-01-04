import json
from openai import OpenAI
import numpy as np
from numpy.linalg import norm
import os
from dotenv import load_dotenv
load_dotenv()
with open('data.json', 'r') as f:
    data = json.load(f)

apiKey = os.getenv('API_KEY')
client = OpenAI(api_key=apiKey)
def answerQuestion(question):
    questionQuery = client.embeddings.create(
            input=question,
            model = "text-embedding-ada-002"
        )
    questionVector = np.array(questionQuery.data[0].embedding)
    relevantPositions = []
    textInfo = []
    for i in range(len(data['embeddings'])):
        embeddingVector = data['embeddings'][i]
        cosine = np.dot(embeddingVector, questionVector) / (norm(embeddingVector) * norm(questionVector))

        # Binary Insertion (OPTIMAL.)
        if len(relevantPositions) == 0:
            relevantPositions.append(cosine)
            textInfo.append(data['associatedInfo'][i])
        else:
            left = 0
            right = len(relevantPositions) - 1
            while left != right:
                middle = left + ((right - left) // 2)
                if cosine > relevantPositions[middle]:
                    right = middle
                else:
                    left = middle + 1

            if cosine > relevantPositions[left]:
                relevantPositions = relevantPositions[:left] + [cosine] + relevantPositions[left:]
                textInfo = textInfo[:left] + [data['associatedInfo'][i]] + textInfo[left:]
            else:
                relevantPositions = relevantPositions[:left + 1] + [cosine] + relevantPositions[left + 1:]
                textInfo = textInfo[:left + 1] + [data['associatedInfo'][i]] + textInfo[left + 1:]



    if len(relevantPositions) != 0:
        chatMessage = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Here is some relevant information about me: " + ', '.join(textInfo[:5]) + ". Use this information to answer the following question."},
                {"role": "user", "content": question}
            ],
            stream=True
        )
        return chatMessage
    else:
        return None
