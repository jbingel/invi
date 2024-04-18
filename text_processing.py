from functools import lru_cache
from typing import Iterable
from openai import OpenAI
from constants import OPENAI_API_KEY


# Initialize OpenAI client with an API key
client = OpenAI(api_key=OPENAI_API_KEY)
GPT3 = "gpt-3.5-turbo"
GPT4 = "gpt-4-0125-preview"
DEFAULT_MODEL = GPT4

@lru_cache(10000)
def condense_response(response: str, question: str="", model: str=DEFAULT_MODEL):

    system_prompt = f""""Din opgave er at kondensere besvarelser i et spørgeskema ned til et enkelt ord.
    Besvarelserne kommer fra et spørgeskema til eksperter i forskellige samfundsrelevante emner.
    Eksperterne bliver spurgt, hvad de anser som mulige årsager eller løsninger på et givet problem.
    Eksperterne svarer på meget forskellige måder, og vi vil gerne kunne sammenligne dem automatisk.
    Derfor skal du koge alle besvarelser ned til et enkelt ord, som kan fungere som en slags overskrift.
    """
    
    prompt = ""
    if question:
        prompt += f"""Her følger spørgsmålet: {question}\n\n"""
    prompt += f"""Her følger besvarelsen: {response}"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content

@lru_cache(10000)
def get_embedding(text: str, model="text-embedding-3-large"):
    if isinstance(text, str):
        return client.embeddings.create(input = [text], model=model, dimensions=256).data[0].embedding

@lru_cache(10000)
def get_cluster_label(question: str, responses: Iterable[str], labels: Iterable[str], model: str=DEFAULT_MODEL):
    completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"""Brugeren vil give dig en masse tekster. 
                 - Du skal med MAKSIMALT 3 ord give en årsag til problemerne.
                 - Dit svar skal bruges til at klassificere teksterne. 
                 - Svaret må ikke være tæt på en af disse '{', '.join(labels)}'.
                 - Teksterne er relateret til dette spørgsmål: {question}"""},
                {"role": "user", "content": "---\n".join(responses)}
            ]
        )
    return completion.choices[0].message.content
