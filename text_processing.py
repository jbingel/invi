from functools import lru_cache
import json
from typing import Iterable
from openai import OpenAI
from constants import OPENAI_API_KEY


# Initialize OpenAI client with an API key
client = OpenAI(api_key=OPENAI_API_KEY)
GPT3 = "gpt-3.5-turbo"
GPT4 = "gpt-4-0125-preview"
DEFAULT_MODEL = GPT4

@lru_cache(10000)
def augment_response(response: str, question: str="", model: str=DEFAULT_MODEL):

    system_prompt = """"Din opgave er at kondensere besvarelser i et spørgeskema ned til mellem ét og tre ord.
    Du skal svare med tre outputs: 
    1. en kondensering til ét enkelt ord
    2. en kondensering til et til tre ord
    3. en kondensering til et til fire ord
    4. en vurdering af, hvorvidt besvarelsen er et relevant svar på spørgsmålet.

    Baggrund:
    Besvarelserne kommer fra et spørgeskema til eksperter i forskellige samfundsrelevante emner.
    Eksperterne bliver spurgt, hvad de anser som mulige årsager eller løsninger på et givet problem.
    Eksperterne svarer på meget forskellige måder, og vi vil gerne kunne sammenligne dem automatisk.
    Derfor skal du koge alle besvarelser ned til overskrifter af forskellig længde,
    for eksempel ét til tre ord, som kan fungere som en slags overskrift.
    Hvis en besvarelse er mindre end tre ord i forvejen, kan du bare gengive det orginale svar.
    Dit besvarelse indeholder KUN overskriften, ingen prefiks som 'overskrift'.

    Du skal desuden vurdere, hvorvidt besvarelsen er et relevant svar på spørgsmålet. Hvis svaret
    for eksempel kun gengiver spørgsmålet, eller kun består af spørgsmålstegn, skal det anses som
    ikke-relevant.

    Dit svar skal være et JSON-objekt med formatet: 
    ```{
        'condensed_one': [condensed to one word]
        'condensed_max3': [condensed to one-three words]
        'condensed_max4': [condensed to one-four words]
        'relevant': [true if relevant, else false]
    }``` 

    """
    
    prompt = ""
    if question:
        prompt += f"""Her følger spørgsmålet: {question}\n\n"""
    prompt += f"""Her følger besvarelsen: {response}"""

    for _ in range(3):
        try:

            response = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

            augmented = response.choices[0].message.content

            augmented = augmented.strip("```").strip("json").strip()

            # augmented = eval(augmented)
            augmented = json.loads(augmented)

            condensed = augmented['condensed_one']
            condensed_max3 = augmented['condensed_max3']
            condensed_max4 = augmented['condensed_max4']
            relevant = augmented['relevant']
            return condensed, condensed_max3, condensed_max4, relevant
        
        except Exception as e:
            # repeat
            print(e)
            pass
    return "N/A", "N/A", "N/A", False


@lru_cache(10000)
def get_embedding(text: str, model="text-embedding-3-large"):
    # import numpy as np
    # return np.array([0.0] * 256).reshape(-1)
    if isinstance(text, str):
        return client.embeddings.create(input = [text], model=model, dimensions=256).data[0].embedding

@lru_cache(10000)
def get_cluster_label(question: str, responses: Iterable[str], labels: Iterable[str], model: str=DEFAULT_MODEL):

    system_prompt = f"""Din opgave er at finde overskrifter til besvarelser af et
    spørgeskema omkring løsninger og årsager til samfundsrelevante problemer.
    Brugeren har allerede grupperet besvarelserne tematisk. Det er nu din opgave
    at give hver gruppe en overskrift.

    Overskriften må kun være ét ord.

    DU MÅ ALTID KUN SVARE MED ÉT ORD.

    Den skal ikke være tæt på en af disse: {', '.join(labels)}

    Det her er spørgsmålet, som besvarelserne handler om: {question}

    """

    completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Her er besvarelserne ---\n".join(responses)}
            ]
        )
    return completion.choices[0].message.content
