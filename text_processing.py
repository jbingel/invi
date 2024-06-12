from functools import lru_cache
import json
from typing import Iterable, List
from openai import OpenAI
from constants import OPENAI_API_KEY


# Initialize OpenAI client with an API key
client = OpenAI(api_key=OPENAI_API_KEY)
GPT3 = "gpt-3.5-turbo"
GPT4 = "gpt-4o"
DEFAULT_MODEL = GPT4
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
# DEFAULT_EMBEDDING_MODEL = "KennethEnevoldsen/dfm-sentence-encoder-large-exp2-no-lang-align"

@lru_cache(1)
def get_embedding_model(model: str = DEFAULT_EMBEDDING_MODEL) -> List[float]:
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model)


@lru_cache(10000)
def augment_response(response: str, question: str="", model: str=DEFAULT_MODEL):

    # responses = seperate_complex_responses(response, question)

    system_prompt = """"Din opgave er at opdele og kondensere besvarelser i et spørgeskema 
    ned til mellem ét og fem ord.
    
    Når en besvarelse reelt set rummer flere forskellige perspektiver, skal du dele dem op.

    For hver del skal du svare med fire outputs: 
    
    1. en kondensering til ét enkelt ord
    2. en kondensering til et til fire ord
    3. en kondensering til et til fem ord
    4. en vurdering af, hvorvidt besvarelsen er et relevant svar på spørgsmålet.

    Baggrund:
    Besvarelserne kommer fra et spørgeskema til eksperter i forskellige samfundsrelevante emner.
    Eksperterne bliver spurgt, hvad de anser som mulige årsager eller løsninger på et givet problem.
    Eksperterne svarer på meget forskellige måder, og vi vil gerne kunne sammenligne dem automatisk.
    Derfor skal du koge alle besvarelser ned til overskrifter af forskellig længde,
    for eksempel ét til fem ord, som kan fungere som en slags overskrift.
    Hvis en besvarelse er mindre end fem ord i forvejen, kan du bare gengive det orginale svar.
    Dit besvarelse indeholder KUN overskriften, ingen prefiks som 'overskrift'.

    Du skal desuden vurdere, hvorvidt besvarelsen er et relevant svar på spørgsmålet. Hvis svaret
    for eksempel kun gengiver spørgsmålet, eller kun består af spørgsmålstegn, skal det anses som
    ikke-relevant.

    Dit svar skal være et JSON-array med formatet: 
    ```
    [
        {
            "condensed_one": [condensed to one word],
            "condensed_max4": [condensed to one-four words],
            "condensed_max5": [condensed to one-five words],
            "relevant": [true if relevant, else false],
        },
        {
            "condensed_one": [condensed to one word],
            "condensed_max4": [condensed to one-four words],
            "condensed_max5": [condensed to one-five words],
            "relevant": [true if relevant, else false],
        },
        ...
    ]``` 

    """

    outputs = []

    prompt = ""
    if question:
        prompt += f"""Her følger spørgsmålet: {question}\n\n"""
    prompt += f"""Her følger besvarelsen: {response}"""

    for _ in range(3):
        try:

            llm_response = client.chat.completions.create(
                model=model,
                # response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

            augmented = llm_response.choices[0].message.content
            augmented = augmented.strip("```").strip("json").strip()

            # augmented = eval(augmented)
            augmented = json.loads(augmented)

            for response_part in augmented:
                condensed = response_part['condensed_one']
                condensed_max4 = response_part['condensed_max4']
                condensed_max5 = response_part['condensed_max5']
                relevant = response_part['relevant']
                outputs.append((condensed, condensed_max4, condensed_max5, relevant))

            break
        
        except Exception as e:
            # repeat
            print(e)
            pass
    return outputs


@lru_cache(10000)
def seperate_complex_responses(response, question, model=DEFAULT_MODEL):

    system_prompt = """"Din opgave er opdele svar i et spørgeskema i flere svar, hvis det oprindelige
    svar er komplekst, dvs. hvis det i virkeligheden indeholder flere svar. 
    Svarene er i dette spørgeskema enten bud på *årsager* på et bestemt problem, eller også er de bud på
    *løsninger* på problemet. Hvilken af de to svartyper de drejer sig om, fremgår af spørgsmålet.

    Du skal kun dele svarene op, hvis de enkelte dele rent faktisk bidrager med forskellige perspektiver.
    Hvis delene ikke rummer indhold der er forskellig nok, skal du ikke opdele.

    Dit svar skal være et JSON-objekt med formatet: 
    ```{
        "responses": "[list of seperated responses]"        
    }``` 

    Det kan sagtens være, at besvarelsen ikke er komplekst, altså at de ikke skal opdeles. I dette tilfælde
    indeholder listen kun det oprindelige svar.

    """
    
    prompt = ""
    if question:
        prompt += f"""Her følger spørgsmålet: {question}\n\n"""
    prompt += f"""Her følger besvarelsen: {response}"""

    llm_response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        
    )

    try:
        
        r_content = llm_response.choices[0].message.content.strip("```").strip("json")
        responses = eval(r_content).get("responses")
    except Exception as e:
        print(f"Did not manage to seperate responses. Returning original response. Error: {e}")
        responses = [response]
    return responses


@lru_cache(10000)
def get_embedding(text: str):
    return get_embedding_model().encode([text], normalize_embeddings=True)[0].tolist()
    

@lru_cache(10000)
def get_cluster_label(question: str, responses: Iterable[str], labels: Iterable[str], model: str=DEFAULT_MODEL):

    system_prompt = f"""Du er en hjælpsom assitent. Du får i det følgende
    en række besvarelser fra et spørgeskema omkring et samfundsrelevant problem.

    Din opgave er at genere et overskfift, der grupperer de følgende besvarelser, altså et overordnet tema. 
    Det skal være specifik og relevant ift det originale spørgsmål. 
    Det skal ikke være en generisk omformulering af det originale spørgsmål, men så specifik som muligt, men det stadig dækker over de fleste besvarelser.

    Overskriften må ikke være længere end femord.

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
