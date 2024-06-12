# Analyse af vilde problemer


Dette program analyserer besvarelser af INVIs spørgeskema om vilde problemer. Målet er at beregne et uenigheds-score over svarene, hvor antagelsen er, at høj uenighed kendetegner et særligt vildt problem.

Derfor benytter koden sig af såkaldte *embeddings* til at repræsentere besvarelserne numerisk (med højdimensionale vektorer), sådan at man med statistiske værktøjer kan beregne, hvor ensartede eller forksellige besvarelserne er for et givet problem.

## Get started
Koden er skrevet i Python og testet med Python version 3.11. For at installere de nødvendige packages, kan du køre `pip install -r requirements.txt`. 

## Sådan bruger man koden

Du starter analysen ved at køre
```bash
python main.py [PROBLEMS] 
```
Her skal der indsættes de problemer du vil analysere i stedet for `[PROBLEMS]` (separeret med mellemrum), fx. `python main.py mistrivsel trafik`. Problemerne er konfigureret i `CONFIG`-objektet i `main.py`.
Hvis flere end ét problem angives, gennemfører programmet desuden en meta-analyse, hvor det tester, om problemerne er signifikant anderledes ift. hvor stor spredning der er mellem besvarelserne. 

Programmet består af tre trin:
 - Preprocessing af Excelfilen med besvarelser, hvor word embeddings beregnes både for det originale svar men også for en "kondenseret" udgave af svaret, hvor en Language Model koger alle besvarelser til et enkelt eller få ord, så de er nemmere at sammenligne. Dette er med stor afstand det mest langvarige trin, og kan tage flere minutter.
 - Analyse af vektorafstande og andre uenighedsmål
 - Visualisering

Der er også en række optioner, som giver kontrol over analysen: 
 - `-p` indikerer, at preprocessing allerede er sket på et tidligere tidspunkt, og outputtet fra denne proces er at finde i output-folderen.
 - `-c` indikerer, at analysen skal gøre brug af de kondenserede svar
 - `-w` indikerer, at svarene skal vægtes efter vægntnigs-spørgsmålet (dette er typisk, hvor kompetent subjektet opfatter sig selv til at svare på spørgsmålet)
 - `-v` indikerer, at koden skal producere en visualisering


## Analyse og kommentarer

### Embedding-modellens indflydelse på resultaterne
Køres koden på datafilerne som de foreligger per juni 2024, vil man muligvis undre sig over metrikkernes værdier, og at forskellene mellem problemerne synes ret små. Dette kan være tilfældet, når embedding-modellen er trænet sådan, at den er god til at afspejle *relative forskelle* i inputtets betydning, som er hjælpsom når man fx. skal bruge den i en søgemaskine. Har man tre begreb som fx. 'priser', 'omkostninger' og 'fleksibilitet', så vil en sådan model muligvis tilskrive de første to ord en lighed på 98%, mens det første og tredje kan få en lighed på 95%. Det kan altså lyde som en lille forskel, men fra modellens synspunkt er det væsentlige her er, at de første to er mere lige end det første ord og det tredje.

Sådan forholder det sig for den populære embedding modelfamilie E5, som vi har brugt i dette projekt. Disse modeller virker bedst på dansk ifølge [Scandinavian Embedding Benchmark](https://kennethenevoldsen.github.io/scandinavian-embedding-benchmark/). E5-modellerne oplyser dette eksplicit [i deres dokumentation](https://huggingface.co/intfloat/multilingual-e5-large#faq).

Andre embedding-modeller vil sandssynligvis generere andre resultater, hvor forskellene mellem tallene er mere intuitive. I filen `text_processing.py` kan man eksperimentere med andre embedding-modeller ved at re-definere variablen `DEFAULT_EMBEDDING_MODEL`. I dette tilfælde rådes man ikke at køre hele pre-processingen om, da dette indebærer kondenserings-trinnet med en masse kald til OpenAI, hvilket tager lang tid og er relativt dyrt. I stedet kan man udkommentere en linie i `analysis.py`, hvor embeddings-beregningen gøres om. 

### Next steps

For at opnå mere intuitive resultater kan man, som beskrevet ovenfor, eksperimentere med andre embedding-modeller. Det kunne ligeledes være en god ide at re-skalere de parvise distancer så de spænder over hele intervallet mellem 0 og 1.

Det er også muligt, at andre metrikker er bedre egnet til at afspejle spredningen i besvarelserne.

Visualiseringen kunne have gavn af en forbedring af den prompt til GPT4, som genrerer cluster-overskrifter, sådan at disse bliver mere specifikke ift. det enkelte cluster.


> Koden er skrevet af Joachim Bingel (joabingel@gmail.com) baseret på tidligere arbejde af Mads Henrichsen.