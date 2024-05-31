# Analyse af vilde problemer
Dette program analyserer besvarelser af INVIs spørgeskema om vilde problemer. Målet er at beregne et uenigheds-score over svarene, hvor antagelsen er, at høj uenighed kendetegner et særligt vildt problem.

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


