# Analyse af vilde problemer
Dette program analyserer besvarelser af INVIs spørgeskema om vilde problemer. Målet er at beregne et uenigheds-score over svarene, hvor antagelsen er, at høj uenighed kendetegner et særligt vildt problem.

## Sådan bruger man koden

Du starter analysen ved at køre
```bash
python main.py [PROBLEM] 
```
Her skal der indsættes `mistrivsel` eller `trafik` i stedet for `[PROBLEM]`.

Programmet består af tre trin:
 - Preprocessing af Excelfilen med besvarelser, hvor word embeddings beregnes både for det originale svar men også for en "kondenseret" udgave af svaret, hvor en Language Model koger alle besvarelser til et enkelt ord, så de er nemmere at sammenligne. Dette er med stor afstand det mest langvarige trin, og kan tage flere minutter.
 - Analyse af vektorafstande og andre uenighedsmål
 - Visualisering

Der er også en række optioner, som giver kontrol over analysen: 
 - `-p` indikerer, at preprocessing allerede er sket på et tidligere tidspunkt, og outputtet fra denne proces er at finde i output-folderen.
 - `-c` indikerer, at analysen skal gøre brug af de kondenserede svar
 - `-w` indikerer, at svarene skal vægtes efter vægntnigs-spørgsmålet (dette er typisk, hvor kompetent subjektet opfatter sig selv til at svare på spørgsmålet)
 - `-v` indikerer, at koden skal producere en visualisering


