# Analyse af vilde problemer
## Komponenter

Projektet består af to primære scripts:

1. **create_csv.py**: Dette script transformerer data fra en Excel-fil til et format, der er klar til analyse. Det filtrerer data baseret på specifikke kriterier og beregner tekstlige embeddings for de angivne spørgsmål og svar.

2. **visualization.py**: Efter data er blevet forberedt af det første script, anvender dette script forskellige teknikker til data visualisering og klyngeanalyse, herunder UMAP-reduktion og k-means klyngedannelse, for at identificere mønstre og grupperinger i dataene.

Koden beregner også 3 distance scores, der kan bruges til at måle "vildheden" af et problem/årsag:

``````
Single value distance cosine: 0.3322840775867809 - Et mål for den gennemsnitlige spredning/uenighed. Måles som gennemsnittet af alle embeddings og derefter af hver embeddings afstand til gennemsnittet. Dette kan ses som et vægtet gennemsnit af hvor vildt et problem/årsag er.
Single value distance euc:  0.740568916160349 - Samme som ovenstående, bare hvor distancen er til center vektoren er målt med euklidisk distance.
Intercluster-disagreement value: 2.6223664079975366 - Et mål for hvor mange forskellige holdninger der er til et givent problem, samt hvor langt de i gennemsnit er fra hinanden. Flere holdninger (clusters), giver typisk en højere værdi.
``````


## Funktionalitet

### Data Forberedelse

- **Input**: Excel-fil (`Resultater_mistrivsel.xlsx`) med undersøgelsesdata.
- **Behandling**: Scriptet `create_csv.py` læser dataene, anvender en filtrering baseret på respondenternes relevans for trivslen blandt børn og unge, og genererer embeddings for hver besvarelse relateret til årsager til og løsninger på mistrivsel.
- **Output**: To CSV-filer (`cause_output.csv` og `solution_output.csv`) med de forarbejdede data, klar til videre analyse.

### Data Visualisering og Analyse

- **Input**: De forarbejdede CSV-filer fra det første skridt.
- **Behandling**: Scriptet `visualization.py` udfører UMAP-reduktion for at omdanne højdimensionelle embeddings til en 2D-repræsentation, anvender k-means klyngedannelse for at identificere grupper af lignende svar, og genererer visualiseringer af dataene.
- **Output**: Billedfiler med plots af klyngerne og deres silhouetscores, som hjælper med at forstå de dominerende årsager til og foreslåede løsninger på mistrivsel blandt børn og unge.

## Brug

For at køre disse scripts, skal du have Python installeret på dit system samt de nødvendige biblioteker, som inkluderer `pandas`, `numpy`, `matplotlib`, `sklearn`, `umap-learn`, og `openai`. Du kan installere disse ved at køre:

```bash
pip install -r requirements.txt
```


