# Analyse af vilde problemer
## Komponenter

Projektet består af to primære scripts:

1. **create_csv.py**: Dette script transformerer data fra en Excel-fil til et format, der er klar til analyse. Det filtrerer data baseret på specifikke kriterier og beregner tekstlige embeddings for de angivne spørgsmål og svar.

2. **visualization.py**: Efter data er blevet forberedt af det første script, anvender dette script forskellige teknikker til data visualisering og klyngeanalyse, herunder UMAP-reduktion og k-means klyngedannelse, for at identificere mønstre og grupperinger i dataene.

Koden beregner også 3 distance scores, der kan bruges til at måle "vildheden" af et problem/årsag:

```
Single value distance cosine: 0.3322840775867809 - Et mål for den gennemsnitlige spredning/uenighed. Måles som gennemsnittet af alle embeddings og derefter af hver embeddings afstand til gennemsnittet. Dette kan ses som et vægtet gennemsnit af hvor vildt et problem/årsag er. Værdien 0 betyder at de er ens, og værdien 1 betyder at de er komplet forskellige.
Single value distance euc:  0.740568916160349 - Samme som ovenstående, bare hvor distancen er til center vektoren er målt med euklidisk distance.
Intercluster-disagreement value: 2.6223664079975366 - Et mål for hvor mange forskellige holdninger der er til et givent problem, samt hvor langt de i gennemsnit er fra hinanden. Flere holdninger (clusters), giver typisk en højere værdi. Beregnes som afstanden af center clusterene imellem divideret med antallet af clustre. 
```

3. **main.py**: Dette script skal bruges til at køre hele pipelinen. Her defineres dit relevante spørgsmål samt spørgsmål man vil filtrere på og en tilhørende værdi.


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
### Installation

For at køre disse scripts, skal du have Python 3.9 (eller højere) installeret på dit system samt de nødvendige biblioteker, som inkluderer `pandas`, `numpy`, `matplotlib`, `sklearn`, `umap-learn`, og `openai`. Du kan installere disse ved at køre:

```bash
pip install -r requirements.txt
```

### Start
Du kører programmet ved at udfylde de relevante variable i main.py. Dette inkluderer stien til excelfilen, `cause_question`, `solution_question` og `filter_question`.

Du kører programmet ved at indtaste `python main.py`


## Forklaring af kode
### create_csv.py
Denne kode udfører en række trin for at bearbejde og analysere data fra en Excel-fil ved hjælp af flere Python-biblioteker, herunder pandas, openai, og sklearn. Den definerer en funktion create_files, der automatiserer processen med at filtrere data, generere tekstembeddings og beregne cosine-ligheden mellem disse embeddings og foruddefinerede spørgsmål.

    1. Importerer nødvendige biblioteker:
        pandas for datahåndtering og analyse.
        openai for at interagere med OpenAI's API'er, herunder at hente tekstembeddings.
        constants for at få adgang til API-nøglen til OpenAI.
        cosine_similarity fra sklearn.metrics.pairwise til at beregne lighed mellem vektorer.
        numpy for effektiv numerisk beregning, her specifikt til at håndtere arrays.

    2. Initialiserer OpenAI-klienten: Bruger en API-nøgle til at initialisere en klient for at interagere med OpenAI's API.

    3. Definerer funktionen create_files: Denne funktion tager stien til en Excel-fil, to spørgsmål og et filterkriterium med en tilhørende værdi som input. Funktionens formål er at filtrere data baseret på et spørgsmål og en værdi, generere embeddings for to specifikke spørgsmål og beregne cosine-ligheden mellem disse embeddings og dataens tekstindhold.

    4. Læser Excel-filen: Bruger pandas til at indlæse et specifikt ark ('Complete') fra en Excel-fil og gemmer dataene i en DataFrame.

    5. Filtrerer data:
        Mapper tekstuelle svar på et specifikt spørgsmål til numeriske værdier baseret på en foruddefineret mapping.
        Filtrer dataframen baseret på en specifik værdi for det transformerede spørgsmål.

    6. Genererer embeddings for de angivne spørgsmål: Bruger OpenAI's API til at skabe embeddings for de to inputspørgsmål (cause_question og solution_question). En embedding er en vektorrepræsentation af tekst, der kan bruges til at måle tekstlig lighed.

    7. Finder relevante kolonner: Ud fra de angivne spørgsmål finder funktionen kolonner i dataframen, der starter med disse spørgsmåls præfikser.

    8. Kombinerer tekstdata: For hver relevant kolonne kombinerer funktionen ikke-tomme tekstværdier til en liste for både årsags- og løsningsrelaterede kolonner.

    9. Opretter nye DataFrames: For hver tekstliste (årsager og løsninger) oprettes der nye DataFrames, som indeholder hver tekst som en række.

    10. Beregner embeddings og afstande: For hver tekst i de nye DataFrames genereres embeddings, og derefter beregnes cosine-ligheden mellem disse embeddings og det oprindelige spørgsmåls embedding. Cosine-lighed bruges til at estimere, hvor "nær" en given tekst er til det oprindelige spørgsmål i det semantiske rum.

    11. Gemmer output: De endelige DataFrames, der indeholder teksterne, deres embeddings og beregnede afstande til de oprindelige spørgsmåls embeddings, gemmes til CSV-filer.

Funktionen returnerer stierne til de to output CSV-filer (cause_output.csv og solution_output.csv), der indeholder de filtrerede og analyserede data. Disse output skal videre bruges til visualization.py for at kunne fortolkes.


### visualization.py
Denne kode er designet til at udføre dataanalyse og visualisering af tekstdata ved hjælp af forskellige algoritmer. Funktionen create_visualizations tager tre parametre: en sti til en CSV-fil med data (data), et spørgsmål (question), og en visualiseringstype (visualization_type). Her er en trinvis forklaring af, hvad koden gør:

    1. Indlæser Data: Funktionen begynder med at indlæse data fra en CSV-fil ind i en pandas DataFrame.

    2. Forbereder Embeddings: Den ekstraherer og transformerer tekstembeddings fra kolonnen question_embedding ved at anvende ast.literal_eval, som konverterer tekststrengen repræsentation af listen tilbage til en faktisk liste af tal. Disse embeddings repræsenterer tekstdata i en numerisk form, der kan behandles af maskinlæringsmodeller.

    3. Beregner Afstande: Funktionen beregner både Euklidisk afstand og cosine afstand mellem hver embedding og et "center" embedding, som er gennemsnittet af alle embeddings. Disse afstandsmål anvendes til at evaluere, hvor tæt hver tekst er på et gennemsnitligt punkt i det semantiske rum.

    4. Standardiserer Embeddings: Ved at anvende StandardScaler skalerer funktionen embeddings, så de har gennemsnit 0 og varians 1. Dette trin er ofte nødvendigt for at forbedre præstationen af mange maskinlæringsalgoritmer.

    5. Vælger det Optimale Antal Clustere: Funktionen anvender K-Means clustering med forskellige antal clustere for at finde det antal, der giver den højeste silhouette-score. Silhouette-scoren er et mål for, hvor godt et datapunkt er matchet til sit eget cluster sammenlignet med det nærmeste nabo-cluster, og hjælper med at vælge det mest passende antal clustere.

    6. Plotter Silhouette Scores: Funktionen plotter silhouette-scores for forskellige antal clustere for visuelt at evaluere, hvilket antal der er bedst.

    7. Anvender UMAP for Dimensionalitetsreduktion: For at visualisere dataene i to dimensioner anvender funktionen UMAP (Uniform Manifold Approximation and Projection), en teknik for dimensionalitetsreduktion, der bevare kosinussimilarity. Den bruger det bedste antal clustere identificeret i det forrige trin til at gruppere embeddings.

    8. Opretter Cluster Labels via OpenAI's api: For hvert cluster genererer funktionen en label ved at bruge OpenAI's GPT-model til at skabe en kort beskrivelse baseret på teksterne i det cluster.

    9. Visualiserer Clustere og Centroids: Funktionen plotter de reducerede dimensioner af embeddings med clusters vist i forskellige farver og centroids markeret. Hver centroid er annoteret med den genererede label.

    10. Beregner og Gemmer Cluster-Afstande: Den beregner afstandene mellem hvert clusters centroid og gemmer disse afstande i en CSV-fil. Dette kan hjælpe med at forstå, hvor distinkte de forskellige clustere er fra hinanden.

    11. Udskriver Statistikker: Til sidst udskriver funktionen nogle nøglestatistikker, herunder gennemsnitlig cosine afstand, gennemsnitlig Euklidisk afstand, og et "uoverensstemmelsesværdi" baseret på centroid-afstandene, som kan give indsigt i den samlede cluster-struktur.

