from create_csv import create_files
from visualization import create_visualizations

# Definerer spørgsmål, der skal anvendes i analysen
cause_question = "Hvad er efter din mening de(n) vigtigste årsag(er) til mistrivslen blandt børn og unge?"
solution_question = "Hvordan kan man efter din opfattelse bedst forbedre trivslen blandt børn og unge?"
filter_question = "I hvor høj grad synes du, at du i dit arbejde, er relevant for trivslen blandt børn og unge?"

# Angiver stien til Excel-filen, der indeholder data
xlsx_path = "Resultater_mistrivsel.xlsx"

# Anvender create_files funktionen med den angivne sti og spørgsmål. Funktionen returnerer stierne til to CSV-filer: en for årsager og en for løsninger.
cause, solution = create_files(xlsx_path, cause_question, solution_question, filter_question, filter_value=0)

# Bruger create_visualizations funktionen til at generere visualiseringer for årsager til mistrivsel.
create_visualizations(cause, cause_question, "cause")

# Bruger create_visualizations funktionen til at generere visualiseringer for løsninger på mistrivsel.
create_visualizations(solution, solution_question, "solution")

