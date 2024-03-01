from create_csv import create_files
from visualization import create_visualizations

# Questions
cause_question = "Hvad er efter din mening de(n) vigtigste årsag(er) til mistrivslen blandt børn og unge?"
solution_question = "Hvordan kan man efter din opfattelse bedst forbedre trivslen blandt børn og unge?"
filter_question = "I hvor høj grad synes du, at du i dit arbejde, er relevant for trivslen blandt børn og unge?"

# xlsx path
xlsx_path = "Resultater_mistrivsel.xlsx"

cause, solution = create_files(xlsx_path, cause_question, solution_question, filter_question, filter_value=5)

create_visualizations(cause, cause_question, "cause")
create_visualizations(solution, solution_question, "solution")
