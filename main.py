from create_csv import create_files
from visualization import create_visualizations

# Questions
cause_question = "Hvad er efter din mening de(n) vigtigste årsag(er) til, at den kollektive trafik i yderområderne i Danmark ikke er tilstrækkelig?"
solution_question = "Hvordan kan man efter din opfattelse bedst forbedre den kollektive trafik i yderområderne i Danmark?"

# xlsx path
xlsx_path = "Resultater_kollektiv trafik.xlsx"

cause, solution = create_files(xlsx_path, cause_question, solution_question)

create_visualizations(cause, cause_question, "cause")
create_visualizations(solution, solution_question, "solution")