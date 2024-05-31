import os
import argparse

from preprocess import augment_csv
from analysis import analyze


CONFIG = {
    "mistrivsel": {
        "input_file": "Resultater_mistrivsel_final.xlsx",
        "cause_question": "Hvad er efter din mening de(n) vigtigste årsag(er) til mistrivslen blandt børn og unge?",
        "solution_question": "Hvordan kan man efter din opfattelse bedst forbedre trivslen blandt børn og unge?",
        "weighting_question": "I hvor høj grad synes du, at du i dit arbejde, er relevant for trivslen blandt børn og unge?"
    },
    "trafik": {
        "input_file": "Resultater_kollektiv trafik_final.xlsx",
        "cause_question": "Hvad er efter din mening de(n) vigtigste årsag(er) til, at den kollektive trafik i yderområderne i Danmark ikke er tilstrækkelig?",
        "solution_question": "Hvordan kan man efter din opfattelse bedst forbedre den kollektive trafik i yderområderne i Danmark?",
        "weighting_question": "I hvor høj grad synes du, at du i dit arbejde, er relevant for den kollektive trafik i yderområderne i Danmark?"
    },
    "potentialegruppen": {
        "input_file": "Resultater_potentialegruppen.xlsx",
        "cause_question": "Hvad er efter din mening de(n) vigtigste årsag(er) til, at unge i potentialegruppen ikke har tilknytning til hverken uddannelse eller arbejdsmarkedet?",
        "solution_question": "Hvordan kan man efter din mening bedst styrke tilknytningen til uddannelse og arbejdsmarked blandt de unge i potentialegruppen?",
        "weighting_question": "I hvilken grad synes du, at du i dit arbejde er relevant for potentialegruppen?"
    },
    "co2-udledning": {
        "input_file": "Resultater_co2-udledning i landbruget.xlsx",
        "cause_question": "Hvad er efter din mening de(n) vigtigste årsag(er) til, at landbrugets CO2-udledning har den størrelse, det har?",
        "solution_question": "Hvordan kan man efter din mening bedst nedbringe landbrugets CO2-udledning?",
        "weighting_question": "I hvilken grad synes du, at du i dit arbejde er relevant for landbrugets CO2-udledning?"
    }
}


def main():

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", help="The problem to solve.")
    # option: is precompted?
    parser.add_argument("-p", "--precomputed", action="store_true", help="Use precomputed data.")
    # option: use condensed responses for analysis
    parser.add_argument("-c", "--condensed", action="store_true", help="Use condensated responses for analysis.")
    # option to weight responses by importance
    parser.add_argument("-w", "--weighted", action="store_true", help="Weight responses by importance.")
    # option to create clusters and visualization 
    parser.add_argument("-v", "--visualize", action="store_true", help="Create clusters and visualization.")
    args = parser.parse_args()


    problem = args.problem

    # Angiver stien til output-mappen
    output_dir = f"output/{problem}"
    os.makedirs(output_dir, exist_ok=True)

    print(args)

    # If not already done, precompute embeddings and condensated responses
    if not args.precomputed:
        augment_csv(
            CONFIG[problem]["input_file"], 
            CONFIG[problem]["cause_question"], 
            CONFIG[problem]["solution_question"], 
            CONFIG[problem]["weighting_question"], 
            output_dir
        )
    
    for response_type in ["cause", "solution"]:
        analyze(
            augmented_data_path=f"{output_dir}/{response_type}s_augmented.csv", 
            question=CONFIG[problem][f"{response_type}_question"], 
            response_type=response_type, 
            condensed=args.condensed, 
            importance_weighting=args.weighted,
            visualize=args.visualize, 
            output_dir=output_dir
        )
    

if __name__ == "__main__":
    main()