from typing import Dict, List
import numpy as np
from scipy.stats import ks_2samp


def compare_problems(average_distances: Dict[str, Dict[str, List[float]]]):
    problems = list(average_distances.keys())
    response_types = list(average_distances[problems[0]].keys())

    cross_problem_significance = {r: {p: {} for p in problems} for r in response_types} 

    print(f"Comparing {len(problems)} problems: {problems}\n")

    for p, problem in enumerate(problems):
            for other_problem in problems[p+1:]:
                for response_type in response_types:
                    if other_problem != problem:

                        print(f"## Comparing '{problem}' with '{other_problem}' ({response_type})\n")
                        difference_of_means = np.mean(average_distances[problem][response_type]) - np.mean(average_distances[other_problem][response_type])
                        interpretation = "more similar" if difference_of_means < 0 else "less simliar"
                        test_result = ks_2samp(average_distances[problem][response_type], average_distances[other_problem][response_type])
                        print(f"Difference of means: {difference_of_means}")
                        print(f"p-value: {test_result.pvalue}")
                        print(f"Interpretation: Responses for '{problem}' are {interpretation} than responses for '{other_problem}'.")
                        if test_result.pvalue < 0.05:
                            print("The results are statistically significant at p<0.05.")
                        else:
                            print("The results are not statistically significant (p>=0.05)")
                        print()
                        cross_problem_significance[response_type][problem][other_problem] = {"difference_of_means": difference_of_means, "pvalue": test_result.pvalue}