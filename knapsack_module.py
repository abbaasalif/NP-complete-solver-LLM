import re
import os
from ortools.algorithms.python import knapsack_solver
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from API_key.env
load_dotenv("API_key.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_final_knapsack_solution(llm_response):
    """
    Extracts the final knapsack solution from GPT-4's response.
    Expects a format like "Selected items: 0, 2, 5".
    Extraneous characters (such as asterisks) are removed.
    """
    pattern = r"(?:Selected items)[:\-]?\s*(.+)"
    match = re.search(pattern, llm_response, re.IGNORECASE)
    if match:
        selected = match.group(1).strip()
        # Remove any characters that are not digits, commas, or hyphens.
        cleaned = re.sub(r"[^\d,\-]", "", selected)
        try:
            return sorted([int(x.strip()) for x in cleaned.split(",") if x.strip() != ""])
        except Exception as e:
            print("Error converting token to int:", e)
            return []
    else:
        try:
            cleaned = re.sub(r"[^\d,\-]", "", llm_response)
            return sorted([int(x.strip()) for x in cleaned.split(",") if x.strip() != ""])
        except Exception:
            return []

def get_gpt4_knapsack_solution(values, weights, capacity, feedback=None):
    """
    Uses GPT-4 to produce a knapsack solution.
    Items are indexed 0, 1, ..., n-1.
    The GPT-4 answer should be in EXACTLY the format:
    "Selected items: i, j, k"
    If feedback (the OR-Tools solution) is provided, it is appended to the prompt.
    """
    n = len(values)
    items = ", ".join([str(i) for i in range(n)])
    feedback_text = ""
    if feedback is not None:
        feedback_text = f"\nFeedback: The optimal selection of items is {feedback}. Please adjust your answer accordingly."
    prompt = f"""
Solve the 0/1 Knapsack Problem.

Items: {items}
Values: {values}
Weights: {weights}
Capacity: {capacity}

Determine which items to select to maximize the total value without exceeding the capacity.
Return your answer in EXACTLY the following format:
Selected items: i, j, k
For example, if the optimal selection is items 1, 2, and 5, your answer should be:
Selected items: 1, 2, 5
Do not include any additional text.
{feedback_text}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in combinatorial optimization."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "text"},
        temperature=0.5
    )
    raw_response = response.choices[0].message.content
    return extract_final_knapsack_solution(raw_response)

def solve_knapsack(values, weights, capacity):
    """
    Solves the 0/1 knapsack problem using OR-Tools' latest KnapsackSolver API.
    
    - values: list of values for each item.
    - weights: list of weights for each item.
    - capacity: maximum capacity.
    
    Returns a list of tuples for each selected item: (index, weight, value).
    """
    # Convert inputs to integers
    profits = [int(v) for v in values]
    w_int = [int(w) for w in weights]
    capacities = [int(capacity)]
    # OR-Tools expects weights as a list of lists.
    weights_list = [w_int]
    
    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "KnapsackExample"
    )
    solver.init(profits, weights_list, capacities)
    computed_value = solver.solve()
    print("Total value =", computed_value)
    selected_items = []
    for i in range(len(profits)):
        if solver.best_solution_contains(i):
            selected_items.append((i, weights[i], values[i]))
    return selected_items

def compare_knapsack_solutions(llm_solution, or_tools_solution_tuples):
    """
    Compares GPT-4's solution (a list of indices) to OR-Tools' solution
    (a list of tuples (index, weight, value)). Returns True if they match.
    """
    or_tools_indices = sorted([item[0] for item in or_tools_solution_tuples])
    return llm_solution == or_tools_indices

def get_improved_knapsack_solution(values, weights, capacity, max_iterations=5):
    """
    Iteratively queries GPT-4 with feedback from OR-Tools until GPT-4's solution
    matches the OR-Tools solution or the maximum number of iterations is reached.
    
    Returns a tuple: (llm_solution, or_tools_solution_tuples, iterations, correct)
    """
    llm_solution = get_gpt4_knapsack_solution(values, weights, capacity)
    or_tools_solution_tuples = solve_knapsack(values, weights, capacity)
    correct = compare_knapsack_solutions(llm_solution, or_tools_solution_tuples)
    iteration = 0
    while not correct and iteration < max_iterations:
        # Provide feedback as a string representation of the OR-Tools indices
        feedback = str(sorted([item[0] for item in or_tools_solution_tuples]))
        llm_solution = get_gpt4_knapsack_solution(values, weights, capacity, feedback=feedback)
        correct = compare_knapsack_solutions(llm_solution, or_tools_solution_tuples)
        iteration += 1
    return llm_solution, or_tools_solution_tuples, iteration, correct
