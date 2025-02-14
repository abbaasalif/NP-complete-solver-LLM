import re
import os
from flask import Flask, request, render_template, redirect, url_for, session
from openai import OpenAI
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from knapsack_module import get_gpt4_knapsack_solution, solve_knapsack, compare_knapsack_solutions, get_improved_knapsack_solution
from graph_coloring_module import get_gpt4_graph_coloring_solution, graph_coloring_solver, compare_graph_coloring_solutions, get_improved_graph_coloring_solution
from dotenv import load_dotenv

# Load environment variables from API_key.env
load_dotenv("API_key.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
app.secret_key = 'my_secret_key'  # required for session management

# ----- TSP Functions -----
def extract_final_route(llm_response):
    """
    Extracts the final route from the GPT-4o response.
    It looks for lines that begin with 'Optimal path:' or 'Final route:'.
    """
    pattern = r"(?:Optimal path|Final route)[:\-]?\s*\**([A-Za-z](?:\s*→\s*[A-Za-z])+)\**"
    match = re.search(pattern, llm_response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return llm_response.strip()

def get_gpt4o_tsp_solution(distance_matrix, cities, feedback=None):
    """
    Uses GPT-4o to generate a TSP solution.
    If 'feedback' is provided, it is appended to the prompt.
    """
    city_names = ", ".join(cities)
    feedback_text = ""
    if feedback:
        feedback_text = f"\nFeedback: The computed optimal route is {feedback}. Please revise your answer accordingly."
    prompt = f"""
Solve the Traveling Salesman Problem step by step.

Cities: {city_names}
Distance Matrix:
{distance_matrix}

Think step by step:
1. List all cities.
2. Compute all possible routes and their distances.
3. Find the route with the minimum cost.
4. Return the optimal path in the format: 'A → B → C → D → A'.

{feedback_text}

The final output should be a string with only the optimal path with prefix 'Optimal path:' or 'Final route:'.
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
    final_route = extract_final_route(raw_response)
    return final_route

def tsp_solver(distance_matrix):
    """
    Solves TSP using OR-Tools and returns the optimal route as a list of city indices.
    """
    num_cities = len(distance_matrix)
    manager = pywrapcp.RoutingIndexManager(num_cities, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return None

    index = routing.Start(0)
    route = []
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    route.append(manager.IndexToNode(index))
    return route

def compare_solutions(llm_solution, optimal_solution, cities):
    """
    Compares the LLM's solution to the optimal solution.
    """
    llm_route = [city.strip() for city in llm_solution.replace("→", "->").split("->")]
    optimal_route = [cities[i] for i in optimal_solution]
    if llm_route == optimal_route:
        return True, llm_route
    else:
        return False, optimal_route

def process_tsp_input(cities_input, matrix_input):
    """
    Processes the user input from the TSP form.
    """
    if not cities_input.strip():
        cities = ["A", "B", "C", "D"]
    else:
        cities = [c.strip() for c in cities_input.split(",")]
    lines = matrix_input.strip().splitlines()
    matrix = []
    for line in lines:
        parts = line.split(",")
        if len(parts) != len(cities):
            raise ValueError("The number of values in each row must match the number of cities.")
        row = [float(x.strip()) for x in parts]
        matrix.append(row)
    return cities, matrix

def classify_problem(description):
    """
    Uses GPT-4o to classify the NP-complete problem type from natural language.
    Returns one of: TSP, KNAPSACK, GRAPH_COLORING, or INSUFFICIENT.
    """
    prompt = f"""
You are an expert in combinatorial optimization. Based on the following user description, determine which NP-complete problem is being described. Respond with one of these keywords: TSP, KNAPSACK, GRAPH_COLORING. If the description does not contain sufficient details, respond with "INSUFFICIENT".
User description: {description}
Answer:
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in NP-complete problems."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "text"},
        temperature=0.0
    )
    classification = response.choices[0].message.content.strip().upper()
    return classification

# ----- Routes for Problem Classification & TSP -----
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        description = request.form.get("description", "")
        classification = classify_problem(description)
        session["problem_description"] = description
        session["classification"] = classification
        if classification == "TSP":
            return redirect(url_for("tsp_details"))
        elif classification == "KNAPSACK":
            return redirect(url_for("knapsack_details"))
        elif classification == "GRAPH_COLORING":
            return redirect(url_for("graph_coloring_details"))
        else:
            error_msg = f"Unsupported problem type or insufficient details. Classified as: {classification}."
            return render_template("index.html", error=error_msg)
    else:
        return render_template("index.html", error=None)

@app.route("/tsp", methods=["GET", "POST"])
def tsp_details():
    if request.method == "POST":
        cities_input = request.form.get("cities", "")
        matrix_input = request.form.get("matrix", "")
        error = None
        try:
            cities, matrix = process_tsp_input(cities_input, matrix_input)
        except Exception as e:
            error = str(e)
            return render_template("tsp.html", error=error)
        # Iterative solution process:
        llm_solution = get_gpt4o_tsp_solution(matrix, cities)
        optimal_solution = tsp_solver(matrix)
        optimal_solution_cities = [cities[i] for i in optimal_solution]
        correct, best_route = compare_solutions(llm_solution, optimal_solution, cities)
        max_iterations = 5
        iteration = 0
        while not correct and iteration < max_iterations:
            feedback = " → ".join(optimal_solution_cities)
            llm_solution = get_gpt4o_tsp_solution(matrix, cities, feedback=feedback)
            correct, best_route = compare_solutions(llm_solution, optimal_solution, cities)
            iteration += 1
        response_data = {
            "problem_description": session.get("problem_description", ""),
            "cities": cities,
            "matrix": matrix,
            "llm_solution": llm_solution,
            "optimal_solution": " → ".join(optimal_solution_cities),
            "correct": correct,
            "best_route": best_route,
            "iterations": iteration
        }
        return render_template("result.html", response=response_data)
    else:
        return render_template("tsp.html", error=None)

# ----- Knapsack Route -----
@app.route("/knapsack", methods=["GET", "POST"])
def knapsack_details():
    if request.method == "POST":
        items_input = request.form.get("items", "")
        capacity_input = request.form.get("capacity", "")
        try:
            capacity = float(capacity_input)
            items = []
            # Expect items in format: "weight:value; weight:value; ..."
            for part in items_input.split(";"):
                if part.strip():
                    weight, value = part.split(":")
                    items.append((float(weight.strip()), float(value.strip())))
        except Exception as e:
            return render_template("knapsack.html", error=str(e))
        
        values = [item[1] for item in items]
        weights = [item[0] for item in items]
        
        # Use iterative feedback to refine GPT-4's solution.
        llm_solution, or_tools_solution_tuples, iterations, correct = get_improved_knapsack_solution(values, weights, capacity)
        
        response_data = {
            "item_list": items,  # for displaying all items
            "capacity": capacity,
            "llm_solution": llm_solution,  # list of indices from GPT-4
            "or_tools_solution_tuples": or_tools_solution_tuples,  # list of (index, weight, value) tuples
            "iterations": iterations,
            "correct": correct
        }
        return render_template("result_knapsack.html", response=response_data)
    else:
        return render_template("knapsack.html", error=None)

# ----- Graph Coloring Route -----
@app.route("/graph_coloring", methods=["GET", "POST"])
def graph_coloring_details():
    if request.method == "POST":
        matrix_input = request.form.get("matrix", "")
        num_colors_input = request.form.get("num_colors", "")
        try:
            num_colors = int(num_colors_input)
            lines = matrix_input.strip().splitlines()
            adjacency_matrix = []
            for line in lines:
                row = [int(x.strip()) for x in line.split(",")]
                adjacency_matrix.append(row)
        except Exception as e:
            return render_template("graph_coloring.html", error=str(e))
        
        # Use iterative feedback for graph coloring.
        llm_solution, or_tools_solution, iterations, correct = get_improved_graph_coloring_solution(adjacency_matrix, num_colors)
        
        response_data = {
            "adjacency_matrix": adjacency_matrix,
            "num_colors": num_colors,
            "llm_solution": llm_solution,
            "or_tools_solution": or_tools_solution,
            "iterations": iterations,
            "correct": correct
        }
        return render_template("result_graph_coloring.html", response=response_data)
    else:
        return render_template("graph_coloring.html", error=None)

if __name__ == "__main__":
    app.run(debug=True)
