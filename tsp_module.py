import re
import os
from openai import OpenAI
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from dotenv import load_dotenv

# Load environment variables from API_key.env
load_dotenv("API_key.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_final_route(llm_response):
    """
    Extracts the final route from the GPT-4o response.
    It looks for lines that begin with 'Optimal path:' or 'Final route:'.
    """
    # Pattern looks for "Optimal path:" or "Final route:" followed by the route string
    pattern = r"(?:Optimal path|Final route)[:\-]?\s*\**([A-Za-z](?:\s*→\s*[A-Za-z])+)\**"
    match = re.search(pattern, llm_response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # If no pattern is found, assume the entire response is the route
        return llm_response.strip()

def get_gpt4o_tsp_solution(distance_matrix, cities):
    """
    Use GPT-4o to generate a TSP solution with Chain-of-Thought prompting,
    and extract only the final route.
    """
    city_names = ", ".join(cities)
    
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
    route.append(manager.IndexToNode(index))  # Return to start

    return route

def compare_solutions(llm_solution, optimal_solution, cities):
    """
    Check if LLM's TSP solution is optimal by comparing routes.
    """
    # Convert LLM solution to a list of cities (assuming '→' as a delimiter)
    llm_route = [city.strip() for city in llm_solution.replace("→", "->").split("->")]
    optimal_route = [cities[i] for i in optimal_solution]

    if llm_route == optimal_route:
        return True, llm_route
    else:
        return False, optimal_route

def collect_tsp_details():
    """
    Interactively collects TSP details from the user.
    This version validates the input and re-prompts the user if the input is empty or not in the expected format.
    """
    print("Welcome to the TSP Chat Agent!")
    cities_input = input("Please enter the city names separated by commas (or press Enter for default: A, B, C, D): ")
    if cities_input.strip() == "":
        cities = ["A", "B", "C", "D"]
    else:
        cities = [city.strip() for city in cities_input.split(",")]
    print(f"Cities: {cities}")
    
    print("\nNow, please enter the distance matrix.")
    print("For each row, enter distances separated by commas. There should be one row per city.")
    matrix = []
    for i, city in enumerate(cities):
        while True:
            row_input = input(f"Enter distances from {city} to {', '.join(cities)}: ")
            if row_input.strip() == "":
                print("Input cannot be empty. Please enter the distances in the correct format (e.g., 0, 10, 15, 20).")
                continue
            try:
                row = [float(x.strip()) for x in row_input.split(",")]
                if len(row) != len(cities):
                    raise ValueError("The number of values does not match the number of cities.")
                matrix.append(row)
                break  # Valid row entered; exit the loop.
            except Exception as e:
                print(f"Error: {e}. Please enter the row in the correct format (e.g., 0, 10, 15, 20).")
    print("\nTSP instance details collected successfully!")
    return cities, matrix

def chat_tsp_agent():
    """
    Main function that drives the conversation with the user,
    collects details, and provides the TSP solutions.
    """
    cities, distance_matrix = collect_tsp_details()
    print("\nComputing solutions...\n")
    
    # GPT-4o solution:
    llm_solution = get_gpt4o_tsp_solution(distance_matrix, cities)
    print("GPT-4o Solution:", llm_solution)
    
    # OR-Tools optimal solution:
    optimal_solution = tsp_solver(distance_matrix)
    optimal_solution_cities = [cities[i] for i in optimal_solution]
    print("Optimal Solution:", " → ".join(optimal_solution_cities))
    
    # Compare solutions:
    correct, best_route = compare_solutions(llm_solution, optimal_solution, cities)
    if not correct:
        print(f"\nLLM solution is incorrect. The correct route is: {' → '.join(best_route)}. Retrying...\n")
        llm_solution = get_gpt4o_tsp_solution(distance_matrix, cities)
        correct, best_route = compare_solutions(llm_solution, optimal_solution, cities)
    print("Final LLM Solution:", llm_solution)
    
    print("\nThank you for using the TSP Chat Agent!")
