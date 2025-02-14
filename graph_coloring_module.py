import re
import json
from ortools.sat.python import cp_model
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv("API_key.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_final_coloring(llm_response):
    """
    Extracts a graph coloring solution from GPT-4's response.
    Expects a text output exactly in the format:
      "Coloring: 0:0, 1:1, 2:2, ..."
    Returns a dictionary mapping vertex (int) to color index (int).
    """
    pattern = r"Coloring[:\-]?\s*(.+)"
    match = re.search(pattern, llm_response, re.IGNORECASE)
    if match:
        assignments_str = match.group(1).strip()
        assignments = {}
        # Split by comma to get each "vertex:color" assignment.
        for assignment in assignments_str.split(","):
            parts = assignment.strip().split(":")
            if len(parts) == 2:
                try:
                    vertex = int(parts[0].strip())
                    color = int(parts[1].strip())
                    assignments[vertex] = color
                except Exception as e:
                    print("Error parsing assignment:", e)
                    continue
        return assignments
    else:
        return {}

def get_gpt4_graph_coloring_solution(adjacency_matrix, num_colors, feedback=None):
    """
    Uses GPT-4 to produce a graph coloring solution.
    The graph is defined by an adjacency matrix.
    num_colors: maximum number of colors available.
    feedback: if provided, a text string representing the OR-Tools solution,
              for example: "0:0, 1:1, 2:2" (vertex:color).
              
    The GPT-4 prompt instructs the model to return exactly the text output:
      "Coloring: 0:0, 1:1, 2:2, ..."  
    where the numbers are numeric color indices.
    """
    prompt = f"""
Solve the graph coloring problem.
The graph is represented by the following adjacency matrix (0 means no edge, 1 means edge):
{adjacency_matrix}
Use exactly {num_colors} colors, where colors are represented by numeric indices 0 to {num_colors - 1}.
Return your answer in EXACTLY the following format:
Coloring: vertex0:color0, vertex1:color1, vertex2:color2, ...
For example, if the solution is to assign color 0 to vertex 0, color 1 to vertex 1, and color 2 to vertex 2, your answer should be:
Coloring: 0:0, 1:1, 2:2
Do not include any additional text.
"""
    if feedback:
        prompt += f"\nFeedback: The optimal coloring computed is {feedback}. Please adjust your answer accordingly."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in graph theory."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "text"},
        temperature=0.5
    )
    raw_response = response.choices[0].message.content
    print("GPT-4 raw response:", raw_response)
    return extract_final_coloring(raw_response)

def graph_coloring_solver(adjacency_matrix, num_colors):
    """
    Solves the graph coloring problem using OR-Tools CP-SAT.
    adjacency_matrix: 2D list (n x n) with 0/1.
    num_colors: maximum number of colors.
    Returns a dictionary mapping vertex (int) to color index (int).
    """
    n = len(adjacency_matrix)
    model = cp_model.CpModel()
    # Create variables: color for each vertex
    colors = [model.NewIntVar(0, num_colors - 1, f"color_{i}") for i in range(n)]
    
    # Add constraints: for every edge (i, j) with i < j and adjacency_matrix[i][j] == 1, enforce different colors.
    for i in range(n):
        for j in range(i+1, n):
            if adjacency_matrix[i][j]:
                model.Add(colors[i] != colors[j])
    
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return {i: solver.Value(colors[i]) for i in range(n)}
    else:
        return None

def compare_graph_coloring_solutions(llm_solution, or_tools_solution):
    """
    Compares two graph coloring solutions (dictionaries mapping vertex to color index).
    Returns True if they match exactly.
    """
    return llm_solution == or_tools_solution

def get_improved_graph_coloring_solution(adjacency_matrix, num_colors, max_iterations=5):
    """
    Iteratively queries GPT-4 with feedback until its graph coloring solution matches
    the OR-Tools solution or max_iterations is reached.
    Returns a tuple: (llm_solution, or_tools_solution, iterations, correct)
    """
    llm_solution = get_gpt4_graph_coloring_solution(adjacency_matrix, num_colors)
    or_tools_solution = graph_coloring_solver(adjacency_matrix, num_colors)
    correct = compare_graph_coloring_solutions(llm_solution, or_tools_solution)
    iteration = 0
    while not correct and iteration < max_iterations:
        if or_tools_solution is None:
            feedback = ""
        else:
            feedback = ", ".join([f"{k}:{or_tools_solution[k]}" for k in sorted(or_tools_solution.keys())])
        llm_solution = get_gpt4_graph_coloring_solution(adjacency_matrix, num_colors, feedback=feedback)
        correct = compare_graph_coloring_solutions(llm_solution, or_tools_solution)
        iteration += 1
    return llm_solution, or_tools_solution, iteration, correct
