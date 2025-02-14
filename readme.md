# NP‑Complete Solver LLM

This project is a Flask‑based web application that leverages GPT‑4 and OR‑Tools to solve NP‑complete problems. The application supports several problems—including the Traveling Salesman Problem (TSP), 0/1 Knapsack Problem, and Graph Coloring Problem—by classifying natural language descriptions provided by the user and then iteratively refining GPT‑4’s solution using feedback from OR‑Tools until convergence.

## Features

- **Natural Language Problem Classification:**  
  Enter a plain language description of your NP‑complete problem. GPT‑4 classifies it (e.g., TSP, KNAPSACK, GRAPH_COLORING) and directs you to the appropriate form.

- **Detailed Input Forms:**  
  Based on the classification, you can provide structured details (cities and a distance matrix for TSP, items and capacity for Knapsack, or an adjacency matrix and number of colors for Graph Coloring).

- **Iterative Feedback Loop:**  
  For each problem, if GPT‑4’s solution does not match the solver’s solution, the system feeds back the solver’s output to GPT‑4 until the responses converge (or a maximum number of iterations is reached).

- **Rich Result Display:**  
  View both GPT‑4’s solution and the OR‑Tools/CP‑SAT solution along with iteration counts and additional details (such as selected items with weights and values for the knapsack problem).

## Project Structure

The repository has the following structure:

```
NP-complete-solver-LLM/
├── API_key.env
├── app.py
├── knapsack_module.py
├── graph_coloring_module.py
├── environment.yml
└── templates/
    ├── index.html
    ├── tsp.html
    ├── result.html
    ├── knapsack.html
    ├── result_knapsack.html
    ├── graph_coloring.html
    └── result_graph_coloring.html
```

- **API_key.env:** Contains your OpenAI API key (do not commit this file).
- **app.py:** Main Flask application file, containing routes for NP‑complete problem classification and specific problem detail forms.
- **knapsack_module.py:** Contains functions to solve the knapsack problem using GPT‑4 and OR‑Tools with an iterative feedback loop.
- **graph_coloring_module.py:** Contains functions to solve the graph coloring problem using GPT‑4 and OR‑Tools CP‑SAT with iterative feedback.
- **environment.yml:** Conda environment file for reproducing the environment.
- **templates/:** Contains HTML templates for the home page, input forms, and result pages for each problem.

## Installation with Conda

To create and activate a Conda environment based on your existing environment:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/abbaasalif/NP-complete-solver-LLM.git
   cd NP-complete-solver-LLM
   ```

2. **Create the Conda environment from the existing environment file:**

   If you have an existing environment and want to export it:
   
   ```bash
   conda env export > environment.yml
   ```
   
   Otherwise, use the provided `environment.yml` to create the environment:
   
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment:**

   ```bash
   conda activate np_solver
   ```

## Running the Application

1. **Start the Flask server:**

   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to [http://127.0.0.1:5000/](http://127.0.0.1:5000/). On the home page, you can either enter a natural language description of your NP‑complete problem or click one of the provided links to directly go to a problem form (TSP, Knapsack, or Graph Coloring).

## Sample Scenarios

### 1. Traveling Salesman Problem (TSP) – 20 Cities

**Natural Language Description:**

```
I have a complex traveling salesman problem involving 20 cities spread over a large region. I have a detailed distance matrix that specifies the travel cost between every pair of cities. I need to find the shortest route that visits each city exactly once and returns to the starting city.
```

**Detailed Input for TSP Form:**

- **Cities:**

  ```
  City1, City2, City3, ..., City20
  ```

- **Distance Matrix:**

  ```
  0, 56, 47, 78, 90, 33, 68, 59, 72, 88, 45, 60, 91, 37, 82, 49, 77, 52, 83, 65
  56, 0, 65, 49, 75, 60, 80, 45, 68, 70, 57, 63, 76, 52, 69, 85, 54, 66, 74, 61
  ...
  ```
  *(Complete with 20 rows)*

### 2. Knapsack Problem – 10 Items

**Natural Language Description:**

```
I need to solve a knapsack problem with 10 items, where each item has a weight and a value. My goal is to maximize the total value without exceeding the knapsack's capacity of 50. The items are provided in a detailed list.
```

**Detailed Input for Knapsack Form:**

- **Items (format "weight:value; ..."):**

  ```
  10:60; 20:100; 30:120; 25:90; 15:50; 5:30; 12:40; 18:70; 22:80; 8:25
  ```

- **Capacity:**

  ```
  50
  ```

### 3. Graph Coloring Problem – 6 Vertices

**Natural Language Description:**

```
I have a scheduling problem that can be modeled as a graph coloring problem. The graph is represented by an adjacency matrix for 6 vertices, and I need to assign a color (represented as a numeric index) to each vertex such that no two adjacent vertices have the same color. I want to use exactly 3 colors.
```

**Detailed Input for Graph Coloring Form:**

- **Adjacency Matrix:**

  ```
  0, 1, 1, 0, 0, 1
  1, 0, 1, 1, 0, 0
  1, 1, 0, 1, 1, 0
  0, 1, 1, 0, 1, 1
  0, 0, 1, 1, 0, 1
  1, 0, 0, 1, 1, 0
  ```

- **Number of Colors:**

  ```
  3
  ```
## Sample Scenarios

### 1. Traveling Salesman Problem (TSP)

#### Natural Language Description

```
I have a challenging traveling salesman problem involving 20 cities spread over a large region. The distances between these cities vary significantly, and I have compiled a detailed distance matrix. I need to determine the shortest possible route that visits each city exactly once and returns to the starting city. Please help me solve this problem.
```

#### Structured Input

- **Cities (comma‑separated):**

  ```
  City1, City2, City3, City4, City5, City6, City7, City8, City9, City10, City11, City12, City13, City14, City15, City16, City17, City18, City19, City20
  ```

- **Distance Matrix (one row per line, comma‑separated):**

  ```
  0, 56, 47, 78, 90, 33, 68, 59, 72, 88, 45, 60, 91, 37, 82, 49, 77, 52, 83, 65
  56, 0, 65, 49, 75, 60, 80, 45, 68, 70, 57, 63, 76, 52, 69, 85, 54, 66, 74, 61
  47, 65, 0, 68, 54, 77, 59, 82, 63, 88, 45, 71, 90, 56, 67, 73, 50, 64, 80, 59
  78, 49, 68, 0, 62, 74, 59, 70, 81, 55, 68, 80, 72, 63, 58, 66, 77, 52, 69, 60
  90, 75, 54, 62, 0, 68, 73, 66, 80, 72, 64, 79, 85, 60, 57, 69, 71, 63, 82, 55
  33, 60, 77, 74, 68, 0, 55, 68, 73, 62, 59, 65, 70, 58, 62, 80, 53, 64, 71, 60
  68, 80, 59, 59, 73, 55, 0, 66, 70, 54, 62, 74, 80, 57, 63, 68, 69, 60, 75, 55
  59, 45, 82, 70, 66, 68, 66, 0, 64, 72, 53, 59, 77, 56, 68, 63, 58, 71, 67, 60
  72, 68, 63, 81, 80, 73, 70, 64, 0, 65, 60, 68, 74, 59, 70, 75, 62, 66, 72, 58
  88, 70, 88, 55, 72, 62, 54, 72, 65, 0, 70, 60, 66, 55, 68, 57, 73, 59, 80, 61
  45, 57, 45, 68, 64, 59, 62, 53, 60, 70, 0, 64, 78, 55, 61, 66, 70, 54, 65, 59
  60, 63, 71, 80, 79, 65, 74, 59, 68, 60, 64, 0, 75, 57, 66, 69, 62, 70, 63, 58
  91, 76, 90, 72, 85, 70, 80, 77, 74, 66, 78, 75, 0, 62, 68, 64, 79, 60, 71, 65
  37, 52, 56, 63, 60, 58, 57, 56, 59, 55, 55, 57, 62, 0, 61, 68, 54, 65, 60, 52
  82, 69, 67, 58, 57, 62, 63, 68, 70, 68, 61, 66, 68, 61, 0, 59, 63, 60, 75, 58
  49, 85, 73, 66, 69, 80, 68, 63, 75, 57, 66, 69, 64, 68, 59, 0, 70, 54, 65, 60
  77, 54, 50, 77, 71, 53, 69, 58, 62, 73, 70, 62, 79, 54, 63, 70, 0, 66, 60, 57
  52, 66, 64, 52, 63, 64, 60, 71, 66, 59, 54, 70, 60, 65, 60, 54, 66, 0, 68, 59
  83, 74, 80, 69, 82, 71, 75, 67, 72, 80, 65, 63, 71, 60, 75, 65, 60, 68, 0, 62
  65, 61, 59, 60, 55, 60, 55, 60, 58, 61, 59, 58, 65, 52, 58, 60, 57, 59, 62, 0
  ```

---

### 2. Knapsack Problem

#### Natural Language Description

```
I need to solve a knapsack problem where I have a list of 10 items, each with a specific weight and value. My goal is to select a subset of these items that maximizes the total value without exceeding a knapsack capacity of 50. Please help me determine the optimal selection.
```

#### Detailed Input

- **Items (format "weight:value; weight:value; ..."):**

  ```
  10:60; 20:100; 30:120; 25:90; 15:50; 5:30; 12:40; 18:70; 22:80; 8:25
  ```

- **Capacity:**

  ```
  50
  ```

---

### 3. Graph Coloring Problem

#### Natural Language Description

```
I have a scheduling problem that can be modeled as a graph coloring problem. Each vertex in the graph represents a class, and an edge between two vertices indicates that the classes conflict (cannot be scheduled at the same time). I need to assign colors (numeric indices) to each class so that no two adjacent vertices share the same color, using exactly 3 colors. Please help me find a valid coloring solution.
```

#### Detailed Input

- **Adjacency Matrix (one row per line, comma-separated):**

  ```
  0, 1, 0, 1, 0, 1
  1, 0, 1, 1, 0, 0
  0, 1, 0, 1, 1, 0
  1, 1, 1, 0, 1, 1
  0, 0, 1, 1, 0, 1
  1, 0, 0, 1, 1, 0
  ```

- **Number of Colors:**

  ```
  3
  ```

---

### How to Use These Scenarios

- **For TSP:**  
  Paste the natural language description into the home page’s description field. Once the problem is classified as TSP, use the detailed input (cities and the 20×20 distance matrix) in the TSP form to solve the problem.

- **For Knapsack:**  
  Use the provided natural language description on the home page to have the system classify it as a knapsack problem. Then, in the knapsack form, paste the items string and enter the capacity (50).

- **For Graph Coloring:**  
  Enter the natural language description to have the system classify it as a graph coloring problem. In the graph coloring form, paste the provided adjacency matrix and enter the number of colors (3).