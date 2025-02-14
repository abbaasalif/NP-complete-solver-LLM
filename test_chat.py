import pytest

# Test case for default inputs (i.e. pressing Enter to use defaults)
def test_chat_agent_default(monkeypatch, capsys):
    """
    Simulate a user who accepts the default cities and default distance matrix.
    The agent should use the default values.
    """
    # For default test, simulate:
    # - City names: empty (-> defaults to ["A", "B", "C", "D"])
    # - Then for each of the 4 cities, first an empty input (to trigger an error) then a valid row.
    inputs = iter([
        "",                           # Cities input: empty => defaults to A, B, C, D
        "", "0, 10, 15, 20",           # Row for city A: first empty then valid row
        "", "10, 0, 35, 25",           # Row for city B
        "", "15, 35, 0, 30",           # Row for city C
        "", "20, 25, 30, 0",           # Row for city D
    ])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    
    # Patch the GPT-4o solution function to return a fixed result.
    from tsp_module import chat_tsp_agent
    monkeypatch.setattr("tsp_module.get_gpt4o_tsp_solution", 
                        lambda dm, cities: "Optimal path: A → B → D → C → A")
    
    # Run the chat agent
    chat_tsp_agent()
    
    # Capture printed output
    captured = capsys.readouterr().out
    
    # Check for key expected output strings.
    assert "GPT-4o Solution:" in captured
    assert "Optimal Solution:" in captured
    assert "Final LLM Solution:" in captured
    
    # Print captured output if test is successful.
    print("\n--- Captured output for default test ---")
    print(captured)
    print("--- End of captured output ---\n")


# Test case for custom user inputs
def test_chat_agent_custom(monkeypatch, capsys):
    """
    Simulate a user who provides custom city names and a valid custom distance matrix.
    """
    # For custom test:
    # - Custom cities: "X, Y, Z"
    # - Then one valid row per city.
    inputs = iter([
        "X, Y, Z",       # Custom city names
        "0, 5, 10",      # Row for X
        "5, 0, 3",       # Row for Y
        "10, 3, 0",      # Row for Z
    ])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    
    from tsp_module import chat_tsp_agent
    monkeypatch.setattr("tsp_module.get_gpt4o_tsp_solution", 
                        lambda dm, cities: "Optimal path: X → Y → Z → X")
    
    chat_tsp_agent()
    captured = capsys.readouterr().out
    
    assert "Cities: ['X', 'Y', 'Z']" in captured
    assert "GPT-4o Solution:" in captured
    assert "Optimal Solution:" in captured
    assert "Final LLM Solution:" in captured
    
    print("\n--- Captured output for custom test ---")
    print(captured)
    print("--- End of captured output ---\n")


# Test case for insufficient matrix input
def test_chat_agent_insufficient_matrix(monkeypatch, capsys):
    """
    Simulate a scenario where the user enters custom city names but initially provides an empty input
    for each row of the distance matrix. The agent will re-prompt until valid input is provided.
    """
    # For insufficient matrix test:
    # - Custom cities: "A, B, C"
    # - For each of the 3 cities, first an empty input then a valid row.
    inputs = iter([
        "A, B, C",         # Custom city names
        "", "0, 10, 15",    # Row for A: first empty then valid row
        "", "10, 0, 35",    # Row for B
        "", "15, 35, 0",    # Row for C
    ])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    
    from tsp_module import chat_tsp_agent
    monkeypatch.setattr("tsp_module.get_gpt4o_tsp_solution",
                        lambda dm, cities: "Optimal path: A → B → C → A")
    
    chat_tsp_agent()
    captured = capsys.readouterr().out
    
    # Check that the re-prompt message appears and that custom cities are printed.
    assert "Input cannot be empty. Please enter the distances" in captured
    assert "Cities: ['A', 'B', 'C']" in captured
    assert "GPT-4o Solution:" in captured
    assert "Optimal Solution:" in captured
    assert "Final LLM Solution:" in captured
    
    print("\n--- Captured output for insufficient matrix test ---")
    print(captured)
    print("--- End of captured output ---\n")


# Test case for invalid matrix input
def test_chat_agent_invalid_matrix(monkeypatch, capsys):
    """
    Simulate a scenario where the user enters custom city names and initially provides non-numeric
    values for each row of the distance matrix. The agent will print an error message and re-prompt.
    Then valid input is provided.
    """
    # For invalid matrix test:
    # - Custom cities: "A, B, C"
    # - For each city, first an invalid row then a valid row.
    inputs = iter([
        "A, B, C",                # Custom city names
        "a, b, c", "0, 10, 15",     # Row for A: invalid then valid row
        "x, y, z", "10, 0, 35",     # Row for B: invalid then valid row
        "hello", "15, 35, 0",       # Row for C: invalid then valid row
    ])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    
    from tsp_module import collect_tsp_details
    cities, matrix = collect_tsp_details()
    captured = capsys.readouterr().out
    
    # Assert that error messages appear
    assert "Error:" in captured
    assert cities == ['A', 'B', 'C']
    
    print("\n--- Captured output for invalid matrix test ---")
    print(captured)
    print("--- End of captured output ---\n")
