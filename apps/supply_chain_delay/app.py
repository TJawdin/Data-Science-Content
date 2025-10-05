# Updated app.py

# Other code...

# Change ARTIFACTS variable to use __file__ for correct path resolution
ARTIFACTS = Path(__file__).parent / "artifacts"

# Other code...

# Change sample_available to use __file__ for correct path resolution
sample_available = (Path(__file__).parent / "data/sample_input.csv").exists()

# Other code...

# Change file opening to use __file__ for correct path resolution
with open(Path(__file__).parent / "data/sample_input.csv", "rb") as f:
    # Process file
    pass

# Other code...