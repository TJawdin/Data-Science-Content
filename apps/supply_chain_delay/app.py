from pathlib import Path

# Other code...

ARTIFACTS = Path(__file__).parent / "artifacts"

# Other code...

sample_available = (Path(__file__).parent / "data/sample_input.csv").exists()

# Other code...

with open(Path(__file__).parent / "data/sample_input.csv", "rb") as f:
    # Process the file
    pass

# Other code...