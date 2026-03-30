import os

# Define structure relative to current working directory
structure = {
    "README.md": "",
    "requirements.txt": "",

    "data": {
        "raw": {},
        "processed": {},
        "sample_routes.csv": ""
    },

    "notebooks": {
        "01_data_pipeline.ipynb": "",
        "02_route_model.ipynb": "",
        "03_results_analysis.ipynb": ""
    },

    "src": {
        "data_ingestion.py": "",
        "weather_model.py": "",
        "routing_model.py": "",
        "cost_function.py": ""
    },

    "outputs": {
        "route_maps.png": "",
        "cost_comparison.csv": "",
        "performance_summary.txt": ""
    },

    "docs": {
        "capability_statement.pdf": "",
        "methodology.md": ""
    }
}


def create_structure(base_path, structure_dict):
    for name, content in structure_dict.items():
        path = os.path.join(base_path, name)

        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            # Create empty file
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write("")


# Execute in current directory
create_structure(".", structure)

print("Repository structure created successfully.")