import os
from src.models import MultiTaskModel

def main():
    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)
    output_path = 'results/model_architecture.txt'

    # Instantiate the model with our default synthetic input dimension (128)
    mtl_model = MultiTaskModel(input_dim=128)
    
    # Build the graph explicitly so Keras can calculate parameter counts
    graph_model = mtl_model.build_graph(input_shape=(128,))

    with open(output_path, 'w') as f:
        f.write("=== Full Multi-Task Model Architecture ===\n")
        # Route the print output directly into the file
        graph_model.summary(print_fn=lambda x: f.write(x + '\n'))

        f.write("\n\n=== Shared Backbone Architecture ===\n")
        mtl_model.backbone.summary(print_fn=lambda x: f.write(x + '\n'))

        f.write("\n\n=== Task A Head Architecture ===\n")
        mtl_model.head_a.summary(print_fn=lambda x: f.write(x + '\n'))

        f.write("\n\n=== Task B Head Architecture ===\n")
        mtl_model.head_b.summary(print_fn=lambda x: f.write(x + '\n'))

    print(f"Model architecture successfully written to {output_path}")

if __name__ == "__main__":
    main()