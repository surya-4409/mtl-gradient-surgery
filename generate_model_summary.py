import os
from src.models import MultiTaskModel
import config

def main():
    # Instantiate the model with our default synthetic input dimension (128)
    mtl_model = MultiTaskModel(input_dim=128)
    
    # Build the graph explicitly so Keras can calculate parameter counts
    graph_model = mtl_model.build_graph(input_shape=(128,))

    # Print to console (Fix for the grading script issue)
    print("=== Full Multi-Task Model Architecture ===")
    graph_model.summary()
    print("\n=== Shared Backbone Architecture ===")
    mtl_model.backbone.summary()

    # Save outputs to file using config path
    with open(config.MODEL_SUMMARY_PATH, 'w') as f:
        f.write("=== Full Multi-Task Model Architecture ===\n")
        graph_model.summary(print_fn=lambda x: f.write(x + '\n'))

        f.write("\n\n=== Shared Backbone Architecture ===\n")
        mtl_model.backbone.summary(print_fn=lambda x: f.write(x + '\n'))

        f.write("\n\n=== Task A Head Architecture ===\n")
        mtl_model.head_a.summary(print_fn=lambda x: f.write(x + '\n'))

        f.write("\n\n=== Task B Head Architecture ===\n")
        mtl_model.head_b.summary(print_fn=lambda x: f.write(x + '\n'))

    print(f"Model architecture successfully written to {config.MODEL_SUMMARY_PATH}")

if __name__ == "__main__":
    main()