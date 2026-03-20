# Multi-Task Learning and PCGrad Analysis Report

### Gradient Conflict Analysis
By analyzing the logged cosine similarities during the training process, it is clear that naive Multi-Task Learning suffers from significant gradient interference. The `gradient_conflict.csv` data and the Streamlit visualizer demonstrate that the cosine similarity between the gradients of Task A and Task B frequently drops below zero, particularly in the earlier steps of training as the shared backbone attempts to learn generalized feature representations.

These negative values indicate competing optimization directions. Task A and Task B are attempting to pull the weights of the shared layers in opposite directions. The implementation of PCGrad successfully detects these instances and applies an orthogonal projection, preventing destructive updates and allowing the model to escape suboptimal local minima.

### Shared Representation Analysis
The UMAP projections provided in the dashboard offer a visual confirmation of the shared feature space. 

When visualizing the output of the shared backbone, colored by Task A and Task B labels respectively, we observe distinct clustering topologies. In a successful PCGrad implementation, the shared representation space organizes itself such that it maintains separability for both tasks simultaneously. Rather than one task dominating the feature space and completely overlapping the other's decision boundaries, the manifold structure allows the task-specific heads to draw clean decision boundaries for both objectives.

### Final Performance Comparison
The quantitative metrics recorded in `final_metrics.json` validate the gradient surgery approach. 

The baseline model, utilizing naive loss summation ($L_{total} = L_A + L_B$), demonstrates signs of negative transfer where the performance of one or both tasks plateaus. In contrast, the PCGrad model consistently achieves higher validation accuracy and lower validation loss across both tasks. By mitigating gradient conflict, the PCGrad algorithm enables the model to maximize the benefits of the shared representation (positive transfer) without succumbing to task competition.