import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from src.dataset import get_datasets
from src.models import MultiTaskModel

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

def main():
    os.makedirs('results', exist_ok=True)

    print("Loading datasets...")
    train_dataset, val_dataset = get_datasets(batch_size=BATCH_SIZE)

    print("Initializing PCGrad model, losses, and optimizer...")
    model = MultiTaskModel()
    loss_fn_a = tf.keras.losses.BinaryCrossentropy()
    loss_fn_b = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Trackers for validation metrics
    val_acc_a = tf.keras.metrics.BinaryAccuracy(name='val_acc_a')
    val_acc_b = tf.keras.metrics.BinaryAccuracy(name='val_acc_b')
    val_loss_a = tf.keras.metrics.Mean(name='val_loss_a')
    val_loss_b = tf.keras.metrics.Mean(name='val_loss_b')

    # Data structures for logging
    history = []
    conflict_log = []
    global_step = tf.Variable(0, dtype=tf.int64)

    @tf.function
    def train_step_pcgrad(inputs, labels_a, labels_b, step):
        # 1. Separate Gradient Calculation
        with tf.GradientTape(persistent=True) as tape:
            pred_a, pred_b = model(inputs, training=True)
            loss_a = loss_fn_a(labels_a, pred_a)
            loss_b = loss_fn_b(labels_b, pred_b)

        # Calculate gradients for backbone and heads independently
        grad_a = tape.gradient(loss_a, model.backbone.trainable_variables)
        grad_b = tape.gradient(loss_b, model.backbone.trainable_variables)
        grad_head_a = tape.gradient(loss_a, model.head_a.trainable_variables)
        grad_head_b = tape.gradient(loss_b, model.head_b.trainable_variables)
        del tape  # Release persistent tape resources

        # 2. Gradient Conflict Detection (Cosine Similarity)
        # Flatten the backbone gradients
        flat_grad_a = tf.concat([tf.reshape(g, [-1]) for g in grad_a if g is not None], axis=0)
        flat_grad_b = tf.concat([tf.reshape(g, [-1]) for g in grad_b if g is not None], axis=0)

        # Calculate cosine similarity with a small epsilon to avoid division by zero
        dot_product = tf.reduce_sum(flat_grad_a * flat_grad_b)
        norm_a = tf.norm(flat_grad_a) + 1e-8
        norm_b = tf.norm(flat_grad_b) + 1e-8
        cosine_sim = dot_product / (norm_a * norm_b)

        # 3. PCGrad Implementation (Orthogonal Projection)
        if cosine_sim < 0:
            modified_grad_a = []
            modified_grad_b = []
            for ga, gb in zip(grad_a, grad_b):
                if ga is not None and gb is not None:
                    # Layer-wise projection using the formula from Yu et al. (2020)
                    dot_ab = tf.reduce_sum(ga * gb)
                    ga_proj = ga - (dot_ab / (tf.norm(gb)**2 + 1e-8)) * gb
                    gb_proj = gb - (dot_ab / (tf.norm(ga)**2 + 1e-8)) * ga
                    modified_grad_a.append(ga_proj)
                    modified_grad_b.append(gb_proj)
                else:
                    modified_grad_a.append(ga)
                    modified_grad_b.append(gb)
            
            grad_a, grad_b = modified_grad_a, modified_grad_b

        # 4. Integrate into Training Loop
        # Combine the processed backbone gradients
        final_gradients_backbone = [ga + gb if ga is not None and gb is not None else ga or gb 
                                    for ga, gb in zip(grad_a, grad_b)]

        # Group all variables and their final gradients
        all_vars = model.backbone.trainable_variables + model.head_a.trainable_variables + model.head_b.trainable_variables
        all_grads = final_gradients_backbone + grad_head_a + grad_head_b

        # Apply gradients
        optimizer.apply_gradients(zip(all_grads, all_vars))
        
        return loss_a, loss_b, cosine_sim

    @tf.function
    def val_step(inputs, labels_a, labels_b):
        pred_a, pred_b = model(inputs, training=False)
        loss_a = loss_fn_a(labels_a, pred_a)
        loss_b = loss_fn_b(labels_b, pred_b)

        val_loss_a.update_state(loss_a)
        val_loss_b.update_state(loss_b)
        val_acc_a.update_state(labels_a, pred_a)
        val_acc_b.update_state(labels_b, pred_b)

    print("Starting PCGrad training loop...")
    for epoch in range(EPOCHS):
        # Training loop
        for inputs, (labels_a, labels_b) in train_dataset:
            _, _, cosine_sim = train_step_pcgrad(inputs, labels_a, labels_b, global_step)
            
            # Log the cosine similarity for the Streamlit monitor
            conflict_log.append({
                'step': int(global_step.numpy()),
                'cosine_similarity': float(cosine_sim.numpy())
            })
            global_step.assign_add(1)

        # Reset and run validation
        val_acc_a.reset_states()
        val_acc_b.reset_states()
        val_loss_a.reset_states()
        val_loss_b.reset_states()

        for inputs, (labels_a, labels_b) in val_dataset:
            val_step(inputs, labels_a, labels_b)

        # Record metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'val_loss_a': float(val_loss_a.result()),
            'val_loss_b': float(val_loss_b.result()),
            'val_acc_a': float(val_acc_a.result()),
            'val_acc_b': float(val_acc_b.result()),
        }
        history.append(epoch_metrics)
        
        print(f"Epoch {epoch + 1}/{EPOCHS} | "
              f"Val Acc A: {epoch_metrics['val_acc_a']:.4f} | "
              f"Val Acc B: {epoch_metrics['val_acc_b']:.4f}")

    # --- Save Outputs to fulfill rubrics ---
    
    # 1. Save PCGrad metrics
    df_history = pd.DataFrame(history)
    df_history.to_csv('results/pcgrad_metrics.csv', index=False)
    
    # 2. Save Gradient Conflict logs
    df_conflict = pd.DataFrame(conflict_log)
    df_conflict.to_csv('results/gradient_conflict.csv', index=False)
    
    # 3. Generate final_metrics.json comparing baseline and PCGrad
    try:
        baseline_df = pd.read_csv('results/baseline_metrics.csv')
        baseline_final = baseline_df.iloc[-1].to_dict()
    except FileNotFoundError:
        print("Warning: baseline_metrics.csv not found. Did you run train_baseline.py first?")
        baseline_final = {"val_acc_a": 0.0, "val_loss_a": 0.0, "val_acc_b": 0.0, "val_loss_b": 0.0}

    pcgrad_final = df_history.iloc[-1].to_dict()

    final_metrics_payload = {
        "baseline": {
            "task_a": {
                "accuracy": baseline_final["val_acc_a"],
                "loss": baseline_final["val_loss_a"]
            },
            "task_b": {
                "accuracy": baseline_final["val_acc_b"],
                "loss": baseline_final["val_loss_b"]
            }
        },
        "pcgrad": {
            "task_a": {
                "accuracy": pcgrad_final["val_acc_a"],
                "loss": pcgrad_final["val_loss_a"]
            },
            "task_b": {
                "accuracy": pcgrad_final["val_acc_b"],
                "loss": pcgrad_final["val_loss_b"]
            }
        }
    }

    with open('results/final_metrics.json', 'w') as f:
        json.dump(final_metrics_payload, f, indent=4)

    print("PCGrad training complete. Metrics and logs saved to results/ directory.")

if __name__ == '__main__':
    main()