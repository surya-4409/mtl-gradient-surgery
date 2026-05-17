import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from src.dataset import get_datasets
from src.models import MultiTaskModel
import config

def main():
    print("Loading datasets...")
    train_dataset, val_dataset = get_datasets(batch_size=config.BATCH_SIZE)

    print("Initializing PCGrad model, losses, and optimizer...")
    model = MultiTaskModel()
    loss_fn_a = tf.keras.losses.BinaryCrossentropy()
    loss_fn_b = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)

    val_acc_a = tf.keras.metrics.BinaryAccuracy(name='val_acc_a')
    val_acc_b = tf.keras.metrics.BinaryAccuracy(name='val_acc_b')
    val_loss_a = tf.keras.metrics.Mean(name='val_loss_a')
    val_loss_b = tf.keras.metrics.Mean(name='val_loss_b')

    history = []
    conflict_log = []
    global_step = tf.Variable(0, dtype=tf.int64)

    @tf.function
    def train_step_pcgrad(inputs, labels_a, labels_b, step):
        with tf.GradientTape(persistent=True) as tape:
            pred_a, pred_b = model(inputs, training=True)
            loss_a = loss_fn_a(labels_a, pred_a)
            loss_b = loss_fn_b(labels_b, pred_b)

        grad_a = tape.gradient(loss_a, model.backbone.trainable_variables)
        grad_b = tape.gradient(loss_b, model.backbone.trainable_variables)
        grad_head_a = tape.gradient(loss_a, model.head_a.trainable_variables)
        grad_head_b = tape.gradient(loss_b, model.head_b.trainable_variables)
        del tape 

        flat_grad_a = tf.concat([tf.reshape(g, [-1]) for g in grad_a if g is not None], axis=0)
        flat_grad_b = tf.concat([tf.reshape(g, [-1]) for g in grad_b if g is not None], axis=0)

        dot_product = tf.reduce_sum(flat_grad_a * flat_grad_b)
        norm_a = tf.norm(flat_grad_a) + 1e-8
        norm_b = tf.norm(flat_grad_b) + 1e-8
        cosine_sim = dot_product / (norm_a * norm_b)

        if cosine_sim < 0:
            modified_grad_a = []
            modified_grad_b = []
            for ga, gb in zip(grad_a, grad_b):
                if ga is not None and gb is not None:
                    dot_ab = tf.reduce_sum(ga * gb)
                    ga_proj = ga - (dot_ab / (tf.norm(gb)**2 + 1e-8)) * gb
                    gb_proj = gb - (dot_ab / (tf.norm(ga)**2 + 1e-8)) * ga
                    modified_grad_a.append(ga_proj)
                    modified_grad_b.append(gb_proj)
                else:
                    modified_grad_a.append(ga)
                    modified_grad_b.append(gb)
            
            grad_a, grad_b = modified_grad_a, modified_grad_b

        final_gradients_backbone = [ga + gb if ga is not None and gb is not None else ga or gb 
                                    for ga, gb in zip(grad_a, grad_b)]

        all_vars = model.backbone.trainable_variables + model.head_a.trainable_variables + model.head_b.trainable_variables
        all_grads = final_gradients_backbone + grad_head_a + grad_head_b

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
    for epoch in range(config.EPOCHS):
        for inputs, (labels_a, labels_b) in train_dataset:
            _, _, cosine_sim = train_step_pcgrad(inputs, labels_a, labels_b, global_step)
            
            conflict_log.append({
                'step': int(global_step.numpy()),
                'cosine_similarity': float(cosine_sim.numpy())
            })
            global_step.assign_add(1)

        val_acc_a.reset_states()
        val_acc_b.reset_states()
        val_loss_a.reset_states()
        val_loss_b.reset_states()

        for inputs, (labels_a, labels_b) in val_dataset:
            val_step(inputs, labels_a, labels_b)

        epoch_metrics = {
            'epoch': epoch + 1,
            'val_loss_a': float(val_loss_a.result()),
            'val_loss_b': float(val_loss_b.result()),
            'val_acc_a': float(val_acc_a.result()),
            'val_acc_b': float(val_acc_b.result()),
        }
        history.append(epoch_metrics)
        print(f"Epoch {epoch + 1}/{config.EPOCHS} | Val Acc A: {epoch_metrics['val_acc_a']:.4f} | Val Acc B: {epoch_metrics['val_acc_b']:.4f}")

    # Save Outputs utilizing config paths
    df_history = pd.DataFrame(history)
    df_history.to_csv(config.PCGRAD_METRICS_PATH, index=False)
    
    df_conflict = pd.DataFrame(conflict_log)
    df_conflict.to_csv(config.GRADIENT_CONFLICT_PATH, index=False)
    
    try:
        baseline_df = pd.read_csv(config.BASELINE_METRICS_PATH)
        baseline_final = baseline_df.iloc[-1].to_dict()
    except FileNotFoundError:
        print("Warning: baseline_metrics.csv not found. Run train_baseline.py first.")
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

    with open(config.FINAL_METRICS_PATH, 'w') as f:
        json.dump(final_metrics_payload, f, indent=4)

    print(f"PCGrad training complete. Metrics saved to {config.RESULTS_DIR}/")

if __name__ == '__main__':
    main()