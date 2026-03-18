import os
import tensorflow as tf
import pandas as pd
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

    print("Initializing model, losses, and optimizer...")
    model = MultiTaskModel()
    loss_fn_a = tf.keras.losses.BinaryCrossentropy()
    loss_fn_b = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Trackers for validation metrics
    val_acc_a = tf.keras.metrics.BinaryAccuracy(name='val_acc_a')
    val_acc_b = tf.keras.metrics.BinaryAccuracy(name='val_acc_b')
    val_loss_a = tf.keras.metrics.Mean(name='val_loss_a')
    val_loss_b = tf.keras.metrics.Mean(name='val_loss_b')

    @tf.function
    def train_step(inputs, labels_a, labels_b):
        with tf.GradientTape() as tape:
            pred_a, pred_b = model(inputs, training=True)
            loss_a = loss_fn_a(labels_a, pred_a)
            loss_b = loss_fn_b(labels_b, pred_b)
            
            # NAIVE SUMMATION: The baseline approach to MTL
            total_loss = loss_a + loss_b

        # Calculate and apply gradients for all trainable variables together
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return loss_a, loss_b

    @tf.function
    def val_step(inputs, labels_a, labels_b):
        pred_a, pred_b = model(inputs, training=False)
        loss_a = loss_fn_a(labels_a, pred_a)
        loss_b = loss_fn_b(labels_b, pred_b)

        val_loss_a.update_state(loss_a)
        val_loss_b.update_state(loss_b)
        val_acc_a.update_state(labels_a, pred_a)
        val_acc_b.update_state(labels_b, pred_b)

    history = []

    print("Starting baseline training loop...")
    for epoch in range(EPOCHS):
        # Training loop
        for inputs, (labels_a, labels_b) in train_dataset:
            train_step(inputs, labels_a, labels_b)

        # Reset validation metrics at the start of each epoch's validation phase
        val_acc_a.reset_states()
        val_acc_b.reset_states()
        val_loss_a.reset_states()
        val_loss_b.reset_states()

        # Validation loop
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

    # Save metrics to CSV for the Streamlit dashboard and final evaluation
    df = pd.DataFrame(history)
    df.to_csv('results/baseline_metrics.csv', index=False)
    print("Baseline metrics successfully saved to results/baseline_metrics.csv")

if __name__ == '__main__':
    main()