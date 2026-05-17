import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def generate_synthetic_data(num_samples=5000, input_dim=128):
    np.random.seed(42)
    X = np.random.randn(num_samples, input_dim).astype(np.float32)

    hidden_a = np.sum(X[:, :input_dim//2], axis=1)
    y_a = (hidden_a > 0).astype(np.float32)

    hidden_b = np.sum(X[:, :input_dim//4], axis=1) - np.sum(X[:, input_dim//2:3*input_dim//4], axis=1)
    y_b = (hidden_b > 0).astype(np.float32)

    return X, y_a, y_b

def create_tf_dataset(X, y_a, y_b, batch_size=32, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, (y_a, y_b)))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(X))
    else:
        # OPTIMIZATION ADDED HERE: Cache the validation dataset
        dataset = dataset.cache()
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def get_datasets(batch_size=32, input_dim=128):
    X, y_a, y_b = generate_synthetic_data(input_dim=input_dim)
    
    X_train, X_val, ya_train, ya_val, yb_train, yb_val = train_test_split(
        X, y_a, y_b, test_size=0.2, random_state=42
    )
    
    train_dataset = create_tf_dataset(X_train, ya_train, yb_train, batch_size, is_training=True)
    val_dataset = create_tf_dataset(X_val, ya_val, yb_val, batch_size, is_training=False)
    
    return train_dataset, val_dataset   