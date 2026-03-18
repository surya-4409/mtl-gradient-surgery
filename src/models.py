import tensorflow as tf

class MultiTaskModel(tf.keras.Model):
    def __init__(self, input_dim=128, **kwargs):
        super(MultiTaskModel, self).__init__(**kwargs)
        
        # 1. Shared Backbone
        # Responsible for learning a general, shared representation
        self.backbone = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation='relu', name='shared_dense_1'),
            tf.keras.layers.BatchNormalization(name='shared_bn_1'),
            tf.keras.layers.Dense(64, activation='relu', name='shared_dense_2'),
            tf.keras.layers.BatchNormalization(name='shared_bn_2'),
        ], name='shared_backbone')
        
        # 2. Task A Head (Binary Classification)
        self.head_a = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', name='task_a_dense_1'),
            tf.keras.layers.Dense(1, activation='sigmoid', name='task_a_output')
        ], name='task_a_head')
        
        # 3. Task B Head (Binary Classification)
        self.head_b = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', name='task_b_dense_1'),
            tf.keras.layers.Dense(1, activation='sigmoid', name='task_b_output')
        ], name='task_b_head')

    def call(self, inputs, training=False):
        # Forward pass through shared backbone
        shared_representation = self.backbone(inputs, training=training)
        
        # Forward pass through task-specific heads
        pred_a = self.head_a(shared_representation, training=training)
        pred_b = self.head_b(shared_representation, training=training)
        
        return pred_a, pred_b

    def build_graph(self, input_shape):
        """
        Helper method to allow Keras to build the model graph explicitly.
        This makes model.summary() work perfectly with subclassed models.
        """
        x = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))