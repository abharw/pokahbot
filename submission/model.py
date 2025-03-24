# submission/model.py
import tensorflow as tf
import numpy as np


class PolicyNetwork(tf.keras.Model):
    """
    Neural network model for poker decision making
    """
    def __init__(self, input_dim, hidden_layers, output_dim):
        """
        Initialize the policy network
        
        Args:
            input_dim (int): Input dimension size
            hidden_layers (list): List of hidden layer sizes
            output_dim (int): Output dimension size
        """
        super(PolicyNetwork, self).__init__()
        
        # Create the model layers
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(input_dim,))
        
        # Hidden layers with batch normalization and dropout
        self.hidden_layers = []
        for i, size in enumerate(hidden_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(
                size, 
                activation='relu',
                kernel_initializer=tf.keras.initializers.HeNormal(),
                name=f'hidden_{i}'
            ))
            self.hidden_layers.append(tf.keras.layers.BatchNormalization(name=f'batch_norm_{i}'))
            self.hidden_layers.append(tf.keras.layers.Dropout(0.2, name=f'dropout_{i}'))
        
        # Output layer for action probabilities (first 4 outputs)
        self.action_output = tf.keras.layers.Dense(
            4, 
            activation='softmax',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            name='action_output'
        )
        
        # Output for raise sizing (one output, range 0-1)
        self.raise_output = tf.keras.layers.Dense(
            1, 
            activation='sigmoid',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            name='raise_output'
        )
        
        # Output for discard decision (one output, probability of discarding)
        self.discard_output = tf.keras.layers.Dense(
            1, 
            activation='sigmoid',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            name='discard_output'
        )
    
    def call(self, inputs, training=False):
        """
        Forward pass through the network
        
        Args:
            inputs (tf.Tensor): Input tensor
            training (bool): Whether in training mode
            
        Returns:
            tf.Tensor: Output tensor
        """
        x = self.input_layer(inputs)
        
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        
        # Get action probabilities
        action_probs = self.action_output(x)
        
        # Get raise sizing
        raise_size = self.raise_output(x)
        
        # Get discard decision
        discard_prob = self.discard_output(x)
        
        # Concatenate all outputs
        return tf.concat([action_probs, raise_size, discard_prob], axis=1)
    
    @tf.function
    def get_action(self, state, action_mask):
        """
        Get an action using the policy network
        
        Args:
            state (tf.Tensor): State tensor
            action_mask (tf.Tensor): Binary mask for valid actions
            
        Returns:
            tf.Tensor: Action index tensor
        """
        # Forward pass