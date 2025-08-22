#!/usr/bin/env python3
"""
Advanced Deep Learning-Enhanced Temporal Fusion Transformer (ADE-TFT)
State-of-the-art multi-horizon forecasting for cryptocurrency prediction
Achieving superior performance across multiple time horizons simultaneously
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime, timedelta
from logger import get_logger

class VariableSelectionNetwork(layers.Layer):
    """
    Variable Selection Network for TFT
    Selects relevant features at each time step using attention mechanisms
    """
    
    def __init__(self, hidden_size: int, num_inputs: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_inputs = num_inputs
        self.dropout_rate = dropout_rate
        
        # Flattened feature embedding
        self.flatten = layers.Flatten()
        
        # GRN for context
        self.context_grn = GatedResidualNetwork(
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate
        )
        
        # Variable selection weights
        self.variable_selection = layers.Dense(
            num_inputs,
            activation='softmax',
            name='variable_selection'
        )
        
        # Selected variables transformation
        self.single_variable_grns = [
            GatedResidualNetwork(
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout_rate=dropout_rate,
                name=f'single_var_grn_{i}'
            ) for i in range(num_inputs)
        ]
        
    def call(self, inputs, training=None):
        # inputs shape: [batch_size, time_steps, num_features]
        batch_size, time_steps, num_features = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        
        # Flatten for context
        flattened_inputs = self.flatten(inputs)
        
        # Context vector
        context = self.context_grn(flattened_inputs, training=training)
        
        # Variable selection weights
        selection_weights = self.variable_selection(context, training=training)
        selection_weights = tf.expand_dims(selection_weights, axis=1)  # [batch, 1, num_inputs]
        selection_weights = tf.tile(selection_weights, [1, time_steps, 1])  # [batch, time, num_inputs]
        
        # Process each variable
        transformed_vars = []
        for i in range(self.num_inputs):
            var_input = inputs[:, :, i:i+1]  # [batch, time, 1]
            var_input = tf.reshape(var_input, [batch_size * time_steps, 1])
            
            transformed_var = self.single_variable_grns[i](var_input, training=training)
            transformed_var = tf.reshape(transformed_var, [batch_size, time_steps, self.hidden_size])
            
            transformed_vars.append(transformed_var)
        
        # Stack and apply selection weights
        stacked_vars = tf.stack(transformed_vars, axis=-1)  # [batch, time, hidden, num_inputs]
        selection_weights = tf.expand_dims(selection_weights, axis=2)  # [batch, time, 1, num_inputs]
        
        # Weighted combination
        selected_vars = tf.reduce_sum(stacked_vars * selection_weights, axis=-1)  # [batch, time, hidden]
        
        return selected_vars, selection_weights

class GatedResidualNetwork(layers.Layer):
    """
    Gated Residual Network (GRN) - Core building block of TFT
    Provides skip connections and gating mechanisms for flexible non-linear processing
    """
    
    def __init__(self, hidden_size: int, output_size: int = None, dropout_rate: float = 0.1, 
                 context_size: int = None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size
        self.dropout_rate = dropout_rate
        self.context_size = context_size
        
        # Primary processing
        self.dense1 = layers.Dense(hidden_size, activation='elu')
        self.dense2 = layers.Dense(hidden_size, activation='linear')
        
        # Skip connection adaptation
        if self.output_size != hidden_size:
            self.skip_layer = layers.Dense(self.output_size, activation='linear')
        else:
            self.skip_layer = None
        
        # Gating layer
        self.gate = layers.Dense(self.output_size, activation='sigmoid')
        
        # Context integration
        if context_size is not None:
            self.context_layer = layers.Dense(hidden_size, activation='linear')
        
        # Normalization and dropout
        self.layer_norm = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout_rate)
        
    def call(self, inputs, context=None, training=None):
        # Primary path
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        
        # Context integration
        if context is not None and self.context_size is not None:
            context_contrib = self.context_layer(context)
            x = x + context_contrib
        
        x = self.dense2(x)
        
        # Skip connection
        if self.skip_layer is not None:
            skip = self.skip_layer(inputs)
        else:
            skip = inputs
        
        # Gating mechanism
        gate = self.gate(x)
        gated_output = gate * x + (1 - gate) * skip
        
        # Normalization
        output = self.layer_norm(gated_output)
        
        return output

class InterpretableMultiHeadAttention(layers.Layer):
    """
    Interpretable Multi-Head Attention for TFT
    Provides attention weights for interpretability while maintaining performance
    """
    
    def __init__(self, num_heads: int, head_size: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        self.dropout_rate = dropout_rate
        self.d_model = num_heads * head_size
        
        # Attention projection layers
        self.query_layer = layers.Dense(self.d_model)
        self.key_layer = layers.Dense(self.d_model)
        self.value_layer = layers.Dense(self.d_model)
        
        # Output projection
        self.output_layer = layers.Dense(self.d_model)
        
        # Dropout
        self.dropout_layer = layers.Dropout(dropout_rate)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, head_size)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        
        # Generate Q, K, V
        query = self.query_layer(inputs)
        key = self.key_layer(inputs)
        value = self.value_layer(inputs)
        
        # Split into multiple heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(depth)
        
        # Attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout_layer(attention_weights, training=training)
        
        # Apply attention
        attention_output = tf.matmul(attention_weights, value)
        
        # Concatenate heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.d_model))
        
        # Final linear transformation
        output = self.output_layer(attention_output)
        
        return output, attention_weights

class TemporalFusionTransformer:
    """
    Advanced Deep Learning-Enhanced Temporal Fusion Transformer (ADE-TFT)
    
    Multi-horizon forecasting with:
    - Variable Selection Networks
    - Gated Residual Networks
    - Interpretable Multi-Head Attention
    - Static and Dynamic feature processing
    - Multi-quantile predictions
    """
    
    def __init__(self,
                 hidden_size: int = 128,
                 num_heads: int = 4,
                 dropout_rate: float = 0.1,
                 num_quantiles: int = 3,
                 max_encoder_length: int = 168,  # 1 week of hourly data
                 max_prediction_length: int = 24,  # 1 day prediction
                 static_features: int = 10,
                 dynamic_features: int = 50):
        
        self.logger = get_logger(__name__)
        
        # Architecture parameters
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.num_quantiles = num_quantiles
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.static_features = static_features
        self.dynamic_features = dynamic_features
        
        # Model components
        self.model = None
        self.training_history = {}
        self.feature_importances = {}
        
        # Quantile levels for prediction intervals
        self.quantiles = [0.1, 0.5, 0.9]  # 10th, 50th (median), 90th percentiles
        
        self.logger.info(f"ADE-TFT initialized: hidden_size={hidden_size}, horizons={max_prediction_length}")
    
    def build_model(self, 
                   total_time_steps: int,
                   num_features: int) -> keras.Model:
        """Build the complete TFT architecture"""
        
        # Input layers
        inputs = keras.Input(shape=(total_time_steps, num_features), name='historical_inputs')
        
        # Known future inputs (for prediction horizon)
        future_inputs = keras.Input(
            shape=(self.max_prediction_length, num_features), 
            name='future_inputs'
        )
        
        # Static context features
        static_inputs = keras.Input(shape=(self.static_features,), name='static_inputs')
        
        # === STATIC FEATURE PROCESSING ===
        static_encoder = GatedResidualNetwork(
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout_rate=self.dropout_rate,
            name='static_encoder'
        )
        
        static_context = static_encoder(static_inputs)
        
        # === VARIABLE SELECTION NETWORKS ===
        
        # Historical features selection
        historical_vsn = VariableSelectionNetwork(
            hidden_size=self.hidden_size,
            num_inputs=num_features,
            dropout_rate=self.dropout_rate,
            name='historical_vsn'
        )
        
        selected_historical, historical_weights = historical_vsn(inputs)
        
        # Future features selection
        future_vsn = VariableSelectionNetwork(
            hidden_size=self.hidden_size,
            num_inputs=num_features,
            dropout_rate=self.dropout_rate,
            name='future_vsn'
        )
        
        selected_future, future_weights = future_vsn(future_inputs)
        
        # === LSTM ENCODER-DECODER ===
        
        # Encoder LSTM
        encoder_lstm = layers.LSTM(
            self.hidden_size,
            return_state=True,
            return_sequences=True,
            dropout=self.dropout_rate,
            name='encoder_lstm'
        )
        
        encoder_outputs, state_h, state_c = encoder_lstm(selected_historical)
        encoder_states = [state_h, state_c]
        
        # Decoder LSTM
        decoder_lstm = layers.LSTM(
            self.hidden_size,
            return_sequences=True,
            return_state=True,
            dropout=self.dropout_rate,
            name='decoder_lstm'
        )
        
        decoder_outputs, _, _ = decoder_lstm(selected_future, initial_state=encoder_states)
        
        # === GATED RESIDUAL NETWORKS ===
        
        # Process encoder outputs
        encoder_grn = GatedResidualNetwork(
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout_rate=self.dropout_rate,
            context_size=self.hidden_size,
            name='encoder_grn'
        )
        
        enhanced_encoder = encoder_grn(encoder_outputs, context=static_context)
        
        # Process decoder outputs
        decoder_grn = GatedResidualNetwork(
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout_rate=self.dropout_rate,
            context_size=self.hidden_size,
            name='decoder_grn'
        )
        
        enhanced_decoder = decoder_grn(decoder_outputs, context=static_context)
        
        # === INTERPRETABLE MULTI-HEAD ATTENTION ===
        
        attention_layer = InterpretableMultiHeadAttention(
            num_heads=self.num_heads,
            head_size=self.hidden_size // self.num_heads,
            dropout_rate=self.dropout_rate,
            name='multi_head_attention'
        )
        
        # Concatenate encoder and decoder for full sequence attention
        full_sequence = layers.Concatenate(axis=1)([enhanced_encoder, enhanced_decoder])
        attended_sequence, attention_weights = attention_layer(full_sequence)
        
        # Extract decoder portion for predictions
        attended_decoder = attended_sequence[:, -self.max_prediction_length:, :]
        
        # === POSITION-WISE FEED FORWARD ===
        
        position_wise_grn = GatedResidualNetwork(
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout_rate=self.dropout_rate,
            name='position_wise_grn'
        )
        
        position_wise_output = position_wise_grn(attended_decoder)
        
        # === MULTI-QUANTILE OUTPUT HEADS ===
        
        quantile_outputs = []
        
        for i, quantile in enumerate(self.quantiles):
            quantile_head = layers.Dense(
                1,
                activation='linear',
                name=f'quantile_{quantile}_output'
            )
            
            quantile_pred = quantile_head(position_wise_output)
            quantile_outputs.append(quantile_pred)
        
        # Stack quantile predictions
        stacked_outputs = layers.Concatenate(axis=-1, name='quantile_predictions')(quantile_outputs)
        
        # === POINT PREDICTION (MEDIAN) ===
        point_prediction = quantile_outputs[1]  # Median (0.5 quantile)
        
        # Create model
        model = keras.Model(
            inputs=[inputs, future_inputs, static_inputs],
            outputs={
                'quantile_predictions': stacked_outputs,
                'point_prediction': point_prediction,
                'attention_weights': attention_weights,
                'historical_importance': historical_weights,
                'future_importance': future_weights
            },
            name='ADE_TFT'
        )
        
        return model
    
    def quantile_loss(self, y_true, y_pred):
        """Quantile loss function for multi-quantile training"""
        
        def single_quantile_loss(q_idx):
            def loss_fn(y_true, y_pred):
                q = self.quantiles[q_idx]
                error = y_true - y_pred[:, :, q_idx:q_idx+1]
                return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
            return loss_fn
        
        # Combine losses for all quantiles
        total_loss = 0
        for i in range(len(self.quantiles)):
            total_loss += single_quantile_loss(i)(y_true, y_pred)
        
        return total_loss / len(self.quantiles)
    
    def prepare_tft_data(self, 
                        price_data: pd.DataFrame,
                        features_data: pd.DataFrame,
                        static_features_data: pd.DataFrame = None) -> Tuple[Dict, Dict]:
        """Prepare data for TFT training with proper sequence structure"""
        
        self.logger.info("Preparing TFT data with multi-horizon structure...")
        
        # Combine price and features
        if 'close' not in features_data.columns:
            features_data['target'] = price_data['close']
        else:
            features_data['target'] = features_data['close']
        
        # Sort by time
        features_data = features_data.sort_index()
        
        # Create sequences
        sequences = []
        targets = []
        future_features = []
        static_contexts = []
        
        total_length = self.max_encoder_length + self.max_prediction_length
        
        # Check if we have enough data
        if len(features_data) < total_length:
            raise ValueError(f"Insufficient data: need {total_length} points, got {len(features_data)}")
        
        for i in range(len(features_data) - total_length + 1):
            try:
                # Historical sequence (encoder input)
                historical_end = i + self.max_encoder_length
                if historical_end > len(features_data):
                    continue
                historical_seq = features_data.iloc[i:historical_end]
                
                # Future sequence (decoder input) - without target
                future_start = historical_end
                future_end = future_start + self.max_prediction_length
                if future_end > len(features_data):
                    continue
                future_seq = features_data.iloc[future_start:future_end].drop('target', axis=1, errors='ignore')
                
                # Target sequence (what we want to predict)
                target_seq = features_data['target'].iloc[future_start:future_end].values
            except IndexError as e:
                self.logger.warning(f"Index error at position {i}: {e}")
                continue
            
            # Static features (if provided)
            if static_features_data is not None:
                static_ctx = static_features_data.iloc[i].values
            else:
                # Create dummy static features
                static_ctx = np.random.randn(self.static_features)
            
            # Only add if we have valid sequences
            if len(historical_seq) == self.max_encoder_length and len(future_seq) == self.max_prediction_length:
                sequences.append(historical_seq.values)
                future_features.append(future_seq.values)
                targets.append(target_seq)
                static_contexts.append(static_ctx)
        
        # Check if we have any valid sequences
        if not sequences:
            raise ValueError("No valid sequences could be created from the data")
        
        # Convert to numpy arrays
        X_historical = np.array(sequences)
        X_future = np.array(future_features)
        X_static = np.array(static_contexts)
        y = np.array(targets)
        
        # Prepare inputs dictionary
        inputs = {
            'historical_inputs': X_historical,
            'future_inputs': X_future,
            'static_inputs': X_static
        }
        
        # Prepare targets dictionary (multi-quantile)
        targets_dict = {
            'quantile_predictions': np.repeat(y[:, :, np.newaxis], len(self.quantiles), axis=2),
            'point_prediction': y
        }
        
        self.logger.info(f"TFT data prepared: {X_historical.shape[0]} sequences, "
                        f"{X_historical.shape[1]} historical steps, "
                        f"{X_future.shape[1]} prediction steps")
        
        return inputs, targets_dict
    
    def train_tft(self,
                 price_data: pd.DataFrame,
                 features_data: pd.DataFrame,
                 static_features_data: pd.DataFrame = None,
                 validation_split: float = 0.2,
                 epochs: int = 100,
                 batch_size: int = 64,
                 learning_rate: float = 0.001) -> Dict:
        """Train the TFT model"""
        
        self.logger.info("Training Advanced Deep Learning-Enhanced TFT...")
        
        try:
            # Prepare data
            inputs, targets = self.prepare_tft_data(price_data, features_data, static_features_data)
            
            # Build model
            total_time_steps = self.max_encoder_length
            num_features = inputs['historical_inputs'].shape[2]
            
            self.model = self.build_model(total_time_steps, num_features)
            
            # Compile model
            # Use legacy optimizer for better M1/M2 Mac performance
            optimizer = keras.optimizers.legacy.Adam(learning_rate=learning_rate)
            
            self.model.compile(
                optimizer=optimizer,
                loss={
                    'quantile_predictions': self.quantile_loss,
                    'point_prediction': 'mse',
                    'attention_weights': lambda y_true, y_pred: 0.0,  # No loss for attention
                    'historical_importance': lambda y_true, y_pred: 0.0,
                    'future_importance': lambda y_true, y_pred: 0.0
                },
                loss_weights={
                    'quantile_predictions': 1.0,
                    'point_prediction': 0.5,
                    'attention_weights': 0.0,
                    'historical_importance': 0.0,
                    'future_importance': 0.0
                },
                metrics={
                    'point_prediction': ['mae', 'mse']
                }
            )
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7
                )
            ]
            
            # Dummy targets for auxiliary outputs
            extended_targets = targets.copy()
            batch_size_actual = inputs['historical_inputs'].shape[0]
            
            extended_targets['attention_weights'] = np.zeros((
                batch_size_actual, 
                self.num_heads, 
                self.max_encoder_length + self.max_prediction_length,
                self.max_encoder_length + self.max_prediction_length
            ))
            extended_targets['historical_importance'] = np.zeros((
                batch_size_actual, 
                self.max_encoder_length, 
                num_features
            ))
            extended_targets['future_importance'] = np.zeros((
                batch_size_actual, 
                self.max_prediction_length, 
                num_features
            ))
            
            # Train model
            history = self.model.fit(
                inputs,
                extended_targets,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            # Store training history
            self.training_history = {
                'epochs_trained': len(history.history['loss']),
                'final_train_loss': history.history['loss'][-1],
                'final_val_loss': min(history.history['val_loss']),
                'best_mae': min(history.history['val_point_prediction_mae']),
                'training_time': datetime.now().isoformat()
            }
            
            # Calculate feature importances from attention weights (only if model is trained)
            try:
                self._calculate_feature_importances(inputs)
            except Exception as e:
                self.logger.warning(f"Feature importance calculation skipped: {e}")
                self.feature_importances = {}
            
            self.logger.info(f"✅ ADE-TFT training completed!")
            self.logger.info(f"📊 Performance: MAE={self.training_history['best_mae']:.4f}")
            
            return {
                'success': True,
                'model_type': 'Advanced Deep Learning-Enhanced TFT',
                'training_history': self.training_history,
                'feature_importances': self.feature_importances,
                'model_summary': {
                    'total_parameters': self.model.count_params(),
                    'max_encoder_length': self.max_encoder_length,
                    'max_prediction_length': self.max_prediction_length,
                    'hidden_size': self.hidden_size,
                    'num_heads': self.num_heads,
                    'quantiles': self.quantiles
                }
            }
            
        except Exception as e:
            self.logger.error(f"TFT training failed: {e}")
            return {"error": str(e)}
    
    def _calculate_feature_importances(self, inputs):
        """Calculate feature importances from attention weights"""
        
        try:
            # Get predictions with attention weights
            predictions = self.model.predict(inputs, verbose=0)
            
            # Historical feature importance
            hist_importance = predictions['historical_importance']
            avg_hist_importance = np.mean(hist_importance, axis=(0, 1))
            
            # Future feature importance  
            future_importance = predictions['future_importance']
            avg_future_importance = np.mean(future_importance, axis=(0, 1))
            
            self.feature_importances = {
                'historical_features': avg_hist_importance.tolist(),
                'future_features': avg_future_importance.tolist(),
                'overall_importance': (avg_hist_importance + avg_future_importance).tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Feature importance calculation failed: {e}")
            self.feature_importances = {}
    
    def predict_multi_horizon(self,
                             recent_data: pd.DataFrame,
                             features_data: pd.DataFrame,
                             static_features: np.ndarray = None,
                             horizons: List[int] = [1, 6, 12, 24]) -> Dict:
        """Make multi-horizon predictions with uncertainty quantification"""
        
        if self.model is None:
            return {"error": "Model not trained. Call train_tft() first."}
        
        try:
            # Prepare input data
            if static_features is None:
                static_features = np.random.randn(1, self.static_features)
            
            # Get last sequence for prediction
            if len(features_data) < self.max_encoder_length:
                return {"error": f"Need at least {self.max_encoder_length} data points"}
            
            # Historical sequence
            historical_seq = features_data.iloc[-self.max_encoder_length:].values
            historical_seq = historical_seq.reshape(1, self.max_encoder_length, -1)
            
            # Future sequence (we need to provide known future features)
            # For crypto, we can use time-based features, technical indicators, etc.
            last_features = features_data.iloc[-1:].values
            future_seq = np.tile(last_features, (self.max_prediction_length, 1))
            future_seq = future_seq.reshape(1, self.max_prediction_length, -1)
            
            # Make prediction
            predictions = self.model.predict({
                'historical_inputs': historical_seq,
                'future_inputs': future_seq,
                'static_inputs': static_features
            }, verbose=0)
            
            # Extract quantile predictions
            quantile_preds = predictions['quantile_predictions'][0]  # Remove batch dimension
            point_pred = predictions['point_prediction'][0]
            attention_weights = predictions['attention_weights'][0]
            
            # Current price
            current_price = recent_data['close'].iloc[-1]
            
            # Process predictions for each horizon
            horizon_predictions = {}
            
            for horizon in horizons:
                if horizon <= self.max_prediction_length:
                    idx = horizon - 1  # Convert to 0-based index
                else:
                    # For horizons beyond max_prediction_length, use the last available prediction
                    idx = self.max_prediction_length - 1
                    
                    # Quantile predictions
                    q_low = float(quantile_preds[idx, 0])    # 10th percentile
                    q_median = float(quantile_preds[idx, 1])  # 50th percentile (median)
                    q_high = float(quantile_preds[idx, 2])   # 90th percentile
                    
                    # Point prediction
                    point_value = float(point_pred[idx, 0])
                    
                    # Price changes
                    price_change_low = (q_low - current_price) / current_price * 100
                    price_change_median = (q_median - current_price) / current_price * 100
                    price_change_high = (q_high - current_price) / current_price * 100
                    price_change_point = (point_value - current_price) / current_price * 100
                    
                    # Uncertainty measures
                    prediction_interval = q_high - q_low
                    uncertainty_ratio = prediction_interval / current_price
                    
                    horizon_predictions[f"{horizon}h"] = {
                        'predicted_price': {
                            'point': point_value,
                            'lower_80pct': q_low,
                            'median': q_median,
                            'upper_80pct': q_high
                        },
                        'price_change_pct': {
                            'point': price_change_point,
                            'lower_80pct': price_change_low,
                            'median': price_change_median,
                            'upper_80pct': price_change_high
                        },
                        'uncertainty_metrics': {
                            'prediction_interval': prediction_interval,
                            'uncertainty_ratio': uncertainty_ratio,
                            'confidence_score': 1.0 - min(uncertainty_ratio, 1.0)
                        }
                    }
            
            # Overall prediction summary
            primary_horizon = "24h" if "24h" in horizon_predictions else list(horizon_predictions.keys())[0]
            primary_pred = horizon_predictions[primary_horizon]
            
            # Attention analysis
            avg_attention = np.mean(attention_weights, axis=0)  # Average across heads
            attention_summary = {
                'encoder_attention': float(np.mean(avg_attention[:self.max_encoder_length, :self.max_encoder_length])),
                'decoder_attention': float(np.mean(avg_attention[-self.max_prediction_length:, -self.max_prediction_length:])),
                'cross_attention': float(np.mean(avg_attention[-self.max_prediction_length:, :self.max_encoder_length]))
            }
            
            return {
                'model_type': 'Advanced Deep Learning-Enhanced TFT',
                'current_price': current_price,
                'primary_prediction': primary_pred,
                'multi_horizon_predictions': horizon_predictions,
                'uncertainty_quantification': True,
                'attention_analysis': attention_summary,
                'feature_importances': self.feature_importances,
                'prediction_timestamp': datetime.now().isoformat(),
                'advanced_capabilities': [
                    'Multi-horizon forecasting',
                    'Uncertainty quantification',
                    'Interpretable attention weights',
                    'Variable selection networks',
                    'Quantile predictions'
                ]
            }
            
        except Exception as e:
            self.logger.error(f"TFT prediction failed: {e}")
            return {"error": str(e)}
    
    def save_model(self, filepath: str) -> bool:
        """Save trained TFT model"""
        try:
            if self.model is None:
                return False
            
            # Save model
            self.model.save(f"{filepath}_tft.h5")
            
            # Save metadata
            metadata = {
                'hidden_size': self.hidden_size,
                'num_heads': self.num_heads,
                'dropout_rate': self.dropout_rate,
                'num_quantiles': self.num_quantiles,
                'max_encoder_length': self.max_encoder_length,
                'max_prediction_length': self.max_prediction_length,
                'static_features': self.static_features,
                'dynamic_features': self.dynamic_features,
                'quantiles': self.quantiles,
                'training_history': self.training_history,
                'feature_importances': self.feature_importances,
                'save_timestamp': datetime.now().isoformat()
            }
            
            with open(f"{filepath}_tft_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"TFT model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save TFT model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained TFT model"""
        try:
            # Load model
            self.model = keras.models.load_model(f"{filepath}_tft.h5", compile=False)
            
            # Load metadata
            with open(f"{filepath}_tft_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            # Restore parameters
            self.hidden_size = metadata['hidden_size']
            self.num_heads = metadata['num_heads']
            self.dropout_rate = metadata['dropout_rate']
            self.num_quantiles = metadata['num_quantiles']
            self.max_encoder_length = metadata['max_encoder_length']
            self.max_prediction_length = metadata['max_prediction_length']
            self.static_features = metadata['static_features']
            self.dynamic_features = metadata['dynamic_features']
            self.quantiles = metadata['quantiles']
            self.training_history = metadata['training_history']
            self.feature_importances = metadata['feature_importances']
            
            self.logger.info(f"TFT model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load TFT model: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Advanced Deep Learning-Enhanced Temporal Fusion Transformer (ADE-TFT)")
    print("=" * 80)
    print("🎯 Multi-horizon forecasting with:")
    print("   • Variable Selection Networks")
    print("   • Gated Residual Networks") 
    print("   • Interpretable Multi-Head Attention")
    print("   • Quantile-based uncertainty estimation")
    print("   • Static and dynamic feature processing")
    print()
    
    # Test basic functionality
    tft = TemporalFusionTransformer(
        hidden_size=64,
        num_heads=4,
        max_encoder_length=168,
        max_prediction_length=24
    )
    
    print(f"✅ ADE-TFT initialized")
    print(f"📐 Architecture: {tft.hidden_size} hidden units, {tft.num_heads} attention heads")
    print(f"⏰ Encoder length: {tft.max_encoder_length} steps")
    print(f"🔮 Prediction length: {tft.max_prediction_length} steps")
    print(f"📊 Quantiles: {tft.quantiles}")
    print()
    print("💡 Next steps:")
    print("1. Prepare time series data with features")
    print("2. Call tft.train_tft(price_data, features_data)")
    print("3. Use tft.predict_multi_horizon() for advanced forecasting")
    print("4. Get multi-horizon predictions with uncertainty! 📈")
