#!/usr/bin/env python3
"""
Helformer Model Implementation - PyTorch Version
Revolutionary 2025 breakthrough model integrating Holt-Winters exponential smoothing 
with Transformer architecture for BNB swing trading

Enhanced with PyTorch 2.0+ and multi-timeframe training data
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.optimize import minimize
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

from logger import get_logger
from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime, timedelta
import os

# Check PyTorch availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
else:
    device = torch.device('cpu')
    print(f"💻 Using CPU for training")


class BNBMultiTimeframeDataset(Dataset):
    """
    Custom dataset for BNB multi-timeframe data (daily, weekly, monthly)
    """
    
    def __init__(self, data_dict: Dict[str, pd.DataFrame], sequence_length: int = 128):
        self.sequence_length = sequence_length
        self.data_dict = data_dict
        self.samples = []
        
        # Prepare samples from all timeframes
        self._prepare_samples()
    
    def _prepare_samples(self):
        """Prepare training samples from all timeframes"""
        for timeframe, data in self.data_dict.items():
            if len(data) >= self.sequence_length:
                # Create sequences for this timeframe
                for i in range(len(data) - self.sequence_length):
                    sequence = data.iloc[i:i + self.sequence_length]
                    target = data.iloc[i + self.sequence_length]['close']
                    
                    self.samples.append({
                        'timeframe': timeframe,
                        'sequence': sequence,
                        'target': target,
                        'start_idx': i,
                        'end_idx': i + self.sequence_length
                    })
        
        print(f"📊 Prepared {len(self.samples)} training samples from {len(self.data_dict)} timeframes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract features (OHLCV)
        sequence_data = sample['sequence'][['open', 'high', 'low', 'close', 'volume']].values
        
        # Normalize data
        sequence_normalized = self._normalize_sequence(sequence_data)
        
        # Convert to PyTorch tensors
        sequence_tensor = torch.FloatTensor(sequence_normalized).to(device)
        target_tensor = torch.FloatTensor([sample['target']]).to(device)
        
        return sequence_tensor, target_tensor, sample['timeframe']
    
    def _normalize_sequence(self, sequence_data):
        """Normalize sequence data using min-max scaling"""
        # Calculate min and max for each feature
        min_vals = np.min(sequence_data, axis=0)
        max_vals = np.max(sequence_data, axis=0)
        
        # Avoid division by zero
        max_vals = np.where(max_vals == min_vals, 1, max_vals - min_vals)
        
        # Normalize
        normalized = (sequence_data - min_vals) / max_vals
        
        # Handle NaN values
        normalized = np.nan_to_num(normalized, nan=0.0)
        
        return normalized


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer architecture
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head attention and feedforward network
    """
    
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dff, d_model)
        )
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        # Feedforward network
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        
        return x


class HelformerPyTorch(nn.Module):
    """
    PyTorch implementation of Helformer model
    """
    
    def __init__(self,
                 sequence_length: int = 128,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dff: int = 1024,
                 dropout_rate: float = 0.1,
                 hw_seasonal_periods: int = 1,
                 forecast_horizon: int = 720):
        
        super(HelformerPyTorch, self).__init__()
        
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.hw_seasonal_periods = hw_seasonal_periods
        self.forecast_horizon = forecast_horizon
        
        # Input projection (OHLCV -> d_model)
        self.input_projection = nn.Linear(5, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, sequence_length)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Holt-Winters components
        self.hw_components = {}
        self.hw_model = None
        
        # Performance tracking
        self.training_history = {}
        self.performance_metrics = {}
        
        # Move to device
        self.to(device)
    
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, 5) - OHLCV
            
        Returns:
            Predicted price tensor of shape (batch_size, 1)
        """
        batch_size, seq_len, features = x.shape
        
        # Validate input shape
        assert features == 5, f"Expected 5 features (OHLCV), got {features}"
        assert seq_len == self.sequence_length, f"Expected sequence length {self.sequence_length}, got {seq_len}"
        
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch_size, d_model)
        
        # Output projection
        x = self.dropout(x)
        x = self.output_projection(x)  # (batch_size, 1)
        
        return x
    
    def extract_holt_winters_components(self, data: pd.DataFrame) -> Dict:
        """
        Extract Holt-Winters components using statsmodels
        
        Args:
            data: DataFrame with 'close' column
            
        Returns:
            Dictionary with level, trend, and seasonal components
        """
        if not STATS_AVAILABLE:
            self.logger.warning("Statsmodels not available, using simplified components")
            return self._calculate_simple_components(data)
        
        try:
            # Validate data
            if len(data) < 12:  # Minimum required for seasonal analysis
                self.logger.warning("Insufficient data for seasonal analysis, using trend-only model")
                return self._calculate_trend_only_components(data)
            
            # Prepare data
            close_prices = data['close'].values
            close_prices = np.nan_to_num(close_prices, nan=0.0)  # Handle NaN values
            
            # Fit Holt-Winters model
            if self.hw_seasonal_periods > 1:
                # Seasonal model
                hw_model = ExponentialSmoothing(
                    close_prices,
                    seasonal_periods=self.hw_seasonal_periods,
                    trend='add',
                    seasonal='add'
                ).fit()
            else:
                # Trend-only model
                hw_model = ExponentialSmoothing(
                    close_prices,
                    trend='add'
                ).fit()
            
            # Extract components
            self.hw_model = hw_model
            self.hw_components = {
                'level': hw_model.level,
                'trend': hw_model.trend,
                'seasonal': hw_model.seasonal if hasattr(hw_model, 'seasonal') else None,
                'residual': hw_model.resid
            }
            
            self.logger.info(f"✅ Holt-Winters components extracted successfully")
            return self.hw_components
            
        except Exception as e:
            self.logger.error(f"Error extracting Holt-Winters components: {e}")
            return self._calculate_simple_components(data)
    
    def _calculate_simple_components(self, data: pd.DataFrame) -> Dict:
        """Calculate simplified components when statsmodels is not available"""
        close_prices = data['close'].values
        close_prices = np.nan_to_num(close_prices, nan=0.0)
        
        # Simple moving averages
        level = np.mean(close_prices)
        trend = np.polyfit(np.arange(len(close_prices)), close_prices, 1)[0]
        
        return {
            'level': level,
            'trend': trend,
            'seasonal': None,
            'residual': close_prices - level
        }
    
    def _calculate_trend_only_components(self, data: pd.DataFrame) -> Dict:
        """Calculate trend-only components for insufficient data"""
        close_prices = data['close'].values
        close_prices = np.nan_to_num(close_prices, nan=0.0)
        
        # Linear trend
        x = np.arange(len(close_prices))
        trend_coeff = np.polyfit(x, close_prices, 1)
        trend = trend_coeff[0]
        level = trend_coeff[1]
        
        return {
            'level': level,
            'trend': trend,
            'seasonal': None,
            'residual': close_prices - (trend * x + level)
        }
    
    def train_model(self, 
                   train_loader: DataLoader,
                   val_loader: Optional[DataLoader] = None,
                   epochs: int = 100,
                   learning_rate: float = 0.001,
                   patience: int = 20) -> Dict:
        """
        Train the Helformer model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        self.train()
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
        
        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"🚀 Starting Helformer training with PyTorch on {device}")
        print(f"📊 Training samples: {len(train_loader.dataset)}")
        print(f"🔧 Model parameters: {sum(p.numel() for p in self.parameters()):,}")
        
        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            for batch_idx, (sequences, targets, timeframes) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self(sequences)
                loss = criterion(predictions, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Progress update
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_loader:
                val_loss = self._validate_model(val_loader, criterion)
                val_losses.append(val_loss)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.state_dict(), 'best_helformer_pytorch.pth')
                else:
                    patience_counter += 1
                
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                if patience_counter >= patience:
                    print(f"🛑 Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.6f}")
        
        # Store training history
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1
        }
        
        print(f"✅ Training completed! Best validation loss: {best_val_loss:.6f}")
        return self.training_history
    
    def _validate_model(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """Validate the model"""
        self.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for sequences, targets, timeframes in val_loader:
                predictions = self(sequences)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def predict(self, data: pd.DataFrame, periods_ahead: int = 1) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            data: Input data with OHLCV columns
            periods_ahead: Number of periods to predict ahead
            
        Returns:
            Array of predictions
        """
        self.eval()
        
        # Prepare input data
        if len(data) < self.sequence_length:
            raise ValueError(f"Data length {len(data) < self.sequence_length} must be >= sequence_length {self.sequence_length}")
        
        # Get the last sequence
        last_sequence = data.tail(self.sequence_length)[['open', 'high', 'low', 'close', 'volume']].values
        
        # Normalize
        min_vals = np.min(last_sequence, axis=0)
        max_vals = np.max(last_sequence, axis=0)
        max_vals = np.where(max_vals == min_vals, 1, max_vals - min_vals)
        normalized_sequence = (last_sequence - min_vals) / max_vals
        normalized_sequence = np.nan_to_num(normalized_sequence, nan=0.0)
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(normalized_sequence).unsqueeze(0).to(device)
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(periods_ahead):
                # Make prediction
                pred = self(sequence_tensor)
                predictions.append(pred.item())
                
                # Update sequence for next prediction (simple approach)
                # In production, you might want to use the predicted value
                new_row = last_sequence[-1].copy()
                new_row[3] = pred.item()  # Update close price
                last_sequence = np.vstack([last_sequence[1:], new_row])
                
                # Normalize updated sequence
                normalized_sequence = (last_sequence - min_vals) / max_vals
                normalized_sequence = np.nan_to_num(normalized_sequence, nan=0.0)
                sequence_tensor = torch.FloatTensor(normalized_sequence).unsqueeze(0).to(device)
        
        return np.array(predictions)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_state = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'sequence_length': self.sequence_length,
                'd_model': self.d_model,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'dff': self.dff,
                'dropout_rate': self.dropout_rate,
                'hw_seasonal_periods': self.hw_seasonal_periods,
                'forecast_horizon': self.forecast_horizon
            },
            'hw_components': self.hw_components,
            'training_history': self.training_history,
            'performance_metrics': self.performance_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(model_state, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_state = torch.load(filepath, map_location=device)
        
        # Load model weights
        self.load_state_dict(model_state['model_state_dict'])
        
        # Load other components
        self.hw_components = model_state.get('hw_components', {})
        self.training_history = model_state.get('training_history', {})
        self.performance_metrics = model_state.get('performance_metrics', {})
        
        print(f"✅ Model loaded from {filepath}")
        print(f"📊 Training history: {len(self.training_history.get('train_losses', []))} epochs")


def main():
    """Main function to demonstrate Helformer training"""
    print("🎯 Helformer PyTorch - Multi-timeframe Training Demo")
    print("="*70)
    
    # Check if data files exist
    data_files = {
        'daily': 'generated_data/bnb_daily_2023_2025.csv',
        'weekly': 'generated_data/bnb_weekly_2023_2025.csv',
        'monthly': 'generated_data/bnb_monthly_2023_2025.csv'
    }
    
    missing_files = [tf for tf, path in data_files.items() if not os.path.exists(path)]
    if missing_files:
        print(f"❌ Missing data files: {missing_files}")
        print("   Please run data_generator.py first")
        return
    
    try:
        # Load data
        print("📊 Loading multi-timeframe data...")
        data_dict = {}
        for timeframe, filepath in data_files.items():
            data = pd.read_csv(filepath)
            data['date'] = pd.to_datetime(data['date'])
            data_dict[timeframe] = data
            print(f"   {timeframe.capitalize()}: {len(data)} records")
        
        # Create dataset
        print("\n🔧 Creating multi-timeframe dataset...")
        dataset = BNBMultiTimeframeDataset(data_dict, sequence_length=64)  # Reduced for demo
        
        # Split into train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        
        # Initialize model
        print("\n🚀 Initializing Helformer PyTorch model...")
        model = HelformerPyTorch(
            sequence_length=64,
            d_model=128,  # Reduced for demo
            num_heads=4,
            num_layers=3,
            dff=512,
            hw_seasonal_periods=3,  # Quarterly seasonality
            forecast_horizon=2880
        )
        
        # Train model
        print("\n🎯 Starting training...")
        training_history = model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=50,  # Reduced for demo
            learning_rate=0.001,
            patience=10
        )
        
        # Save model
        print("\n💾 Saving trained model...")
        model.save_model('helformer_pytorch_trained.pth')
        
        # Test prediction
        print("\n🔮 Testing prediction...")
        test_data = data_dict['daily'].tail(100)
        predictions = model.predict(test_data, periods_ahead=5)
        
        print(f"   Next 5 predictions: {predictions}")
        print(f"   Last actual close: ${test_data['close'].iloc[-1]:.2f}")
        
        print("\n🎉 Helformer PyTorch training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
