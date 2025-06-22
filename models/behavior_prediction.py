"""
LSTM-based Behavior Prediction Module for Autonomous Vehicle
Predicts trajectories and behaviors of detected objects
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import deque
import math

from config.settings import ModelConfig
from core.logger import logger


@dataclass
class TrajectoryPoint:
    """Point in a predicted trajectory"""
    x: float
    y: float
    timestamp: float
    confidence: float


@dataclass
class BehaviorPrediction:
    """Prediction result for an object"""
    object_id: str
    class_name: str
    current_position: Tuple[float, float]
    predicted_trajectory: List[TrajectoryPoint]
    predicted_speed: float
    predicted_direction: float
    confidence: float
    prediction_horizon: float


class LSTMPredictor(nn.Module):
    """LSTM neural network for trajectory prediction"""
    
    def __init__(self, input_size: int = 4, hidden_size: int = 64, num_layers: int = 2, output_size: int = 4):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, output_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM network"""
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class BehaviorPredictor:
    """LSTM-based behavior prediction system"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = None):
        self.device = device or self._select_device()
        self.model_path = model_path
        
        # LSTM model
        self.model: Optional[LSTMPredictor] = None
        self.is_loaded = False
        
        # Trajectory tracking
        self.trajectory_history: Dict[str, deque] = {}
        self.max_history_length = ModelConfig.LSTM_SEQUENCE_LENGTH
        self.prediction_horizon = ModelConfig.PREDICTION_HORIZON
        
        # Performance tracking
        self.prediction_times: List[float] = []
        self.prediction_count = 0
        
        # Model parameters
        self.input_size = 4  # x, y, speed, direction
        self.output_size = 4  # predicted x, y, speed, direction
        
        logger.info(f"Behavior predictor initialized with device: {self.device}")
    
    def _select_device(self) -> str:
        """Select the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self) -> bool:
        """Load or create LSTM model"""
        try:
            logger.start_timer("lstm_model_loading")
            
            # Create new model if no saved model exists
            if self.model_path and torch.load(self.model_path, map_location=self.device):
                self.model = torch.load(self.model_path, map_location=self.device)
                logger.info("Loaded pre-trained LSTM model")
            else:
                self.model = LSTMPredictor(
                    input_size=self.input_size,
                    hidden_size=64,
                    num_layers=2,
                    output_size=self.output_size
                ).to(self.device)
                logger.info("Created new LSTM model")
            
            self.model.eval()
            self.is_loaded = True
            
            loading_time = logger.end_timer("lstm_model_loading")
            logger.info(f"LSTM model loaded successfully in {loading_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            return False
    
    def update_trajectory(self, object_id: str, position: Tuple[float, float], 
                         speed: float, direction: float, timestamp: float):
        """Update trajectory history for an object"""
        if object_id not in self.trajectory_history:
            self.trajectory_history[object_id] = deque(maxlen=self.max_history_length)
        
        # Create trajectory point
        trajectory_point = {
            'x': position[0],
            'y': position[1],
            'speed': speed,
            'direction': direction,
            'timestamp': timestamp
        }
        
        self.trajectory_history[object_id].append(trajectory_point)
    
    def predict_behavior(self, object_id: str, class_name: str, 
                        current_position: Tuple[float, float]) -> Optional[BehaviorPrediction]:
        """Predict behavior for a specific object"""
        if not self.is_loaded or not self.model:
            logger.error("LSTM model not loaded")
            return None
        
        if object_id not in self.trajectory_history:
            logger.warning(f"No trajectory history for object {object_id}")
            return None
        
        history = self.trajectory_history[object_id]
        if len(history) < 3:  # Need at least 3 points for prediction
            logger.warning(f"Insufficient trajectory history for object {object_id}")
            return None
        
        try:
            logger.start_timer("behavior_prediction")
            
            # Prepare input sequence
            input_sequence = self._prepare_input_sequence(history)
            
            # Run prediction
            with torch.no_grad():
                prediction = self.model(input_sequence)
            
            # Parse prediction results
            predicted_trajectory = self._generate_trajectory(prediction, current_position)
            
            # Calculate confidence based on trajectory consistency
            confidence = self._calculate_confidence(history)
            
            # Create behavior prediction
            behavior_prediction = BehaviorPrediction(
                object_id=object_id,
                class_name=class_name,
                current_position=current_position,
                predicted_trajectory=predicted_trajectory,
                predicted_speed=float(prediction[0, 2]),
                predicted_direction=float(prediction[0, 3]),
                confidence=confidence,
                prediction_horizon=self.prediction_horizon
            )
            
            prediction_time = logger.end_timer("behavior_prediction")
            self.prediction_times.append(prediction_time)
            self.prediction_count += 1
            
            # Log prediction
            logger.log_behavior_prediction(behavior_prediction, prediction_time)
            
            return behavior_prediction
            
        except Exception as e:
            logger.error(f"Error during behavior prediction: {e}")
            return None
    
    def _prepare_input_sequence(self, history: deque) -> torch.Tensor:
        """Prepare input sequence for LSTM"""
        # Convert history to tensor
        sequence_data = []
        for point in history:
            sequence_data.append([
                point['x'],
                point['y'],
                point['speed'],
                point['direction']
            ])
        
        # Pad sequence if needed
        while len(sequence_data) < self.max_history_length:
            sequence_data.insert(0, sequence_data[0])  # Repeat first point
        
        # Convert to tensor
        sequence_tensor = torch.tensor(sequence_data, dtype=torch.float32)
        sequence_tensor = sequence_tensor.unsqueeze(0)  # Add batch dimension
        sequence_tensor = sequence_tensor.to(self.device)
        
        return sequence_tensor
    
    def _generate_trajectory(self, prediction: torch.Tensor, 
                           current_position: Tuple[float, float]) -> List[TrajectoryPoint]:
        """Generate trajectory points from prediction"""
        trajectory = []
        current_time = time.time()
        
        # Extract predicted values
        pred_x = float(prediction[0, 0])
        pred_y = float(prediction[0, 1])
        pred_speed = float(prediction[0, 2])
        pred_direction = float(prediction[0, 3])
        
        # Generate trajectory points
        for i in range(self.prediction_horizon):
            # Calculate position at each time step
            time_step = i + 1
            distance = pred_speed * time_step
            
            x = current_position[0] + distance * math.cos(pred_direction)
            y = current_position[1] + distance * math.sin(pred_direction)
            
            # Calculate confidence (decreases with time)
            confidence = max(0.1, 1.0 - (time_step / self.prediction_horizon))
            
            trajectory_point = TrajectoryPoint(
                x=x,
                y=y,
                timestamp=current_time + time_step,
                confidence=confidence
            )
            
            trajectory.append(trajectory_point)
        
        return trajectory
    
    def _calculate_confidence(self, history: deque) -> float:
        """Calculate prediction confidence based on trajectory consistency"""
        if len(history) < 3:
            return 0.5
        
        # Calculate speed consistency
        speeds = [point['speed'] for point in history]
        speed_variance = np.var(speeds)
        speed_confidence = max(0.1, 1.0 - speed_variance / 100.0)
        
        # Calculate direction consistency
        directions = [point['direction'] for point in history]
        direction_variance = np.var(directions)
        direction_confidence = max(0.1, 1.0 - direction_variance / (math.pi ** 2))
        
        # Overall confidence
        confidence = (speed_confidence + direction_confidence) / 2
        return min(1.0, max(0.1, confidence))
    
    def predict_all_behaviors(self, detected_objects: List[Dict]) -> List[BehaviorPrediction]:
        """Predict behaviors for all detected objects"""
        predictions = []
        
        for obj in detected_objects:
            object_id = obj.get('id', f"obj_{len(predictions)}")
            class_name = obj.get('class_name', 'unknown')
            position = obj.get('center', (0, 0))
            
            # Update trajectory
            speed = obj.get('speed', 0.0)
            direction = obj.get('direction', 0.0)
            timestamp = time.time()
            
            self.update_trajectory(object_id, position, speed, direction, timestamp)
            
            # Predict behavior
            prediction = self.predict_behavior(object_id, class_name, position)
            if prediction:
                predictions.append(prediction)
        
        return predictions
    
    def train_model(self, training_data: List[Dict]) -> bool:
        """Train the LSTM model with trajectory data"""
        if not self.model:
            logger.error("Model not initialized")
            return False
        
        try:
            logger.info("Starting LSTM model training...")
            
            # Prepare training data
            X, y = self._prepare_training_data(training_data)
            
            # Training parameters
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            # Training loop
            self.model.train()
            num_epochs = 50
            
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(X)
                loss = criterion(outputs, y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            self.model.eval()
            logger.info("LSTM model training completed")
            
            # Save model
            if self.model_path:
                torch.save(self.model, self.model_path)
                logger.info(f"Model saved to {self.model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return False
    
    def _prepare_training_data(self, training_data: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare training data for LSTM"""
        X_data = []
        y_data = []
        
        for trajectory in training_data:
            points = trajectory.get('points', [])
            
            if len(points) < self.max_history_length + 1:
                continue
            
            # Create input sequence
            input_sequence = []
            for i in range(self.max_history_length):
                point = points[i]
                input_sequence.append([
                    point['x'],
                    point['y'],
                    point['speed'],
                    point['direction']
                ])
            
            # Create target (next point)
            target_point = points[self.max_history_length]
            target = [
                target_point['x'],
                target_point['y'],
                target_point['speed'],
                target_point['direction']
            ]
            
            X_data.append(input_sequence)
            y_data.append(target)
        
        # Convert to tensors
        X = torch.tensor(X_data, dtype=torch.float32).to(self.device)
        y = torch.tensor(y_data, dtype=torch.float32).to(self.device)
        
        return X, y
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get prediction performance metrics"""
        if not self.prediction_times:
            return {}
        
        return {
            "avg_prediction_time": np.mean(self.prediction_times),
            "max_prediction_time": np.max(self.prediction_times),
            "total_predictions": self.prediction_count,
            "model_loaded": self.is_loaded
        }
    
    def clear_history(self, object_id: Optional[str] = None):
        """Clear trajectory history"""
        if object_id:
            if object_id in self.trajectory_history:
                del self.trajectory_history[object_id]
        else:
            self.trajectory_history.clear()
        
        logger.info(f"Cleared trajectory history for {object_id or 'all objects'}") 