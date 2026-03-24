"""
Suspicious Activity Anomaly Detection System
Uses lightweight ML models to train on suspicious activity data
and detect danger zones and safe zones based on timeline analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Lightweight ML libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class RiskLevel(Enum):
    """Risk classification levels"""
    SAFE = "SAFE"
    LOW_RISK = "LOW_RISK"
    MEDIUM_RISK = "MEDIUM_RISK"
    HIGH_RISK = "HIGH_RISK"
    CRITICAL = "CRITICAL"


@dataclass
class ActivityData:
    """Data structure for suspicious activity"""
    timestamp: datetime
    latitude: float
    longitude: float
    ip_address: str
    user_id: str
    action_type: str  # login, transaction, access, etc.
    severity_score: float  # 0-100
    location_name: str = ""
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'latitude': self.latitude,
            'longitude': self.longitude,
            'ip_address': self.ip_address,
            'user_id': self.user_id,
            'action_type': self.action_type,
            'severity_score': self.severity_score,
            'location_name': self.location_name
        }


@dataclass
class ZoneAnalysis:
    """Result of zone analysis"""
    zone_id: str
    risk_level: RiskLevel
    latitude: float
    longitude: float
    anomaly_score: float
    time_window: Tuple[datetime, datetime]
    activity_count: int
    average_severity: float
    recommended_action: str
    
    def to_dict(self):
        return {
            'zone_id': self.zone_id,
            'risk_level': self.risk_level.value,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'anomaly_score': self.anomaly_score,
            'time_window_start': self.time_window[0].isoformat(),
            'time_window_end': self.time_window[1].isoformat(),
            'activity_count': self.activity_count,
            'average_severity': self.average_severity,
            'recommended_action': self.recommended_action
        }


class SuspiciousActivityDetector:
    """
    Lightweight AI/ML detector for suspicious activities
    Identifies danger and safe zones using anomaly detection
    """
    
    def __init__(self, n_clusters: int = 5, contamination: float = 0.1):
        """
        Initialize the detector
        
        Args:
            n_clusters: Number of geographic clusters (zones)
            contamination: Expected proportion of anomalies (0-1)
        """
        self.n_clusters = n_clusters
        self.contamination = contamination
        
        # ML Models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        
        # Storage
        self.activities: List[ActivityData] = []
        self.is_trained = False
        self.feature_columns = ['latitude', 'longitude', 'severity_score', 
                               'hour_of_day', 'day_of_week']
        
    def add_activity(self, activity: ActivityData) -> None:
        """Add a suspicious activity to the training data"""
        self.activities.append(activity)
        
    def add_activities_batch(self, activities: List[ActivityData]) -> None:
        """Add multiple suspicious activities"""
        self.activities.extend(activities)
        
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal and spatial features"""
        df_features = df.copy()
        
        # Temporal features
        df_features['hour_of_day'] = df['timestamp'].dt.hour
        df_features['day_of_week'] = df['timestamp'].dt.dayofweek
        df_features['minute_of_day'] = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
        
        # Spatial features (keep as is)
        # Normalize severity
        df_features['severity_normalized'] = df['severity_score'] / 100.0
        
        return df_features
    
    def train(self) -> Dict:
        """
        Train the anomaly detection models
        
        Returns:
            Training statistics
        """
        if len(self.activities) < 3:
            raise ValueError("Need at least 3 activities to train")
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(a) for a in self.activities])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Feature engineering
        df_features = self._engineer_features(df)
        
        # Prepare features for training
        X = df_features[self.feature_columns].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train isolation forest (anomaly detection)
        anomaly_labels = self.isolation_forest.fit_predict(X_scaled)
        anomaly_scores = -self.isolation_forest.score_samples(X_scaled)
        
        # Train K-means for geographic clustering
        self.kmeans.fit(X_scaled[:, :2])  # Use only lat/lon for clustering
        cluster_labels = self.kmeans.labels_
        
        # Store results in dataframe
        df['anomaly_score'] = (anomaly_scores - anomaly_scores.min()) / \
                              (anomaly_scores.max() - anomaly_scores.min())
        df['is_anomaly'] = anomaly_labels == -1
        df['cluster'] = cluster_labels
        
        self.is_trained = True
        self.training_df = df
        
        stats = {
            'total_activities': len(df),
            'anomalies_detected': int(df['is_anomaly'].sum()),
            'anomaly_percentage': float(df['is_anomaly'].mean() * 100),
            'clusters_created': self.n_clusters,
            'training_timestamp': datetime.now().isoformat()
        }
        
        return stats
    
    def detect_zones(self, time_window_hours: int = 24) -> List[ZoneAnalysis]:
        """
        Detect danger and safe zones based on temporal and spatial patterns
        
        Args:
            time_window_hours: Size of time window for zone analysis
            
        Returns:
            List of zone analyses
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before zone detection")
        
        df = self.training_df.copy()
        zones = []
        
        # Analyze each cluster
        for cluster_id in range(self.n_clusters):
            cluster_data = df[df['cluster'] == cluster_id]
            
            if len(cluster_data) == 0:
                continue
            
            # Create time windows
            min_time = cluster_data['timestamp'].min()
            max_time = cluster_data['timestamp'].max()
            current_time = min_time
            
            window_number = 0
            while current_time < max_time:
                window_end = current_time + timedelta(hours=time_window_hours)
                
                # Filter data in this window
                window_data = cluster_data[
                    (cluster_data['timestamp'] >= current_time) &
                    (cluster_data['timestamp'] < window_end)
                ]
                
                if len(window_data) > 0:
                    # Calculate zone metrics
                    anomaly_score = window_data['anomaly_score'].mean()
                    severity = window_data['severity_score'].mean()
                    activity_count = len(window_data)
                    
                    # Determine risk level
                    risk_level = self._calculate_risk_level(
                        anomaly_score, 
                        severity, 
                        activity_count
                    )
                    
                    # Get recommendation
                    recommendation = self._get_recommendation(risk_level)
                    
                    # Create zone analysis
                    zone = ZoneAnalysis(
                        zone_id=f"ZONE_{cluster_id}_{window_number}",
                        risk_level=risk_level,
                        latitude=window_data['latitude'].mean(),
                        longitude=window_data['longitude'].mean(),
                        anomaly_score=anomaly_score,
                        time_window=(current_time, window_end),
                        activity_count=activity_count,
                        average_severity=severity,
                        recommended_action=recommendation
                    )
                    
                    zones.append(zone)
                
                current_time = window_end
                window_number += 1
        
        return zones
    
    def _calculate_risk_level(self, anomaly_score: float, 
                             severity: float, activity_count: int) -> RiskLevel:
        """Calculate overall risk level based on multiple factors"""
        
        # Normalize metrics
        anomaly_factor = anomaly_score * 0.4
        severity_factor = (severity / 100.0) * 0.4
        activity_factor = min(activity_count / 10.0, 1.0) * 0.2
        
        combined_score = anomaly_factor + severity_factor + activity_factor
        
        # Map to risk levels
        if combined_score < 0.2:
            return RiskLevel.SAFE
        elif combined_score < 0.4:
            return RiskLevel.LOW_RISK
        elif combined_score < 0.6:
            return RiskLevel.MEDIUM_RISK
        elif combined_score < 0.8:
            return RiskLevel.HIGH_RISK
        else:
            return RiskLevel.CRITICAL
    
    def _get_recommendation(self, risk_level: RiskLevel) -> str:
        """Generate action recommendation based on risk level"""
        recommendations = {
            RiskLevel.SAFE: "No action needed - zone is safe",
            RiskLevel.LOW_RISK: "Monitor this zone - low risk detected",
            RiskLevel.MEDIUM_RISK: "Increase monitoring - medium risk zone",
            RiskLevel.HIGH_RISK: "Alert security team - high risk zone detected",
            RiskLevel.CRITICAL: "CRITICAL ALERT - Immediate action required!"
        }
        return recommendations.get(risk_level, "Unknown")
    
    def predict_activity_risk(self, activity: ActivityData) -> Dict:
        """
        Predict risk level for a new activity
        
        Args:
            activity: New activity to evaluate
            
        Returns:
            Risk prediction with score
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(activity)])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Feature engineering
        df_features = self._engineer_features(df)
        
        # Prepare features
        X = df_features[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly score
        anomaly_score = -self.isolation_forest.score_samples(X_scaled)[0]
        anomaly_score = (anomaly_score - self.scaler.scale_[0]) / \
                       (self.scaler.scale_[0] * 2)  # Normalize
        anomaly_score = np.clip(anomaly_score, 0, 1)
        
        # Get assigned cluster
        cluster = self.kmeans.predict(X_scaled[:, :2])[0]
        
        # Calculate risk
        risk_level = self._calculate_risk_level(
            anomaly_score, 
            activity.severity_score, 
            1
        )
        
        return {
            'activity': activity.to_dict(),
            'risk_level': risk_level.value,
            'anomaly_score': float(anomaly_score),
            'assigned_cluster': int(cluster),
            'recommendation': self._get_recommendation(risk_level),
            'prediction_timestamp': datetime.now().isoformat()
        }
    
    def get_summary_report(self) -> Dict:
        """Generate summary report of detected zones"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        zones = self.detect_zones()
        
        # Categorize zones
        danger_zones = [z for z in zones if z.risk_level in 
                       [RiskLevel.HIGH_RISK, RiskLevel.CRITICAL]]
        safe_zones = [z for z in zones if z.risk_level in 
                     [RiskLevel.SAFE, RiskLevel.LOW_RISK]]
        warning_zones = [z for z in zones if z.risk_level == RiskLevel.MEDIUM_RISK]
        
        return {
            'summary': {
                'total_zones': len(zones),
                'danger_zones': len(danger_zones),
                'warning_zones': len(warning_zones),
                'safe_zones': len(safe_zones),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'danger_zones': [z.to_dict() for z in danger_zones],
            'warning_zones': [z.to_dict() for z in warning_zones],
            'safe_zones': [z.to_dict() for z in safe_zones],
            'all_zones': [z.to_dict() for z in zones]
        }
    
    def export_report(self, filepath: str) -> None:
        """Export summary report to JSON file"""
        report = self.get_summary_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report exported to {filepath}")


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = SuspiciousActivityDetector(n_clusters=5, contamination=0.15)
    
    # Generate sample suspicious activity data
    base_time = datetime.now() - timedelta(days=7)
    sample_activities = [
        ActivityData(
            timestamp=base_time + timedelta(hours=i),
            latitude=40.7128 + np.random.normal(0, 0.05),
            longitude=-74.0060 + np.random.normal(0, 0.05),
            ip_address=f"192.168.1.{np.random.randint(1, 255)}",
            user_id=f"USER_{np.random.randint(1, 100)}",
            action_type=np.random.choice(['login', 'transaction', 'access', 'download']),
            severity_score=np.random.uniform(10, 95) if i % 5 == 0 else np.random.uniform(5, 40),
            location_name=np.random.choice(['Downtown', 'Airport', 'Hotel', 'Shopping Mall'])
        )
        for i in range(100)
    ]
    
    # Add activities
    print("Adding suspicious activities...")
    detector.add_activities_batch(sample_activities)
    
    # Train model
    print("\nTraining anomaly detection model...")
    stats = detector.train()
    print(f"Training complete: {stats}")
    
    # Detect zones
    print("\nDetecting danger and safe zones...")
    zones = detector.detect_zones(time_window_hours=6)
    print(f"Detected {len(zones)} zones")
    
    # Get summary report
    print("\nGenerating summary report...")
    report = detector.get_summary_report()
    print(json.dumps(report['summary'], indent=2))
    
    # Predict risk for new activity
    print("\nPredicting risk for new activity...")
    new_activity = ActivityData(
        timestamp=datetime.now(),
        latitude=40.7580,
        longitude=-73.9855,
        ip_address="203.0.113.45",
        user_id="USER_999",
        action_type="transaction",
        severity_score=85.0
    )
    
    risk_prediction = detector.predict_activity_risk(new_activity)
    print(json.dumps(risk_prediction, indent=2))
