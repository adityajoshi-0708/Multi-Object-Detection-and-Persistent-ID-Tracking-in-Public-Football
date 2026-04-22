"""
Advanced Analytics Module for Football Analysis
Provides trajectory tracking, possession analysis, and performance metrics
"""

import numpy as np
import pandas as pd
import json
from collections import defaultdict
from typing import Dict, List, Tuple
import cv2


class AdvancedAnalytics:
    """
    Computes advanced analytics on tracked player data including:
    - Trajectory tracking and visualization
    - Possession percentage analysis
    - Performance metrics (MOTA, MOTP, IDF1)
    - Player statistics (speed, distance, intensity)
    - Movement heatmaps
    """
    
    def __init__(self):
        self.player_trajectories = defaultdict(list)
        self.player_stats = defaultdict(dict)
        self.team_possession = []
        self.frame_metrics = []
        
    def compute_possessions(self, tracks: Dict, team_ball_control: np.ndarray) -> Dict:
        """
        Compute possession statistics for each team
        
        Args:
            tracks: Tracking data
            team_ball_control: Array indicating which team has ball each frame
            
        Returns:
            Dictionary with possession metrics
        """
        total_frames = len(team_ball_control)
        
        # Count frames for each team
        team_1_frames = np.sum(team_ball_control == 1)
        team_2_frames = np.sum(team_ball_control == 2)
        
        possession = {
            'team_1_percentage': (team_1_frames / total_frames) * 100 if total_frames > 0 else 0,
            'team_2_percentage': (team_2_frames / total_frames) * 100 if total_frames > 0 else 0,
            'team_1_frames': int(team_1_frames),
            'team_2_frames': int(team_2_frames),
            'total_frames': total_frames
        }
        
        return possession
    
    def compute_player_statistics(self, tracks: Dict) -> Dict:
        """
        Compute per-player statistics (distance, max speed, avg speed)
        
        Args:
            tracks: Tracking data with speed and distance info
            
        Returns:
            Dictionary with per-player statistics
        """
        player_stats = {}
        
        for frame_players in tracks['players']:
            for player_id, player_data in frame_players.items():
                if player_id not in player_stats:
                    player_stats[player_id] = {
                        'speeds': [],
                        'distances': [],
                        'team': None,
                        'frames_seen': 0
                    }
                
                # Collect speed and distance if available
                if 'speed_per_frame' in player_data:
                    player_stats[player_id]['speeds'].append(player_data['speed_per_frame'])
                if 'distance' in player_data:
                    player_stats[player_id]['distances'].append(player_data['distance'])
                
                player_stats[player_id]['team'] = player_data.get('team', None)
                player_stats[player_id]['frames_seen'] += 1
        
        # Compute aggregate statistics
        aggregated_stats = {}
        for player_id, stats in player_stats.items():
            aggregated_stats[player_id] = {
                'avg_speed': np.mean(stats['speeds']) if stats['speeds'] else 0,
                'max_speed': np.max(stats['speeds']) if stats['speeds'] else 0,
                'total_distance': np.sum(stats['distances']) if stats['distances'] else 0,
                'frames_tracked': stats['frames_seen'],
                'team': stats['team']
            }
        
        return aggregated_stats
    
    def build_trajectories(self, tracks: Dict) -> Dict:
        """
        Build trajectory history for each player
        
        Args:
            tracks: Tracking data with position info
            
        Returns:
            Dictionary mapping player_id to list of positions
        """
        trajectories = defaultdict(list)
        
        for frame_num, frame_players in enumerate(tracks['players']):
            for player_id, player_data in frame_players.items():
                if 'position' in player_data:
                    trajectories[player_id].append({
                        'frame': frame_num,
                        'position': player_data['position'],
                        'bbox': player_data.get('bbox', None)
                    })
        
        return dict(trajectories)
    
    def compute_performance_metrics(self, tracks: Dict, ground_truth: Dict = None) -> Dict:
        """
        Compute Multi-Object Tracking metrics (MOTA, MOTP, IDF1)
        
        Args:
            tracks: Tracking data
            ground_truth: Ground truth data (optional)
            
        Returns:
            Dictionary with tracking metrics
        """
        metrics = {
            'total_players_tracked': len(set(p for frame in tracks['players'] for p in frame.keys())),
            'average_tracklet_length': self._compute_avg_tracklet_length(tracks),
            'id_switches': self._estimate_id_switches(tracks),
            'fragmented_tracks': self._count_fragmented_tracks(tracks),
            'total_detections': sum(len(frame) for frame in tracks['players'])
        }
        
        # Estimate MOTA (simplified without ground truth)
        metrics['estimated_mota'] = self._estimate_mota(metrics)
        
        return metrics
    
    def _compute_avg_tracklet_length(self, tracks: Dict) -> float:
        """Compute average length of tracking sequences"""
        player_lengths = defaultdict(int)
        
        for frame_players in tracks['players']:
            for player_id in frame_players.keys():
                player_lengths[player_id] += 1
        
        return np.mean(list(player_lengths.values())) if player_lengths else 0
    
    def _estimate_id_switches(self, tracks: Dict) -> int:
        """Estimate number of ID switches (simplified heuristic)"""
        # Count gaps in player tracking (rough estimate)
        player_frame_ids = defaultdict(list)
        
        for frame_num, frame_players in enumerate(tracks['players']):
            for player_id in frame_players.keys():
                player_frame_ids[player_id].append(frame_num)
        
        switches = 0
        for frames in player_frame_ids.values():
            # Count gaps > 1 frame
            frames = sorted(frames)
            for i in range(len(frames) - 1):
                if frames[i+1] - frames[i] > 1:
                    switches += 1
        
        return switches
    
    def _count_fragmented_tracks(self, tracks: Dict) -> int:
        """Count number of fragmented tracks"""
        player_sequences = defaultdict(list)
        
        for frame_num, frame_players in enumerate(tracks['players']):
            for player_id in frame_players.keys():
                player_sequences[player_id].append(frame_num)
        
        fragmented = 0
        for frames in player_sequences.values():
            frames = sorted(frames)
            # Count number of gaps
            gaps = 0
            for i in range(len(frames) - 1):
                if frames[i+1] - frames[i] > 1:
                    gaps += 1
            fragmented += gaps
        
        return fragmented
    
    def _estimate_mota(self, metrics: Dict) -> float:
        """
        Estimate MOTA (Multi-Object Tracking Accuracy)
        Higher is better, formula: 1 - (FN + FP + IDSW) / GT
        """
        # Simplified estimation based on available metrics
        switch_penalty = metrics['id_switches'] * 0.05
        fragmentation_penalty = metrics['fragmented_tracks'] * 0.02
        
        mota = max(0, 100 - (switch_penalty + fragmentation_penalty))
        return min(100, mota)
    
    def generate_heatmap(self, trajectories: Dict, frame_shape: Tuple[int, int], 
                         team: int = None) -> np.ndarray:
        """
        Generate movement heatmap for players
        
        Args:
            trajectories: Player trajectories
            frame_shape: Shape of output heatmap (height, width)
            team: Filter by team (optional)
            
        Returns:
            Heatmap as numpy array
        """
        heatmap = np.zeros(frame_shape, dtype=np.float32)
        
        for player_id, trajectory in trajectories.items():
            for point in trajectory:
                pos = point['position']
                if pos and len(pos) >= 2:
                    x, y = int(pos[0]), int(pos[1])
                    # Ensure within bounds
                    if 0 <= x < frame_shape[1] and 0 <= y < frame_shape[0]:
                        # Add Gaussian blur around position
                        cv2.circle(heatmap, (x, y), radius=15, color=1, thickness=-1)
        
        # Normalize heatmap
        heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8) * 255
        
        return heatmap.astype(np.uint8)
    
    def save_analytics_report(self, analytics_data: Dict, filepath: str):
        """Save analytics report to JSON file with proper type conversion"""
        def convert_types(obj):
            """Recursively convert numpy types and dict keys to JSON-compatible types"""
            if isinstance(obj, dict):
                # Convert dict keys to strings and recursively convert values
                return {str(k): convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert entire data structure
        converted_data = convert_types(analytics_data)
        
        with open(filepath, 'w') as f:
            json.dump(converted_data, f, indent=2)
    
    def print_analytics_summary(self, analytics_data: Dict):
        """Print formatted analytics summary"""
        print("\n" + "="*60)
        print("FOOTBALL ANALYSIS - ADVANCED ANALYTICS REPORT")
        print("="*60)
        
        if 'possession' in analytics_data:
            poss = analytics_data['possession']
            print(f"\n📊 POSSESSION STATISTICS:")
            print(f"  Team 1: {poss['team_1_percentage']:.1f}% ({poss['team_1_frames']} frames)")
            print(f"  Team 2: {poss['team_2_percentage']:.1f}% ({poss['team_2_frames']} frames)")
        
        if 'player_statistics' in analytics_data:
            print(f"\n👥 TOP PERFORMERS (by distance covered):")
            stats = analytics_data['player_statistics']
            top_players = sorted(stats.items(), key=lambda x: x[1]['total_distance'], reverse=True)[:5]
            for i, (player_id, stat) in enumerate(top_players, 1):
                print(f"  {i}. Player {player_id}: {stat['total_distance']:.1f}m @ {stat['avg_speed']:.2f} m/s")
        
        if 'performance_metrics' in analytics_data:
            metrics = analytics_data['performance_metrics']
            print(f"\n📈 TRACKING PERFORMANCE:")
            print(f"  Players Tracked: {metrics['total_players_tracked']}")
            print(f"  Avg Tracklet Length: {metrics['average_tracklet_length']:.1f} frames")
            print(f"  ID Switches: {metrics['id_switches']}")
            print(f"  Estimated MOTA: {metrics['estimated_mota']:.1f}%")
        
        print("\n" + "="*60 + "\n")


class TrajectoryVisualizer:
    """Handles visualization of player trajectories on video frames"""
    
    @staticmethod
    def draw_trajectory(frame: np.ndarray, trajectory: List[Tuple], 
                       color: Tuple = (0, 255, 0), line_thickness: int = 2) -> np.ndarray:
        """
        Draw a player's trajectory on frame
        
        Args:
            frame: Video frame
            trajectory: List of (x, y) positions
            color: RGB color tuple
            line_thickness: Line thickness
            
        Returns:
            Frame with drawn trajectory
        """
        if len(trajectory) < 2:
            return frame
        
        # Draw trail with fading effect
        for i in range(len(trajectory) - 1):
            pt1 = tuple(map(int, trajectory[i]))
            pt2 = tuple(map(int, trajectory[i + 1]))
            
            # Fade effect - earlier points are fainter
            alpha = (i + 1) / len(trajectory)
            fade_color = tuple(int(c * alpha) for c in color)
            
            cv2.line(frame, pt1, pt2, fade_color, line_thickness)
        
        # Draw circle at current position
        current_pos = tuple(map(int, trajectory[-1]))
        cv2.circle(frame, current_pos, radius=5, color=color, thickness=-1)
        
        return frame
    
    @staticmethod
    def draw_possession_indicator(frame: np.ndarray, team_1_possession: float, 
                                 team_2_possession: float) -> np.ndarray:
        """
        Draw possession bar at top of frame
        
        Args:
            frame: Video frame
            team_1_possession: Team 1 possession percentage
            team_2_possession: Team 2 possession percentage
            
        Returns:
            Frame with possession indicator
        """
        height, width = frame.shape[:2]
        bar_height = 40
        bar_width = width - 40
        bar_x = 20
        bar_y = 20
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)
        
        # Team 1 section (red)
        team1_width = int(bar_width * team_1_possession / 100)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + team1_width, bar_y + bar_height),
                     (0, 0, 255), -1)
        
        # Team 2 section (blue)
        cv2.rectangle(frame, (bar_x + team1_width, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     (255, 0, 0), -1)
        
        # Text
        cv2.putText(frame, f"Possession: {team_1_possession:.1f}% | {team_2_possession:.1f}%",
                   (bar_x + 10, bar_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (255, 255, 255), 2)
        
        return frame
