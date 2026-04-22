"""
Enhanced Main Pipeline with Advanced Analytics
Features:
- GPU auto-detection for faster processing
- Advanced analytics (possession, trajectories, performance metrics)
- Configuration file support
- Real-time statistics display
- Trajectory visualization
- Performance metrics calculation
"""

from utils import read_video, save_video
from trackers import Tracker
from analytics.advanced_analytics import AdvancedAnalytics, TrajectoryVisualizer
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import torch
import yaml
from pathlib import Path


class PipelineConfig:
    """Load and manage configuration from YAML file"""
    
    def __init__(self, config_path='config.yaml'):
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration if file not found
            self.config = self._get_default_config()
    
    def _get_default_config(self):
        return {
            'VIDEO': {
                'INPUT_PATH': 'input_videos/1.mp4',
                'OUTPUT_PATH': 'output_videos/output_video.avi',
                'FRAME_STRIDE': 1
            },
            'DETECTION': {
                'MODEL_PATH': 'models/best.pt',
                'CONFIDENCE_THRESHOLD': 0.1,
                'USE_GPU': torch.cuda.is_available()
            },
            'ANALYTICS': {
                'CALCULATE_TRAJECTORIES': True,
                'GENERATE_HEATMAP': True,
                'CALCULATE_POSSESSION': True,
                'CALCULATE_PERFORMANCE_METRICS': True
            }
        }
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value else default


def detect_gpu_availability():
    """Detect and report GPU availability"""
    if torch.cuda.is_available():
        print(f"✓ GPU Detected: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("✗ GPU not available. Using CPU (slower processing)")
        return False


def enhanced_main(config_path='config.yaml'):
    """
    Enhanced main pipeline with advanced analytics
    
    Args:
        config_path: Path to configuration YAML file
    """
    
    print("\n" + "="*70)
    print("⚽ ADVANCED FOOTBALL ANALYSIS PIPELINE - MULTI-OBJECT TRACKING")
    print("="*70 + "\n")
    
    # Load configuration
    config = PipelineConfig(config_path)
    use_gpu = detect_gpu_availability()
    
    # Get paths from config or use defaults
    video_path = config.get('VIDEO.INPUT_PATH', 'input_videos/1.mp4')
    output_path = config.get('VIDEO.OUTPUT_PATH', 'output_videos/output_video.avi')
    model_path = config.get('DETECTION.MODEL_PATH', 'models/best.pt')
    
    print(f"📹 Input Video: {video_path}")
    print(f"🎬 Output Path: {output_path}\n")
    
    # Read Video
    print("📂 Loading video frames...")
    video_frames = read_video(video_path)
    print(f"✓ Loaded {len(video_frames)} frames\n")
    
    # Initialize Tracker
    print("🔍 Initializing YOLO v5 detector...")
    tracker = Tracker(model_path)
    
    print("🎯 Running object detection & ByteTrack tracking...")
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=False,
                                       stub_path='stubs/track_stubs.pkl')
    print(f"✓ Detected and tracked {len(set(p for frame in tracks['players'] for p in frame.keys()))} unique players\n")
    
    # Get object positions 
    print("📍 Computing object positions...")
    tracker.add_position_to_tracks(tracks)
    
    # Camera movement estimator
    print("📷 Estimating camera movement (Optical Flow)...")
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=False,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    print("✓ Camera movement compensated\n")
    
    # View Transformer
    print("🔄 Applying perspective transformation...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # Interpolate Ball Positions
    print("⚪ Interpolating ball positions...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # Speed and distance estimator
    print("📊 Computing speed & distance metrics...")
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    print("✓ Speed and distance calculated\n")
    
    # Assign Player Teams
    print("👕 Assigning players to teams (K-means color clustering)...")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    print("✓ Team assignment complete\n")
    
    # Assign Ball Acquisition
    print("🏃 Computing ball possession...")
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
    
    team_ball_control = np.array(team_ball_control)
    print("✓ Ball possession calculated\n")
    
    # ADVANCED ANALYTICS
    print("📈 Computing advanced analytics & performance metrics...")
    analytics = AdvancedAnalytics()
    
    # Compute possession
    possession = analytics.compute_possessions(tracks, team_ball_control)
    
    # Build trajectories
    trajectories = analytics.build_trajectories(tracks)
    
    # Compute player statistics
    player_stats = analytics.compute_player_statistics(tracks)
    
    # Compute performance metrics
    performance_metrics = analytics.compute_performance_metrics(tracks)
    
    print("✓ Advanced analytics computed\n")
    
    # Generate heatmap
    if config.get('ANALYTICS.GENERATE_HEATMAP', True):
        print("🔥 Generating movement heatmap...")
        heatmap = analytics.generate_heatmap(trajectories, (video_frames[0].shape[0], video_frames[0].shape[1]))
        
        # Apply colormap to heatmap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Save heatmap
        heatmap_path = 'output_videos/movement_heatmap.png'
        cv2.imwrite(heatmap_path, heatmap_colored)
        print(f"✓ Heatmap saved to {heatmap_path}")
    
    # Compile analytics data
    analytics_data = {
        'possession': possession,
        'player_statistics': player_stats,
        'performance_metrics': performance_metrics,
        'config': config.config
    }
    
    # Print analytics summary
    analytics.print_analytics_summary(analytics_data)
    
    # Save analytics report
    analytics.save_analytics_report(analytics_data, 'output_videos/analytics_report.json')
    print("✓ Analytics saved to output_videos/analytics_report.json\n")
    
    # Draw output with advanced visualizations
    print("🎨 Generating annotated video with visualizations...")
    
    # Draw object Tracks (with trajectories if enabled)
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    
    # Add trajectory visualization
    if config.get('VISUALIZATION.DRAW_TRAJECTORIES', True):
        trajectory_viz = TrajectoryVisualizer()
        for frame_num, frame in enumerate(output_video_frames):
            # Draw trajectories for each player visible in this frame
            for player_id, trajectory_data in trajectories.items():
                # Get last N positions for this player near current frame
                recent_positions = [
                    point['position'] for point in trajectory_data 
                    if abs(point['frame'] - frame_num) < config.get('VISUALIZATION.TRAJECTORY_LENGTH', 30)
                ]
                
                if recent_positions and len(recent_positions) > 1:
                    # Determine player team for color
                    player_team = None
                    if frame_num < len(tracks['players']) and player_id in tracks['players'][frame_num]:
                        player_team = tracks['players'][frame_num][player_id].get('team', 1)
                    
                    color = (0, 0, 255) if player_team == 1 else (255, 0, 0)  # Red or Blue
                    output_video_frames[frame_num] = trajectory_viz.draw_trajectory(
                        frame, recent_positions, color, line_thickness=1
                    )
    
    # Add possession indicator bar
    if config.get('VISUALIZATION.DRAW_POSSESSION_BAR', True):
        trajectory_viz = TrajectoryVisualizer()
        for frame_num, frame in enumerate(output_video_frames):
            output_video_frames[frame_num] = trajectory_viz.draw_possession_indicator(
                frame, possession['team_1_percentage'], possession['team_2_percentage']
            )
    
    # Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame
    )
    
    # Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
    
    print("✓ Annotations complete\n")
    
    # Save video
    print("💾 Saving output video...")
    save_video(output_video_frames, output_path)
    print(f"✓ Output video saved to {output_path}\n")
    
    print("="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print(f"\n📊 Key Results:")
    print(f"  • Total Frames Processed: {len(video_frames)}")
    print(f"  • Unique Players Tracked: {performance_metrics['total_players_tracked']}")
    print(f"  • Team 1 Possession: {possession['team_1_percentage']:.1f}%")
    print(f"  • Team 2 Possession: {possession['team_2_percentage']:.1f}%")
    print(f"  • Tracking Accuracy (MOTA): {performance_metrics['estimated_mota']:.1f}%")
    print(f"  • ID Switches: {performance_metrics['id_switches']}")
    print(f"\n📁 Output Files:")
    print(f"  • Video: {output_path}")
    print(f"  • Analytics: output_videos/analytics_report.json")
    print(f"  • Heatmap: output_videos/movement_heatmap.png")
    print("\n")


if __name__ == '__main__':
    enhanced_main('config.yaml')
