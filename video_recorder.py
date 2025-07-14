import cv2
import numpy as np
import os
import torch
import gymnasium as gym
from typing import List, Tuple, Optional
import tempfile
import shutil


class VideoRecorder:
    """
    Video recorder for RL agent testing.
    Records multiple trials and saves only the best performing one.
    """
    
    def __init__(self, video_path: str = "videos/", fps: int = 50):
        """
        Initialize the video recorder.
        
        Args:
            video_path: Directory to save videos
            fps: Frames per second for the video (CarRacing typically runs at ~50 fps)
        """
        self.video_path = video_path
        self.fps = fps
        os.makedirs(video_path, exist_ok=True)
        
    def record_best_trial(self, algorithm: str, model_path: str, device: torch.device, 
                         env: gym.Env, num_trials: int = 5, img_stack: int = 4, 
                         gae_lambda: float = 0.0, action_repeat: int = 1) -> Tuple[float, str]:
        """
        Record multiple trials and save only the best performing one.
        
        Args:
            algorithm: The RL algorithm used (dqn, dueling_dqn, ppo)
            model_path: Path to the trained model
            device: Torch device
            env: Gymnasium environment
            num_trials: Number of trials to run
            img_stack: Number of stacked images (for PPO)
            gae_lambda: GAE lambda parameter (for PPO)
            action_repeat: Number of action repeats (affects video speed)
            
        Returns:
            Tuple of (best_score, video_filename)
        """
        print(f"Recording {num_trials} trials to find the best performance...")
        
        # Adjust fps based on action repeat to get natural speed
        if algorithm == "ppo" and action_repeat > 1:
            # For PPO with action repeat, we get fewer unique frames
            # Adjust fps to compensate for the action repeat
            actual_fps = max(8, self.fps // action_repeat)
            print(f"PPO detected with action_repeat={action_repeat}, adjusting fps from {self.fps} to {actual_fps}")
        else:
            actual_fps = self.fps
            
        print(f"Using {actual_fps} fps for video recording")
        
        trials_data = []
        temp_videos = []
        
        # Create a temporary directory for trial videos
        temp_dir = tempfile.mkdtemp()
        
        try:
            for trial in range(num_trials):
                print(f"Recording trial {trial + 1}/{num_trials}")
                
                # Create temporary video file for this trial
                temp_video_path = os.path.join(temp_dir, f"trial_{trial}.mp4")
                temp_videos.append(temp_video_path)
                
                # Record this trial
                score, frames = self._record_single_trial(
                    algorithm, model_path, device, env, img_stack, gae_lambda
                )
                
                # Save frames to temporary video with adjusted fps
                self._save_frames_to_video(frames, temp_video_path, actual_fps)
                
                trials_data.append((score, trial, temp_video_path))
                print(f"Trial {trial + 1} score: {score:.2f}")
            
            # Find the best trial
            best_score, best_trial_idx, best_temp_video = max(trials_data, key=lambda x: x[0])
            
            # Create final video filename
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            final_video_name = f"{algorithm}_{model_name}_best_score_{best_score:.2f}.mp4"
            final_video_path = os.path.join(self.video_path, final_video_name)
            
            # Copy the best trial video to final location
            shutil.copy2(best_temp_video, final_video_path)
            
            print(f"\n=== RECORDING RESULTS ===")
            print(f"Best trial: {best_trial_idx + 1}")
            print(f"Best score: {best_score:.2f}")
            print(f"Video saved as: {final_video_path}")
            print(f"Video fps: {actual_fps}")
            print(f"All trial scores: {[data[0] for data in trials_data]}")
            print(f"========================\n")
            
            return best_score, final_video_path
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _record_single_trial(self, algorithm: str, model_path: str, device: torch.device,
                           env: gym.Env, img_stack: int, gae_lambda: float) -> Tuple[float, List[np.ndarray]]:
        """
        Record a single trial and return the score and frames.
        
        Returns:
            Tuple of (score, frames_list)
        """
        frames = []
        score = 0
        
        # Load the appropriate model and run the trial
        if algorithm == "dqn":
            score, frames = self._run_dqn_trial(model_path, device, env)
        elif algorithm == "dueling_dqn":
            score, frames = self._run_dueling_dqn_trial(model_path, device, env)
        elif algorithm == "ppo":
            score, frames = self._run_ppo_trial(model_path, device, env, img_stack, gae_lambda)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return score, frames
    
    def _run_dqn_trial(self, model_path: str, device: torch.device, env: gym.Env) -> Tuple[float, List[np.ndarray]]:
        """Run a single DQN trial and collect frames."""
        from dqn.q_network import DQN
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        policy_net_state_dict = checkpoint['model_state_dict']
        
        # Setup network
        n_actions = env.action_space.n
        state, _ = env.reset()
        n_observations = state.flatten().shape[0]
        
        policy_net = DQN(n_observations, n_actions).to(device)
        policy_net.load_state_dict(policy_net_state_dict)
        policy_net.eval()
        
        # Run episode
        state, _ = env.reset()
        frames = []
        score = 0
        done = False
        step_count = 0
        
        while not done and step_count < 2000:
            # Render and capture frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            
            # Select action
            tensor_state = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy_net(tensor_state).max(1).indices.view(1, 1)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            score += reward
            step_count += 1
            state = next_state
        
        return score, frames
    
    def _run_dueling_dqn_trial(self, model_path: str, device: torch.device, env: gym.Env) -> Tuple[float, List[np.ndarray]]:
        """Run a single Dueling DQN trial and collect frames."""
        from dueling_dqn.q_network import QNetwork
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            dueling_dqn_param = checkpoint['model_state_dict']
        else:
            dueling_dqn_param = checkpoint['dueling-dqn-param']
        
        # Setup network
        agent = QNetwork().to(device)
        agent.load_state_dict(dueling_dqn_param)
        agent.eval()
        
        # Run episode
        state, _ = env.reset()
        frames = []
        score = 0
        done = False
        step_count = 0
        
        while not done and step_count < 2000:
            # Render and capture frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            
            # Select action
            tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = agent.select_action(tensor_state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            score += reward
            step_count += 1
            state = next_state
        
        return score, frames
    
    def _run_ppo_trial(self, model_path: str, device: torch.device, env, 
                      img_stack: int, gae_lambda: float) -> Tuple[float, List[np.ndarray]]:
        """Run a single PPO trial and collect frames."""
        from ppo.network import Agent, AgentGAE
        
        # Setup agent
        if gae_lambda == 0:
            agent = Agent(env.action_dim, path=model_path, device=device, img_stack=img_stack)
        else:
            agent = AgentGAE(env.action_dim, path=model_path, device=device, img_stack=img_stack)
        
        agent.load_param()
        
        # Run episode
        state = env.reset()
        frames = []
        score = 0
        done = False
        die = False
        step_count = 0
        
        while not done and not die and step_count < 1000:
            # Render and capture frame from the underlying environment
            frame = env.env.render()
            if frame is not None:
                frames.append(frame)
            
            # Select action
            action = agent.select_test_action(state)
            
            # Take action
            state_, reward, done, die = env.step(action)
            
            score += reward
            step_count += 1
            state = state_
        
        return score, frames
    
    def _save_frames_to_video(self, frames: List[np.ndarray], video_path: str, fps: Optional[int] = None):
        """Save a list of frames to a video file."""
        if not frames:
            print("Warning: No frames to save")
            return
        
        # Use provided fps or default
        video_fps = fps if fps is not None else self.fps
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, video_fps, (width, height))
        
        try:
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                out.write(frame_bgr)
        finally:
            out.release()


def record_best_performance(args, env, device):
    """
    Main function to handle video recording of the best performance.
    
    Args:
        args: Parsed command line arguments
        env: Gymnasium environment
        device: Torch device
    """
    if not args.load:
        print("No model specified for recording. Use --load to specify a model.")
        return
    
    print("Starting video recording...")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  Model path: {args.load}")
    print(f"  Number of trials: {args.record_trials}")
    print(f"  Video directory: {args.video_path}")
    
    # Create video recorder
    recorder = VideoRecorder(video_path=args.video_path, fps=args.video_fps)
    
    # Handle environment wrapping for PPO
    if args.algorithm == "ppo":
        from ppo.env import Env
        wrapped_env = Env(env=env, img_stack=args.img_stack, action_repeat=args.action_repeat)
        recording_env = wrapped_env
    else:
        recording_env = env
    
    # Record the best trial
    try:
        best_score, video_path = recorder.record_best_trial(
            algorithm=args.algorithm,
            model_path=args.load,
            device=device,
            env=recording_env,
            num_trials=args.record_trials,
            img_stack=args.img_stack,
            gae_lambda=args.gae_lambda,
            action_repeat=getattr(args, 'action_repeat', 1)
        )
        
        print(f"Successfully recorded best performance: {best_score:.2f}")
        print(f"Video saved at: {video_path}")
        
    except Exception as e:
        print(f"Error during recording: {e}")
        raise
