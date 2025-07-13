import argparse
import gymnasium as gym
import numpy as np
import torch
from plot import plot_training_progress
from main import QNetwork

path = '/home/faggi/repo/caRL/fatto/dueling_dqn_1us_64b_900.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_saved_model(model_path=None, num_episodes=10):
    """
    Testa la rete salvata durante il training.
    Prima di iniziare il test mostra il grafico del training progress.
    
    Args:
        model_path: Percorso del modello salvato (se None usa il path di default)
        num_episodes: Numero di episodi di test da eseguire
    """
    if model_path is None:
        model_path = path
    
    print(f'Testing saved model from: {model_path}')
    
    # Carica il modello salvato
    try:
        checkpoint = torch.load(model_path, map_location=device)
        rewards = checkpoint['rewards']
        episodes = list(range(len(rewards)))
        final_episode = checkpoint['i_ep']
        
        print(f'Loaded model trained for {final_episode} episodes')
        
        # Mostra il grafico del training progress
        print("Displaying training progress...")
        plot_training_progress(scores=rewards, episodes=episodes)
        print("Training progress plot saved in plots/dueling_dqn_training_progress.png")
        
        # Inizializza l'ambiente per il test
        env = gym.make('CarRacing-v3', domain_randomize=False, continuous=False, render_mode='human')
        
        # Inizializza la rete e carica i parametri
        agent = QNetwork().to(device)
        agent.load_state_dict(checkpoint['dueling-dqn-param'])
        agent.eval()  # Modalità evaluation
        
        print(f"\nStarting test on {num_episodes} episodes...")
        test_rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            
            while not done:
                # Usa solo la policy greedy (no epsilon-greedy)
                tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = agent.select_action(tensor_state)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                step_count += 1
                state = next_state
                
                # Limite di sicurezza per evitare loop infiniti
                if step_count > 2000:
                    print(f"Episode {episode + 1} stopped after {step_count} steps")
                    break
            
            test_rewards.append(episode_reward)
            print(f"Episode {episode + 1}/{num_episodes}: Score = {episode_reward:.2f}, Steps = {step_count}")
        
        env.close()
        
        # Statistiche finali
        avg_reward = np.mean(test_rewards)
        std_reward = np.std(test_rewards)
        max_reward = np.max(test_rewards)
        min_reward = np.min(test_rewards)
        
        print(f"\n=== TEST RESULTS ===")
        print(f"Episodes tested: {num_episodes}")
        print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Max reward: {max_reward:.2f}")
        print(f"Min reward: {min_reward:.2f}")
        print(f"Training episodes: {final_episode}")
        print(f"Training average (last 100 episodes): {np.mean(rewards[-100:]):.2f}")
        
        return test_rewards
        
    except FileNotFoundError:
        print(f'No saved model found at: {model_path}')
        return None
    except Exception as e:
        print(f'Error loading model: {e}')
        return None

def main():
    parser = argparse.ArgumentParser(description='Test Dueling DQN saved model')
    parser.add_argument('--model-path', type=str, default=path,
                        help='Path to the saved model (default: saved_models/dueling_dqn_model.pt)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of test episodes to run (default: 10)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DUELING DQN MODEL TEST")
    print("=" * 60)
    
    # Esegui il test
    test_rewards = test_saved_model(args.model_path, args.episodes)
    
    if test_rewards is not None:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")

if __name__ == "__main__":
    main()
