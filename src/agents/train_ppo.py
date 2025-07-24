import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.rl_env.trading_env import TradingEnv
from src.forecast.xgboost_forecast import run_forecast

def train_agent(ticker='AAPL'):
    # Load data
    df = pd.read_csv(f"data/{ticker}.csv")

    # Get forecast and models (you can extend to save model too)
    model, scaler, future_pred, preds, y_test, metrics = run_forecast(df)

    # Create environment with forecast as feature
    env = DummyVecEnv([lambda: TradingEnv(df, future_pred)])

    # Initialize PPO agent
    agent = PPO("MlpPolicy", env, verbose=1)
    agent.learn(total_timesteps=10000)

    agent.save(f"models/ppo_{ticker}")

    print(f"Training complete for {ticker}")

if __name__ == "__main__":
    train_agent()
