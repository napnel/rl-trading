from configparser import MAX_INTERPOLATION_DEPTH
from doctest import master
from email.contentmanager import raw_data_manager

import gym
import ray
import streamlit as st
import torch
from gym.wrappers.record_video import RecordVideo
from ray import tune
from ray.rllib.agents import PPOTrainer, Trainer


def train():
    agent = PPOTrainer(config=config)


def test():
    pass


def main():
    config = {
        "env": "CartPole-v1",
        "framework": "torch",
        "evaluation_num_workers": 0,  # local
    }
    agent: Trainer = PPOTrainer(config=config)
    for _ in range(10):
        result = agent.train()
        print(result)

    agent.workers.local_worker()
    env = gym.make("CartPole-v1")
    env = RecordVideo(env, "cartpole-video.mp4")
    obs = env.reset()
    done = False
    rewards = []
    while not done:
        prob = model.pi(torch.from_numpy(obs).float())
        action = Categorical(prob).sample().item()
        obs, reward, done, info = env.step(action)
        rewards.append(reward)

    print("Reward sum:", sum(rewards))
    env.close()


if __name__ == "__main__":
    main()
    st.title("Small RL App")
    st.video("cartpole-video.mp4/rl-video-episode-0.mp4")
