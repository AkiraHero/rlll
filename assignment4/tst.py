from core.utils import load_progress
import matplotlib.pyplot as plt
import seaborn as sns

ppo_progress = load_progress("CartPole/PPO")
plt.figure(dpi=300)
sns.set("notebook", "darkgrid")
ax = sns.lineplot(
    data=ppo_progress,
    x="total_steps",
    y="training_episode_reward/episode_reward_mean"
)
ax.set_title("A2C training result in MetaDrive-Tut-Easy-v0")
ax.set_ylabel("Episode Reward Mean")
ax.set_xlabel("Sampled Steps")
ax.annotate("REF", (ax.get_xlim()[1] / 3, ax.get_ylim()[0]), size=100, alpha=0.05)