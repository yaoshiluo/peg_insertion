# Reinforcement Learning for Peg Insertion with Force Sensing

This project is built on the Isaac Lab framework and focuses on reinforcement learning for contact-rich peg-in-hole insertion tasks. The system is trained to perform precise and compliant insertion using force sensing and hybrid control strategies, aiming for robust sim-to-real policy transfer.

The environment supports parallelized training using thousands of simulations, leveraging domain randomization and sparse + dense reward structures.

## Environment Setup

Please follow the official Isaac Lab installation guide first:  
[https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

Then activate your Isaac Lab Conda environment:

```
conda activate env_isaaclab
```

Navigate to the `source` directory and install the module (if applicable):

```
cd ~/peg_insertion/source
pip install -e .
```

## Training the Policy

To start training the peg insertion task using reinforcement learning, run:

```
python scripts/rl_games/train.py --task=Isaac-Factory-PegInsert-Direct-v0 --num_envs 64 --headless
```

The training will use 64 parallel environments and run in headless mode.

## Logs and Output

- Training logs, checkpoints, and TensorBoard summaries are saved in:

```
/home/fortiss/peg_insertion/logs
```

- Evaluation videos and success metrics may be saved in:

```
/home/fortiss/peg_insertion/outputs
```

- Raw contact force data (for debugging or reward shaping) is stored in:

```
/home/fortiss/peg_insertion/contact_force_logs
```
- To record and debug contact force data, you must run the policy using the following command with a single environment:

```bash
python scripts/rl_games/play.py --task=Isaac-Factory-PegInsert-Direct-v0 --num_envs 1 --headless
```

