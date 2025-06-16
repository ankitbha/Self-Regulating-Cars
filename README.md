# Self-Regulating-Cars

This repository contains the materials used to perform the experiments discussed in this paper: "Self-Regulating Cars: Automating Traffic Control in Free Flow Road Networks."

---
# Abstract

Road traffic congestion is a growing challenge in suburban and peri-urban areas, where free-flow road networks lack signaling infrastructure. Installing traffic signals in such settings is often infeasible due to cost  and scalability constraints. Motivated by the rise of autonomous vehicles, we propose the concept of *self-regulating cars*—vehicles that automatically regulate traffic through centralized speed modulation. We formulate traffic control as a reinforcement learning (RL) problem and design a Q-learning agent that integrates classical traffic flow principles, including volume-density relationships and gap acceptance theory, into its state and reward design. Our protocol is evaluated in PTV Vissim, a high-fidelity microscopic traffic simulator, using a real-world road network in Mainz, Germany. Compared to conventional signaling baselines, self-regulating cars significantly improve throughput and flow smoothness metrics while demonstrating robust generalization across traffic patterns, highlighting the potential of domain-grounded RL for scalable, infrastructure-free traffic management.

To read more about the paper, please refer to the following link: [Self-Regulating Cars: Automating Traffic Control in Free Flow Road Networks](https://arxiv.org/abs/2506.11973).

If you find this work useful, please cite it using the following BibTeX entry:

```bibtex
@misc{bhardwaj2025selfregulatingcarsautomatingtraffic,
      title={Self-Regulating Cars: Automating Traffic Control in Free Flow Road Networks}, 
      author={Ankit Bhardwaj and Rohail Asim and Sachin Chauhan and Yasir Zaki and Lakshminarayanan Subramanian},
      year={2025},
      eprint={2506.11973},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.11973}, 
}
```

---
# Files

The repository is organized into the following key directories and files:

## Layout
The `Layout` folder contains **traffic network layouts specifically designed for use with PTV VISSIM**, a microscopic traffic flow simulation software. These layouts represent various road network configurations and serve as the **virtual environments** in which the self-regulating car models are tested. Each layout is crafted to simulate realistic traffic scenarios, allowing for comprehensive evaluation of the proposed traffic control mechanisms in direct comparison to different baselines.

## RL.py
The `RL.py` script is the core component for the **Reinforcement Learning (RL) model**. This script handles the entire process of **training and running the RL model** that is designed to optimize traffic flow. It interacts with the PTV VISSIM simulations, receives feedback on traffic conditions, and adjusts the behavior of the self-regulating cars accordingly.
