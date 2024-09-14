# Step-by-Step to build GPT from Scratch

## Description
This repository is a step-by-step reproduction of the code from Andrej Karpathy's YouTube video: [Let's Build GPT from Scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6010s). The code follows the lecture closely, providing a practical implementation of a GPT-like model from scratch. All credit goes to Andrej Karpathy

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [License](#license)
- [Contact](#contact)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/slayermm1122/reprod_ak_gpt2.git
   ```
2. Navigate to the project directory:
   ```bash
   cd reprod_ak_gpt2
   ```
3. Create and activate a virtual environment:
   - On macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     python -m venv venv
     venv\Scripts\ctivate
     ```
4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Simply follow Andrej Karpathy's YouTube lecture and proceed step by step. Each file in this repository corresponds to a part of his lecture, and they build on each other progressively.

### File Descriptions (in the order of the lecture):
- `bigram.py`: Implements the basic bigram model.
- `tok_pos_emb.py`: Adds token and positional embeddings.
- `single_head.py`: Implements a single-head attention mechanism.
- `multi_head.py`: Expands the attention mechanism to multi-head attention.
- `feed_forward.py`: Implements the feed-forward neural network component.
- `block.py`: Combines multiple components into a block.
- `resnet.py`: Introduces residual connections into the architecture.
- `layernorm.py`: Adds Layer Normalization for stabilizing the training.
- `dropout.py`: Implements dropout for regularization.
- `combine_head.py`: combine single and multi-head into one batch dimension.
- `scale.py`: Scales the output properly to prevent vanishing gradients.

Each file builds on the previous file as per the lecture.

## Features
- **Time Logging**: The code prints the time cost of each training step.
- **Checkpointing**: The model automatically saves checkpoints after training, allowing you to resume or test using `test.py` with your trained model (saved as `.pth`).
- **Multi-Head Dim**: The implementation merges single-head and multi-head classes into a unified batch dimension, enabling parallel processing for improved computational efficiency.

## Computation
The code implements two GPT models: a small 45,000-parameter version in combine_head.py and a larger 10 million-parameter GPT-2 in scale.py. For reference, training the smaller model took 45 seconds, while the larger one required 3 hours on a MacBook M1 Pro with 16GB of RAM.

## License
**Disclaimer**: This repository is for educational purposes only and not for commercial use. All code is based on Andrej Karpathy's YouTube lecture: Let's Build GPT from Scratch.

## Credits
This project is based on the work of [Andrej Karpathy](https://github.com/karpathy). Special thanks for his YouTube video [Let's Build GPT from Scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6010s).

Some parts of the code are adapted from his repository: [ng-video-lecture](https://github.com/karpathy/ng-video-lecture).


## Contact
If you have any questions or suggestions, feel free to send a message to me.
