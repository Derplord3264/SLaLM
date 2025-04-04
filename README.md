# SLaLM

An experimental conversational AI agent built using **Spiking Neural Networks (SNNs)** with the **`snntorch`** library. This project explores the feasibility of using bio-inspired SNNs for natural language processing tasks, specifically dialogue generation.

## Features

*   **Data Preprocessing:** Parses the Cornell Movie Dialogues Corpus into context-response pairs (`initialize.py`).
*   **SNN Model:** Implements a sequence-to-sequence like architecture using `snntorch` layers (`snn.Leaky`, `snn.Synaptic`) combined with standard PyTorch embeddings and linear layers (`train.py`).
*   **Surrogate Gradient Training:** Uses backpropagation with surrogate gradients (`snntorch.surrogate`) to train the SNN.
*   **Experimental STDP:** Includes an optional, experimental implementation of Spike-Timing-Dependent Plasticity (STDP) that can be applied *alongside* backpropagation (`--use_stdp` flag in `train.py`).
*   **Interactive Chat:** Allows real-time interaction with the trained SNN model (`chat.py`).
*   **Verbose Logging & Configuration:** Scripts include detailed logging and command-line arguments for configuration.

## Motivation & Novelty

Traditional chatbots often rely on large RNNs or Transformers. This project investigates the potential of SNNs, which offer theoretical advantages in energy efficiency and biological plausibility, for the complex task of dialogue generation.

Key explorations include:
1.  Implementing a text-generating SNN using `snntorch`.
2.  Combining standard backpropagation with biologically-inspired STDP learning rules in a hybrid approach.
3.  Demonstrating the capabilities and challenges of current SNNs for NLP tasks at a level suitable for science fair investigation.

The implementation of a functional SNN chatbot attempting text generation, especially with STDP exploration, by a high school freshman is believed to be a novel undertaking.

## Project Structure

*   `initialize.py`: Script to download (if needed) and preprocess the Cornell Movie Dialogues dataset. Creates vocabulary and conversation pairs.
*   `train.py`: Defines the `SpikeDialogueModel` (SNN architecture) and contains the main training loop, including validation, checkpointing, surrogate gradient backpropagation, and the optional STDP updates.
*   `chat.py`: Provides an interactive command-line interface to load a trained checkpoint and chat with the SNN bot.
*   `data/`: (Needs to be created) Directory intended to hold the raw dataset (e.g., `movie_lines.txt`).
*   `processed/`: (Created by `initialize.py`) Directory holding the processed vocabulary (`word2idx.pt`) and conversation pairs (`processed_pairs.json`).
*   `checkpoints/`: (Created by `train.py`) Directory where model checkpoints are saved during training.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    *(You need to create a `requirements.txt` file!)*
    ```bash
    pip install torch snntorch tqdm numpy # Add any other libraries you used
    # Create requirements.txt: pip freeze > requirements.txt
    pip install -r requirements.txt
    ```

## Usage

1.  **Preprocess Data:**
    ```bash
    python initialize.py
    ```
    This will create the `processed/` directory with `word2idx.pt` and `processed_pairs.json`.

2.  **Train the Model:**
    ```bash
    # Basic training
    python train.py --epochs 20 --train_size 10000 --val_size 2000

    # Training with STDP enabled
    python train.py --epochs 20 --train_size 10000 --val_size 2000 --use_stdp

    # See available options
    python train.py --help
    ```
    Checkpoints will be saved in the `checkpoints/` directory.

3.  **Chat with the Bot:**
    *   Find the path to a saved checkpoint (e.g., `checkpoints/checkpoint_epoch_10.pt`).
    ```bash
    python chat.py --model checkpoints/your_checkpoint_file.pt --temp 0.8 --top_k 50
    ```
    *   Type your message and press Enter.
    *   Use `reset` to clear the SNN state (useful if generation becomes strange).
    *   Use `exit` to quit.

## Model Details

*   **Dataset:** Cornell Movie Dialogues Corpus
*   **Architecture:**
    *   `nn.Embedding`
    *   `nn.Linear` projection
    *   `snn.Leaky` (Leaky Integrate-and-Fire neuron layer)
    *   `nn.Linear` projection
    *   `snn.Synaptic` (Leaky Integrate-and-Fire neuron with synaptic current dynamics)
*   **Training:** AdamW optimizer, CrossEntropyLoss (on spike outputs via surrogate gradients), ReduceLROnPlateau scheduler.
*   **STDP (Optional):** Simple Hebbian-style update based on mean pre- and post-synaptic activity within a time window, applied to `fc1` weights.

## Results & Analysis (Ongoing)

Currently under investigation. Planned analysis includes:

*   Training/validation loss curves (with vs. without STDP).
*   Average network spike rates during training.
*   Qualitative examples of generated conversations.
*   Comparison of generation quality (coherence, relevance) between models trained with and without STDP.
*   Analysis of challenges encountered (e.g., training stability, vanishing/exploding spikes, impact of hyperparameters).

## Future Work

*   Implement more sophisticated, time-dependent STDP rules.
*   Explore different SNN neuron models and network architectures (e.g., recurrent SNN layers).
*   Investigate methods for maintaining conversational context (state) across multiple turns using SNN states.
*   Train on larger datasets and for more epochs.
*   Perform quantitative analysis of computational cost / potential energy efficiency compared to traditional ANNs (though this is challenging without specialized hardware).
*   Improve the STDP update mechanism and target synapse selection.

## Contributing

Contributions, issues, and feature requests are welcome! Please open an issue first to discuss what you would like to change.