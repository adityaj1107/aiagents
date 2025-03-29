# Rust DQN

This project implements a Deep Q-Network (DQN) agent in Rust to learn how to interact with a simulated tree environment (`StructuredTreeEnv`). The agent uses a Multi-Layer Perceptron (MLP) Q-Network built with the `tch-rs` crate (Rust bindings for LibTorch/PyTorch).

This code is a translation of a similar Python implementation (`dqn_mlp_tree_simulator.py`).

## Features

* **DQN Agent:** Implements the core DQN algorithm.
* **MLP Q-Network:** Uses a simple feed-forward neural network defined using `tch-rs`.
* **Experience Replay:** Utilizes a replay buffer (`VecDeque`) to store and sample past experiences.
* **Epsilon-Greedy Exploration:** Balances exploration and exploitation during training.
* **Target Network:** Uses a target network for stable Q-value estimation, updated via soft updates.
* **Simulated Environment:** Includes a basic `StructuredTreeEnv` simulating tree growth dynamics based on actions (Water, Prune, Fertilize, Wait).

## Dependencies

* **Rust Toolchain:** Requires `cargo` and the Rust compiler (latest stable recommended). [Install Rust](https://www.rust-lang.org/tools/install)
* **LibTorch:** This is the C++ distribution of PyTorch and is **required** by the `tch-rs` crate. You MUST download and configure LibTorch *before* attempting to compile this project.
    * Follow the setup instructions carefully: [tch-rs Getting Started](https://github.com/LaurentMazare/tch-rs#getting-started)
    * This typically involves:
        1.  Downloading the correct LibTorch package (CPU or CUDA version) for your OS from the [PyTorch website](https://pytorch.org/get-started/locally/).
        2.  Extracting the archive.
        3.  Setting the `LIBTORCH` environment variable to point to the extracted `libtorch` directory.
        4.  (On Linux/macOS) Potentially adding the `libtorch/lib` directory to your `LD_LIBRARY_PATH` environment variable.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd dqn_tree_rust
    ```
2.  **Install LibTorch:** Follow the [tch-rs Getting Started](https://github.com/LaurentMazare/tch-rs#getting-started) guide linked above to download and set up LibTorch for your system. Make sure the necessary environment variables (`LIBTORCH`, etc.) are set correctly in your terminal session.

## How to Run

1.  **Build the Project:**
    ```bash
    # Build in release mode for performance
    cargo build --release
    ```
    *(If you encounter build errors, double-check your LibTorch installation and environment variables.)*

2.  **Run the Training:**
    ```bash
    cargo run --release
    ```

    You should see output indicating initialization, buffer pre-population, and then training progress per episode, showing average rewards and epsilon decay.

## Code Structure

* `Cargo.toml`: Defines project metadata and dependencies (`tch-rs`, `rand`, `chrono`).
* `src/main.rs`: Contains the entire Rust code:
    * `Config`: Struct for hyperparameters.
    * `StructuredTreeEnv`: The environment simulation struct and methods.
    * `Experience`: Struct representing a single state transition.
    * `ReplayBuffer`: Struct and methods for experience replay.
    * `QNetwork`: Struct defining the MLP network using `tch-rs`.
    * `DQNAgent`: The main agent struct containing networks, buffer, optimizer, and learning logic.
    * `main()`: The main function that sets up the environment, agent, and runs the training loop.

## Configuration

Hyperparameters (learning rate, buffer size, epsilon decay, etc.) are defined within the `Config` struct and its `Default` implementation in `src/main.rs`. You can modify these values directly in the code.

## Notes & Limitations

* **LibTorch Dependency:** This code *will not compile or run* without a correctly configured LibTorch installation.
* **`tch-rs` Complexity:** While powerful, `tch-rs` can be more complex than Python ML libraries, especially regarding tensor manipulation and device handling.
* **Model Saving/Loading:** Basic model saving/loading using `VarStore::save/load` is commented out. Implementing robust checkpointing might require more effort.
* **Direct Translation:** This is primarily a direct translation exercise. Optimizing for Rust's specific strengths (e.g., concurrency beyond simple parallelism in LibTorch) is not implemented.
* **Error Handling:** Error handling is minimal for simplicity (using `unwrap` or basic checks). Production code would require more robust error management (`Result`).
