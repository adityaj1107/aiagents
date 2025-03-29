# Go DQN 

This project implements the **structure and environment interaction loop** of a Deep Q-Network (DQN) agent in Go (Golang). It simulates the `StructuredTree` environment from the corresponding Python example and includes the agent's logic for action selection (epsilon-greedy), experience replay, and the overall training flow.

**Critically Important Limitation:**

Go does **not** have a mature, standard machine learning library equivalent to Python's PyTorch or TensorFlow. Therefore, this implementation **DOES NOT PERFORM ACTUAL NEURAL NETWORK TRAINING OR LEARNING**.

* The `QNetwork` (neural network) is **not implemented**.
* The `learn` method in the agent is a **placeholder** and does not perform gradient descent or network updates.
* The "greedy" action selection path defaults to random actions because there is no network to query.
* There is no GPU acceleration.

This code serves primarily as an exercise in translating the *control flow*, *environment simulation*, and *agent interaction logic* into Go, highlighting the current limitations of Go's ecosystem for implementing complex deep reinforcement learning algorithms from scratch easily.

## Features Implemented

* **Environment Simulation (`StructuredTreeEnv`):** Accurately simulates the environment dynamics and rewards based on the Python example.
* **Agent Structure (`DQNAgent`):** Defines the agent's components (config, buffer, epsilon).
* **Experience Replay (`ReplayBuffer`):** Stores and samples experiences (using `container/list` and basic shuffling).
* **Epsilon-Greedy Logic:** Implements the exploration/exploitation strategy (exploitation path is placeholder).
* **Training Loop:** Mimics the episode and step structure of the Python DQN training process.

## Features NOT Implemented

* Neural Network Definition (MLP Q-Network)
* Neural Network Forward Pass (for greedy actions and learning)
* Automatic Differentiation / Backpropagation
* Gradient-Based Optimization
* Target Network Updates (Soft/Hard)
* Model Saving/Loading
* GPU Acceleration

## Dependencies

* Go Toolchain (>= 1.18 for generics, though not strictly used here, good practice)

## How to Run

1.  **Save:** Save the Go code as `main.go`.
2.  **Run from Terminal:**
    ```bash
    go run main-simulation-dqns.go
    ```

    The program will output initialization messages, indicate buffer pre-population, and then print the simulated training progress per episode. **Remember that no actual learning is occurring.**

## Code Structure

* `main.go`: Contains the entire Go code:
    * `Config`: Struct for hyperparameters.
    * `StructuredTreeEnv`: The environment simulation.
    * `Experience`: Struct for transitions.
    * `ReplayBuffer`: Manages experience storage and sampling.
    * `DQNAgent`: Holds agent state and methods (with NN/learning parts stubbed).
    * `main()`: Entry point, sets up simulation, runs the interaction loop.

## Configuration

Hyperparameters are defined in the `Config` struct and the `DefaultConfig()` function within `main.go`. Note that parameters related to the neural network (`HiddenDim`, `Lr`, `Tau`) are placeholders in this version.
