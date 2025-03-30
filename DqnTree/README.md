# Java DQN Agent for Structured Tree Environment (using DL4J)

This project implements a Deep Q-Network (DQN) agent in Java using the DeepLearning4J (DL4J) library. The agent learns to interact with a simulated `StructuredTree` environment, aiming to choose actions (node prioritizations) that maximize cumulative rewards based on abstract state features like complexity and urgency.

This code is a translation of the Python implementation found in `dqn_tree_agent.py`.

## Features

* **DQN Agent:** Implements the core DQN algorithm with experience replay and target networks.
* **MLP Q-Network:** Uses a Multi-Layer Perceptron defined and trained using DeepLearning4J.
* **Experience Replay:** Employs a replay buffer (`LinkedList` acting as `Deque`) to store and sample transitions.
* **Epsilon-Greedy Exploration:** Balances exploration (random actions) and exploitation (greedy actions based on Q-values) using a decaying epsilon.
* **Target Network:** Utilizes a separate target network, updated via soft updates (`tau`), to stabilize Q-learning.
* **DL4J Integration:** Leverages DL4J for network configuration, training (`fit`), optimization (Adam), and numerical computation (ND4J).
* **Simulated Environment (`StructuredTreeEnv`):** Includes the Java version of the `StructuredTree` simulation.
* **Model Saving/Loading:** Includes methods for saving and loading the trained DL4J network.

## Dependencies

* **Java Development Kit (JDK):** Version 8 or 11+ recommended.
* **Maven or Gradle:** A build tool to manage dependencies.
* **DeepLearning4J:**
    * `deeplearning4j-core`: Core DL4J library.
    * `nd4j-native-platform` (or similar backend): ND4J backend for CPU operations. For GPU support, use an appropriate `nd4j-cuda-X.Y-platform` dependency and ensure you have a compatible NVIDIA driver and CUDA toolkit installed.
* **Logging Framework:** SLF4J API and a backend like Logback (DL4J uses SLF4J).

*(See the example `pom.xml` snippet in the code generation response for Maven dependency setup.)*

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd java-dqn-structured-tree # Or your chosen directory name
    ```
2.  **Configure Build Tool:** Ensure your `pom.xml` (for Maven) or `build.gradle` (for Gradle) includes the necessary DL4J dependencies as shown above. Select the correct ND4J backend (`nd4j-native-platform` for CPU, or a CUDA version for GPU).
3.  **Build the Project:**
    * **Maven:** `mvn clean package`
    * **Gradle:** `gradle clean build`

## How to Run

Execute the compiled code using your build tool:

* **Maven:**
    ```bash
    mvn exec:java -Dexec.mainClass="dqn.Main"
    ```
* **Gradle:**
    ```bash
    gradle run --args='dqn.Main'
    ```
  *(Adjust the main class path `dqn.Main` if your package structure differs.)*

The program will output initialization messages, buffer pre-population status, and then print training progress per episode, showing rewards and the decaying epsilon value.

## Code Structure

The code is organized into several Java classes, typically within a package (e.g., `dqn`):

* `Config.java`: Holds all hyperparameters and configuration constants.
* `Experience.java`: A simple data class representing a single (s, a, r, s', done) transition.
* `StructuredTreeEnv.java`: Simulates the environment dynamics, rewards, and state transitions.
* `ReplayBuffer.java`: Manages the storage and random sampling of experiences.
* `QNetworkBuilder.java`: Helper class to configure and build the DL4J `MultiLayerNetwork`.
* `DQNAgent.java`: The core agent class containing the Q-networks (local and target), replay buffer, learning logic (`learn` method using DL4J), action selection (`selectAction`), and epsilon decay.
* `Main.java`: The main entry point for the application, responsible for setting up the environment and agent, and running the training loop.

## Configuration

All major hyperparameters (learning rate, network dimensions, buffer size, epsilon parameters, episode count, etc.) are defined in the `Config.java` class. Modify these values to experiment with different training setups.