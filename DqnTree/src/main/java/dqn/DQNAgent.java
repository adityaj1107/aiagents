package dqn;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Random;

public class DQNAgent {
    private final Config config;
    private final MultiLayerNetwork qNetworkLocal;
    private final MultiLayerNetwork qNetworkTarget;
    private final ReplayBuffer replayBuffer;
    private double epsilon;
    private int timeStep;
    private final Random random;

    public DQNAgent(Config config) {
        this.config = config;
        this.random = new Random(config.randomSeed);
        this.epsilon = config.epsilonStart;
        this.timeStep = 0;

        // Build networks
        this.qNetworkLocal = QNetworkBuilder.buildQNetwork(config.stateSize, config.actionSize, config.hiddenDim, config.lr, config.randomSeed + 1);
        this.qNetworkTarget = QNetworkBuilder.buildQNetwork(config.stateSize, config.actionSize, config.hiddenDim, config.lr, config.randomSeed + 2);

        // Initialize target network with local network's weights
        this.qNetworkTarget.setParams(this.qNetworkLocal.params());

        // Initialize replay buffer
        this.replayBuffer = new ReplayBuffer(config.bufferSize, config.batchSize, config.randomSeed + 3);

        System.out.println("Initialized DQNAgent (DL4J based).");
        System.out.printf("  State Size: %d\n", config.stateSize);
        System.out.printf("  Action Size: %d\n", config.actionSize);
    }

    // Convert double[] state to INDArray for network input
    private INDArray preprocessState(double[] state) {
        // Create a 1xN row vector
        return Nd4j.create(state, new int[]{1, state.length});
    }

    public int selectAction(double[] state, boolean useEpsilon) {
        // Epsilon-greedy action selection
        if (useEpsilon && random.nextDouble() < this.epsilon) {
            // Exploration
            return random.nextInt(config.actionSize);
        } else {
            // Exploitation
            INDArray stateTensor = preprocessState(state);
            qNetworkLocal.setInput(stateTensor); // Set input (though output method also takes input)
            INDArray actionValues = qNetworkLocal.output(stateTensor, false); // Get Q-values (inference mode)

            // Find the action with the highest Q-value using Nd4j argmax
            // argMax(1) finds the index of the max value along dimension 1 (columns)
            return Nd4j.argMax(actionValues, 1).getInt(0);
        }
    }

    public void storeExperience(double[] state, int action, double reward, double[] nextState, boolean done) {
        Experience exp = new Experience(state, action, reward, nextState, done);
        this.replayBuffer.add(exp);

        // Learn every 'updateEvery' steps if buffer is ready
        this.timeStep = (this.timeStep + 1) % config.updateEvery;
        if (this.timeStep == 0 && this.replayBuffer.size() >= config.batchSize) {
            List<Experience> experiences = this.replayBuffer.sample();
            if (experiences != null) {
                learn(experiences);
            }
        }
    }

    private void learn(List<Experience> experiences) {
        // 1. Prepare batched INDArrays from experiences
        double[][] statesBatch = new double[config.batchSize][config.stateSize];
        int[] actionsBatch = new int[config.batchSize];
        double[] rewardsBatch = new double[config.batchSize];
        double[][] nextStatesBatch = new double[config.batchSize][config.stateSize];
        double[] donesBatch = new double[config.batchSize];

        for (int i = 0; i < config.batchSize; i++) {
            Experience exp = experiences.get(i);
            statesBatch[i] = exp.state;
            actionsBatch[i] = exp.action;
            rewardsBatch[i] = exp.reward;
            nextStatesBatch[i] = exp.nextState;
            donesBatch[i] = exp.done ? 1.0 : 0.0;
        }

        INDArray statesTensor = Nd4j.create(statesBatch);
        // actionsTensor not directly used in this target calculation method
        INDArray rewardsTensor = Nd4j.create(rewardsBatch, new int[]{config.batchSize, 1});
        INDArray nextStatesTensor = Nd4j.create(nextStatesBatch);
        INDArray donesTensor = Nd4j.create(donesBatch, new int[]{config.batchSize, 1});

        // 2. Calculate Target Q-values: Q_targets = r + gamma * max_a' Q_target(s', a') * (1 - done)
        INDArray qTargetNext = qNetworkTarget.output(nextStatesTensor, false); // Inference mode
        INDArray maxQTargetNext = Nd4j.max(qTargetNext, 1); // Max Q value along action dimension
        maxQTargetNext = maxQTargetNext.reshape(config.batchSize, 1); // Ensure column vector

        // target = reward + gamma * maxQ * (1 - done)
        INDArray qTargets = rewardsTensor.add(maxQTargetNext.mul(config.gamma).mul(donesTensor.rsub(1.0))); // rsub(1.0) is 1.0 - dones

        // 3. Calculate Expected Q-values and prepare targets for fitting
        // Get current Q-values from local network for the states
        INDArray qExpectedAll = qNetworkLocal.output(statesTensor, false); // Use current predictions as base
        // Create the target tensor for fitting: Use TD target for the action taken,
        // and keep the network's original prediction for other actions.
        // This way, only the error for the chosen action's Q-value is backpropagated via MSE.
        INDArray qTargetsForFit = qExpectedAll.dup(); // Duplicate existing predictions

        for (int i = 0; i < config.batchSize; i++) {
            int action = actionsBatch[i];
            double targetValue = qTargets.getDouble(i, 0);
            qTargetsForFit.putScalar(i, action, targetValue); // Update only the Q-value for the action taken
        }

        // 4. Train the local network: Fit statesTensor to qTargetsForFit
        qNetworkLocal.fit(statesTensor, qTargetsForFit); // Performs forward, backward pass and updates

        // 5. Soft update target network
        softUpdateTargetNetwork();
    }

    private void softUpdateTargetNetwork() {
        INDArray localParams = qNetworkLocal.params();
        INDArray targetParams = qNetworkTarget.params();

        // target = tau * local + (1 - tau) * target
        INDArray updatedTargetParams = localParams.mul(config.tau).add(targetParams.mul(1.0 - config.tau));
        qNetworkTarget.setParams(updatedTargetParams);
    }

    public void decayEpsilon() {
        this.epsilon = Math.max(config.epsilonEnd, config.epsilonDecay * this.epsilon);
    }

    // --- Model Saving/Loading (using DL4J utilities) ---
    public void saveModel(String filePath) throws IOException {
        File file = new File(filePath);
        boolean saveUpdater = true; // Save optimizer state as well
        ModelSerializer.writeModel(qNetworkLocal, file, saveUpdater);
        System.out.println("Model saved to " + filePath);
    }

    public void loadModel(String filePath) throws IOException {
        File file = new File(filePath);
        qNetworkLocal = ModelSerializer.restoreMultiLayerNetwork(file);
        // Important: Update target network after loading
        qNetworkTarget.setParams(qNetworkLocal.params());
        System.out.println("Model loaded from " + filePath);
        // Note: Optimizer state is loaded if saved, associated with the loaded network.
    }

    public double getEpsilon() {
        return epsilon;
    }
}