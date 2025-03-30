package dqn;

import java.util.Arrays;
import java.util.Random;

public class StructuredTreeEnv {
    private final int stateSize;
    private final int actionSize;
    private final int maxSteps;
    private double[] state;
    private final Random random;
    private int stepsTaken;

    public StructuredTreeEnv(int stateSize, int actionSize, int maxSteps, long seed) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.maxSteps = maxSteps;
        this.random = new Random(seed);
        this.state = new double[stateSize];
        System.out.printf("Initialized StructuredTreeEnv (State Size: %d, Action Size: %d, Max Steps: %d)\n",
                stateSize, actionSize, maxSteps);
    }

    public double[] reset() {
        this.state = new double[stateSize];
        for (int i = 0; i < stateSize; i++) {
            this.state[i] = this.random.nextDouble(); // 0.0 to 1.0
        }
        // Specific initializations from Python
        this.state[1] = this.random.nextDouble() * 0.5; // Lower urgency
        this.state[2] = 0.0; // No interaction flag
        this.stepsTaken = 0;
        return Arrays.copyOf(this.state, this.stateSize); // Return copy
    }

    private double[] generateNextBaseState() {
        double[] nextStateBase = new double[stateSize];
        // Feature 0: Complexity decay
        nextStateBase[0] = Math.max(0.0, Math.min(1.0, this.state[0] * 0.9 + this.random.nextDouble() * 0.1));
        // Feature 1: Urgency random
        nextStateBase[1] = this.random.nextDouble();
        // Feature 2: Interaction Flag random
        nextStateBase[2] = (this.random.nextDouble() > 0.8) ? 1.0 : 0.0;
        // Feature 3+: Noise
        if (stateSize > 3) {
            for (int i = 3; i < stateSize; i++) {
                nextStateBase[i] = this.random.nextDouble() * 0.1;
            }
        }
        return nextStateBase;
    }

    // Returns StepResult containing next state, reward, done
    public StepResult step(int action) {
        if (action < 0 || action >= actionSize) {
            throw new IllegalArgumentException("Invalid action " + action + " for action_size " + actionSize);
        }

        this.stepsTaken++;
        double reward = -0.05; // Base cost per step

        // --- Hypothetical Reward Logic ---
        if (this.state[1] > 0.7 && action == 0) {
            reward += 0.8;
        } else if (this.state[2] > 0.5 && action == 1) {
            reward += 0.6;
        } else if (action == 2) {
            reward -= 0.1;
        } else {
            reward += this.random.nextDouble() * 0.1;
        }

        // --- State Transition ---
        double[] nextStateBase = generateNextBaseState();
        if (action == 0) {
            nextStateBase[1] *= 0.5;
        }
        if (action == 1) {
            nextStateBase[2] = 0.0;
        }
        nextStateBase[0] *= 0.95 + (double) action * 0.01;

        // Clip and update state
        for (int i = 0; i < stateSize; i++) {
            this.state[i] = Math.max(0.0, Math.min(1.0, nextStateBase[i]));
        }

        // --- Termination Condition ---
        boolean done = false;
        if (reward > 0.7) {
            done = true;
        } else if (this.stepsTaken >= this.maxSteps) {
            done = true;
        }

        return new StepResult(Arrays.copyOf(this.state, this.stateSize), reward, done);
    }

    // Inner class to return multiple values from step
    public static class StepResult {
        public final double[] nextState;
        public final double reward;
        public final boolean done;

        public StepResult(double[] nextState, double reward, boolean done) {
            this.nextState = nextState;
            this.reward = reward;
            this.done = done;
        }
    }

    // Optional close method if needed for resource cleanup
    public void close() {
        System.out.println("StructuredTree environment closed.");
    }
}