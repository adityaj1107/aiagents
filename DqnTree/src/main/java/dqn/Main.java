package dqn;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.LinkedList; // For scores window deque
import java.util.Deque;

public class Main {

    public static void main(String[] args) {
        System.out.println("==============================================");
        System.out.println(" Starting DQN Agent Training (Java / DL4J) ");
        System.out.println("==============================================");

        Config config = new Config();

        // Print timestamp and context
        SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        System.out.printf("Timestamp: %s\n", formatter.format(new Date()));
        System.out.println("Location Context: Bengaluru, Karnataka, India");

        // Initialize Environment and Agent
        StructuredTreeEnv env = new StructuredTreeEnv(config.stateSize, config.actionSize, config.maxStepsPerEpisode, config.randomSeed + 4);
        DQNAgent agent = new DQNAgent(config); // Config contains seed

        // --- Pre-populate Replay Buffer ---
        System.out.println("\n--- Pre-populating Replay Buffer ---");
        int prePopulateSteps = config.batchSize * 5;
        double[] state = env.reset();
        Random bufferRand = new Random(config.randomSeed + 5); // Separate RNG if needed

        for (int i = 0; i < prePopulateSteps; i++) {
            int action = bufferRand.nextInt(config.actionSize); // Random actions
            StructuredTreeEnv.StepResult result = env.step(action);
            agent.replayBuffer.add(new Experience(state, action, result.reward, result.nextState, result.done));
            state = result.nextState;
            if (result.done) {
                state = env.reset();
            }
        }
        System.out.printf("Replay buffer pre-populated with %d experiences.\n", agent.replayBuffer.size());

        // --- Training Loop ---
        System.out.println("\n--- Starting Training Loop ---");
        long startTime = System.currentTimeMillis();
        Deque<Double> scoresWindow = new LinkedList<>(); // Use Deque for rolling window
        final int scoreWindowSize = 10; // Match Python example

        for (int iEpisode = 1; iEpisode <= config.numEpisodes; iEpisode++) {
            state = env.reset();
            double episodeReward = 0.0;

            for (int t = 0; t < config.maxStepsPerEpisode; t++) {
                // 1. Agent selects action
                int action = agent.selectAction(state, true); // Use epsilon

                // 2. Environment executes action
                StructuredTreeEnv.StepResult result = env.step(action);

                // 3. Agent stores experience (and potentially learns)
                agent.storeExperience(state, action, result.reward, result.nextState, result.done);

                // Update state and reward
                state = result.nextState;
                episodeReward += result.reward;

                if (result.done) {
                    break; // End episode early
                }
            } // End step loop

            // --- End of Episode ---
            if (scoresWindow.size() >= scoreWindowSize) {
                scoresWindow.removeFirst(); // Remove oldest score
            }
            scoresWindow.addLast(episodeReward); // Add current score
            agent.decayEpsilon(); // Decay exploration rate

            // Calculate average score
            double avgScore = 0.0;
            if (!scoresWindow.isEmpty()) {
                double sum = 0;
                for (double score : scoresWindow) {
                    sum += score;
                }
                avgScore = sum / scoresWindow.size();
            }

            // Print progress (mimic Python output)
            System.out.printf("\rEpisode %d\tTotal Reward: %.2f\tAverage Reward (last %d): %.2f\tEpsilon: %.3f   ",
                    iEpisode,
                    episodeReward,
                    scoresWindow.size(),
                    avgScore,
                    agent.getEpsilon());
            if (iEpisode % 10 == 0) { // Newline every 10 episodes
                System.out.println();
            }

        } // End episode loop

        long endTime = System.currentTimeMillis();
        System.out.println("\n--- Training Finished ---");
        System.out.printf("Total training time: %.2f seconds\n", (endTime - startTime) / 1000.0);

        // Optional: Save the final model
        /*
        try {
            agent.saveModel("dqn_model_final.bin");
        } catch (IOException e) {
            System.err.println("Error saving model: " + e.getMessage());
            e.printStackTrace();
        }
        */

        env.close();

        System.out.println("\n==============================================");
        System.out.println(" DQN Agent Training (Java/DL4J) Finished ");
        System.out.println("==============================================");
    }
}