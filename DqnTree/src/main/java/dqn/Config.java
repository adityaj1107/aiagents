package dqn;

public class Config {
    // Environment/Network parameters
    public final int stateSize = 4;
    public final int actionSize = 3;
    public final int hiddenDim = 128;

    // Replay Buffer parameters
    public final int bufferSize = 50000;
    public final int batchSize = 128;

    // Training parameters
    public final double gamma = 0.99;       // Discount factor
    public final double lr = 1e-4;          // Learning rate
    public final double tau = 1e-3;         // Soft update factor
    public final int updateEvery = 4;       // Steps between learning updates

    // Epsilon-greedy parameters
    public final double epsilonStart = 1.0;
    public final double epsilonEnd = 0.01;
    public final double epsilonDecay = 0.996;

    // Training loop parameters
    public final int numEpisodes = 50;
    public final int maxStepsPerEpisode = 50;

    // Other
    public final long randomSeed = System.currentTimeMillis(); // Or a fixed seed
}