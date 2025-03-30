package dqn;

import java.util.*;

public class ReplayBuffer {
    private final Deque<Experience> memory; // Use Deque interface, LinkedList implementation
    private final int capacity;
    private final int batchSize;
    private final Random random;

    public ReplayBuffer(int capacity, int batchSize, long seed) {
        this.capacity = capacity;
        this.batchSize = batchSize;
        this.memory = new LinkedList<>();
        this.random = new Random(seed);
        // System.out.printf("Initialized ReplayBuffer with capacity=%d, batchSize=%d\n", capacity, batchSize);
    }

    public void add(Experience experience) {
        if (memory.size() >= capacity) {
            memory.removeFirst(); // Remove oldest
        }
        memory.addLast(experience); // Add newest
    }

    public List<Experience> sample() {
        if (memory.size() < batchSize) {
            return null; // Not enough samples
        }
        // Convert Deque to List for easier random sampling
        List<Experience> listSample = new ArrayList<>(memory);
        Collections.shuffle(listSample, random); // Shuffle the list
        // Return a sublist representing the batch
        return listSample.subList(0, batchSize);
    }

    public int size() {
        return memory.size();
    }
}