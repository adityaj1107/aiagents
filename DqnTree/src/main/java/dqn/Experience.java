package dqn;

// Simple class to hold one transition
public class Experience {
    public final double[] state;
    public final int action;
    public final double reward;
    public final double[] nextState;
    public final boolean done;

    public Experience(double[] state, int action, double reward, double[] nextState, boolean done) {
        // It's safer to store copies if the original arrays might be modified
        this.state = state.clone();
        this.action = action;
        this.reward = reward;
        this.nextState = nextState.clone();
        this.done = done;
    }
}