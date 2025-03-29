// main.go
package main

import (
	"container/list"
	"fmt"
	"math"
	"math/rand"
	"time"
	"chrono"
)

const (
	stateSize  = 4
	actionSize = 4
)

type Config struct {
	HiddenDim           int
	BufferSize          int
	BatchSize           int
	Gamma               float64
	Lr                  float64
	Tau                 float64
	UpdateEvery         int
	EpsilonStart        float64
	EpsilonEnd          float64
	EpsilonDecay        float64
	NumEpisodes         int
	MaxStepsPerEpisode  int
}

func DefaultConfig() Config {
	return Config{
		HiddenDim:           128,
		BufferSize:          50000,
		BatchSize:           128,
		Gamma:               0.99,
		Lr:                  1e-4,
		Tau:                 1e-3,
		UpdateEvery:         4,
		EpsilonStart:        1.0,
		EpsilonEnd:          0.01,
		EpsilonDecay:        0.996,
		NumEpisodes:         50,
		MaxStepsPerEpisode:  50,
	}
}

type StructuredTreeEnv struct {
	state     []float64 // Use float64 for standard Go math
	rng       *rand.Rand
	stepsTaken int
	maxSteps   int
}

func NewStructuredTreeEnv(maxSteps int, seed int64) *StructuredTreeEnv {
	fmt.Printf("Initializing StructuredTreeEnv (State Size: %d, Action Size: %d, Max Steps: %d)\n",
		stateSize, actionSize, maxSteps)
	// Create a new random source with the given seed
	source := rand.NewSource(seed)
	// Create a new random number generator from the source
	return &StructuredTreeEnv{
		state:     make([]float64, stateSize),
		rng:       rand.New(source),
		stepsTaken: 0,
		maxSteps:   maxSteps,
	}
}

func (env *StructuredTreeEnv) Reset() []float64 {
	env.state = make([]float64, stateSize)
	for i := 0; i < stateSize; i++ {
		env.state[i] = env.rng.Float64() // Random float between 0.0 and 1.0
	}
	// Specific initializations from Python
	env.state[1] = env.rng.Float64() * 0.5 // Lower urgency
	env.state[2] = 0.0                    // No interaction flag
	env.stepsTaken = 0
	// Return a copy to prevent external modification of internal state
	stateCopy := make([]float64, stateSize)
	copy(stateCopy, env.state)
	return stateCopy
}

// generateNextBaseState calculates the next potential state based on internal dynamics.
func (env *StructuredTreeEnv) generateNextBaseState() []float64 {
	nextStateBase := make([]float64, stateSize)
	// Feature 0: Complexity decay with some randomness
	nextStateBase[0] = math.Max(0.0, math.Min(1.0, env.state[0]*0.9+env.rng.Float64()*0.1))
	// Feature 1: Urgency changes randomly
	nextStateBase[1] = env.rng.Float64()
	// Feature 2: Interaction Flag appears randomly
	if env.rng.Float64() > 0.8 {
		nextStateBase[2] = 1.0
	} else {
		nextStateBase[2] = 0.0
	}

	if stateSize > 3 {
		for i := 3; i < stateSize; i++ {
			nextStateBase[i] = env.rng.Float64() * 0.1
		}
	}
	return nextStateBase
}

func (env *StructuredTreeEnv) Step(action int) ([]float64, float64, bool) {
	// Validate action input
	if action < 0 || action >= actionSize {
		// Handle invalid action - here we panic, could also return an error or default action
		panic(fmt.Sprintf("Invalid action %d for action_size %d", action, actionSize))
	}

	env.stepsTaken++
	reward := -0.05 // Base cost per step encourages efficiency

	// --- Hypothetical Reward Logic based on Python code ---
	if env.state[1] > 0.7 && action == 0 { // Reward action 0 if urgency is high
		reward += 0.8
	} else if env.state[2] > 0.5 && action == 1 { // Reward action 1 if interaction flag is set
		reward += 0.6
	} else if action == 2 { // Penalize action 2
		reward -= 0.1
	} else { // Small random positive reward otherwise
		reward += env.rng.Float64() * 0.1
	}

	// --- State Transition Logic ---
	nextStateBase := env.generateNextBaseState()
	// Apply action effects
	if action == 0 { // Action 0 reduces urgency
		nextStateBase[1] *= 0.5
	}
	if action == 1 { // Action 1 resets interaction flag
		nextStateBase[2] = 0.0
	}
	// Action influences complexity slightly
	nextStateBase[0] *= 0.95 + float64(action)*0.01

	// Clip state values to [0, 1] and update internal state
	for i := 0; i < stateSize; i++ {
		env.state[i] = math.Max(0.0, math.Min(1.0, nextStateBase[i]))
	}

	// --- Termination Condition Logic ---
	done := false
	if reward > 0.7 { // End episode if high reward is achieved
		done = true
	} else if env.stepsTaken >= env.maxSteps { // End episode if max steps are reached
		done = true
	}

	// Return a copy of the new state
	stateCopy := make([]float64, stateSize)
	copy(stateCopy, env.state)
	return stateCopy, reward, done
}

// --- Experience Struct ---
// Represents a single transition (s, a, r, s', done)
type Experience struct {
	State     []float64
	Action    int
	Reward    float64
	NextState []float64
	Done      bool
}

// --- Replay Buffer ---
// Stores experiences and allows sampling batches.
type ReplayBuffer struct {
	memory   *list.List // Using standard library list as a deque
	capacity int
	batchSize int
	rng      *rand.Rand
}

// NewReplayBuffer creates a new replay buffer.
func NewReplayBuffer(capacity int, batchSize int, seed int64) *ReplayBuffer {
	source := rand.NewSource(seed)
	return &ReplayBuffer{
		memory:   list.New(),
		capacity: capacity,
		batchSize: batchSize,
		rng:      rand.New(source),
	}
}

// Add adds a new experience to the buffer, removing the oldest if capacity is reached.
func (rb *ReplayBuffer) Add(state []float64, action int, reward float64, nextState []float64, done bool) {
	if rb.memory.Len() >= rb.capacity {
		// Remove the oldest element (front of the list)
		rb.memory.Remove(rb.memory.Front())
	}
	// Create the experience struct
	exp := Experience{
		State:     state, // Assumes caller provides copies if needed
		Action:    action,
		Reward:    reward,
		NextState: nextState, // Assumes caller provides copies if needed
		Done:      done,
	}
	// Add the new experience to the back of the list
	rb.memory.PushBack(exp)
}

func (rb *ReplayBuffer) Sample() []Experience {
	if rb.memory.Len() < rb.batchSize {
		return nil // Not enough samples for a batch
	}

    // 1. Copy elements from the list to a temporary slice for shuffling.
	tempSlice := make([]Experience, 0, rb.memory.Len())
	for e := rb.memory.Front(); e != nil; e = e.Next() {
		// Type assert the value from the list element
		if exp, ok := e.Value.(Experience); ok {
			tempSlice = append(tempSlice, exp)
        } else {
            // Handle unexpected type if necessary, though list should only contain Experiences
            fmt.Println("Warning: Unexpected type found in replay buffer")
        }
	}

    // 2. Shuffle the temporary slice randomly.
    rb.rng.Shuffle(len(tempSlice), func(i, j int) {
        tempSlice[i], tempSlice[j] = tempSlice[j], tempSlice[i]
    })

    // 3. Return the first `batchSize` elements from the shuffled slice.
	return tempSlice[:rb.batchSize]
}

// Len returns the current number of experiences in the buffer.
func (rb *ReplayBuffer) Len() int {
	return rb.memory.Len()
}

// --- DQN Agent (Structure Only - No NN/Learning Implementation) ---
type DQNAgent struct {
	config      Config
	buffer      *ReplayBuffer
	epsilon     float64
	timeStep    int
	rng         *rand.Rand
}

// NewDQNAgent creates a new agent instance.
func NewDQNAgent(config Config, seed int64) *DQNAgent {
	source := rand.NewSource(seed)
	fmt.Println("Initialized DQNAgent (Structure Only - NO LEARNING CAPABILITIES)")
	fmt.Printf("  State Size: %d\n", stateSize)
	fmt.Printf("  Action Size: %d\n", actionSize)

	return &DQNAgent{
		config:  config,
		// Use a different seed for the buffer's RNG
		buffer:  NewReplayBuffer(config.BufferSize, config.BatchSize, seed+1),
		epsilon: config.EpsilonStart,
		timeStep: 0,
		rng:     rand.New(source),
		// Neural network components are NOT initialized here
	}
}

// SelectAction chooses an action based on the state using epsilon-greedy.
// *** NOTE: The greedy (exploitation) path defaults to random action due to lack of NN. ***
func (a *DQNAgent) SelectAction(state []float64, useEpsilon bool) int {
	// Epsilon-greedy exploration
	if useEpsilon && a.rng.Float64() < a.epsilon {
		// Explore: choose a random action
		return a.rng.Intn(actionSize)
	} else {
		// Exploit: *** NO NETWORK TO QUERY ***
		// In a real implementation, this would involve:
		// 1. Preprocessing state into a tensor format for the network.
		// 2. Performing a forward pass through the local Q-network (qLocal).
		// 3. Finding the action index corresponding to the highest Q-value (argmax).
		// As a placeholder, we return a random action.
		return a.rng.Intn(actionSize)
	}
}

// StoreExperience adds a transition to the replay buffer and potentially triggers the placeholder learn step.
func (a *DQNAgent) StoreExperience(state []float64, action int, reward float64, nextState []float64, done bool) {
	// Create copies of the state slices before storing, as the originals might be modified later.
	stateCopy := make([]float64, len(state))
	copy(stateCopy, state)
	nextStateCopy := make([]float64, len(nextState))
	copy(nextStateCopy, nextState)

	// Add the experience to the replay buffer.
	a.buffer.Add(stateCopy, action, reward, nextStateCopy, done)

	// Check if it's time to trigger the learning step based on `UpdateEvery` config.
	a.timeStep = (a.timeStep + 1) % a.config.UpdateEvery
	if a.timeStep == 0 && a.buffer.Len() >= a.config.BatchSize {
		// Sample a batch of experiences from the buffer.
		experiences := a.buffer.Sample()
		if experiences != nil {
			// Call the placeholder learn function.
			a.learn(experiences)
		}
	}
}

// DecayEpsilon reduces the exploration rate (epsilon) according to the decay factor.
func (a *DQNAgent) DecayEpsilon() {
	a.epsilon = math.Max(a.config.EpsilonEnd, a.config.EpsilonDecay*a.epsilon)
}


// --- Main Execution Function ---
func main() {
	fmt.Println("==============================================")
	fmt.Println(" Starting DQN Agent Training (Go - Structure Only) ")
	fmt.Println(" **** NO ACTUAL NEURAL NETWORK LEARNING **** ")
	fmt.Println("==============================================")

	seed := time.Now().UnixNano()


	globalRand := rand.New(rand.NewSource(seed)) // Used only for initial buffer population here

	// Print timestamp and context
	fmt.Printf("Timestamp: %s\n", time.Now().Format("2006-01-02 15:04:05"))
	fmt.Println("Location Context: Bengaluru, Karnataka, India") // From Python code

	// Load configuration
	config := DefaultConfig()

	// Initialize environment and agent
	env := NewStructuredTreeEnv(config.MaxStepsPerEpisode, seed+1)
	agent := NewDQNAgent(config, seed+2)


	// --- Pre-populate Replay Buffer ---
	fmt.Println("\n--- Pre-populating Replay Buffer ---")
	prePopulateSteps := config.BatchSize * 5 // Match Python logic: ensure enough samples for initial learning
	state := env.Reset()
	for i := 0; i < prePopulateSteps; i++ {
		// Select a random action using the global RNG for pre-population
		action := globalRand.Intn(actionSize)
		// Perform action in the environment
		nextState, reward, done := env.Step(action)

		// Add experience to the agent's replay buffer
		// Make copies of state slices before adding
		stateCopy := make([]float64, len(state))
		copy(stateCopy, state)
		nextStateCopy := make([]float64, len(nextState))
		copy(nextStateCopy, nextState)
		agent.buffer.Add(stateCopy, action, reward, nextStateCopy, done)

		// Update the current state
		state = nextState
		// Reset environment if the episode ended during pre-population
		if done {
			state = env.Reset()
		}
	}
	fmt.Printf("Replay buffer pre-populated with %d experiences.\n", agent.buffer.Len())


	// --- Training Loop ---
	fmt.Println("\n--- Starting Training Loop ---")
	startTime := time.Now()
	// Use a slice acting as a rolling window for recent scores
	scoresWindow := make([]float64, 0, 10) // Capacity 10 like Python example
	// totalScores := make([]float64, 0, config.NumEpisodes) // Optional: store all episode scores

	// Loop through the specified number of episodes
	for iEpisode := 1; iEpisode <= config.NumEpisodes; iEpisode++ {
		// Reset the environment at the start of each episode
		state := env.Reset()
		episodeReward := 0.0

		// Loop through steps within an episode, up to the maximum allowed
		for t := 0; t < config.MaxStepsPerEpisode; t++ {
			// 1. Agent selects an action based on the current state and exploration strategy
			action := agent.SelectAction(state, true) // useEpsilon=true during training

			// 2. Environment executes the chosen action
			nextState, reward, done := env.Step(action)

			// 3. Agent stores the resulting experience (s, a, r, s', done)
			//    This also potentially triggers the (placeholder) learning step inside.
			agent.StoreExperience(state, action, reward, nextState, done)

			// Update the current state for the next iteration
			state = nextState // `Step` already returns a copy
			// Accumulate the reward for the episode
			episodeReward += reward

			// If the episode terminated (e.g., tree died, goal reached), end the inner loop
			if done {
				break
			}
		} // End step loop

		// Update the rolling window of recent scores
		if len(scoresWindow) >= 10 { // If window is full
			scoresWindow = scoresWindow[1:] // Remove the oldest score (inefficient for large windows)
		}
		scoresWindow = append(scoresWindow, episodeReward) // Add the latest score

		// Decay the exploration rate (epsilon) after each episode
		agent.DecayEpsilon()

		// Calculate the average score over the window
		avgScore := 0.0
		if len(scoresWindow) > 0 {
			sum := 0.0
			for _, score := range scoresWindow {
				sum += score
			}
			avgScore = sum / float64(len(scoresWindow))
		}

		fmt.Printf("\rEpisode %d\tTotal Reward: %.2f\tAverage Reward (last %d): %.2f\tEpsilon: %.3f   ",
			iEpisode,
			episodeReward,
			len(scoresWindow),
			avg_score,
			agent.epsilon,
		)

	}

	// Calculate and print total training time
	endTime := time.Now()
	fmt.Println("\n--- Training Finished ---")
	fmt.Printf("Total training time: %v\n", endTime.Sub(startTime))

	fmt.Println("\n==============================================")
	fmt.Println(" DQN Agent Training Simulation (Go) Finished ")
	fmt.Println("==============================================")
}

