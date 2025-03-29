// src/main.rs
use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Device, Kind, Tensor};
use rand::prelude::*;
use std::{collections::VecDeque, time::Instant};
use chrono; // For timestamp

// --- Constants based on Python Config ---
const STATE_SIZE: i64 = 4;
const ACTION_SIZE: i64 = 3;

// --- Configuration Struct ---
#[derive(Debug, Clone)]
struct Config {
    hidden_dim: i64,
    buffer_size: usize,
    batch_size: usize,
    gamma: f64,
    lr: f64,
    tau: f64,
    update_every: u64,
    epsilon_start: f64,
    epsilon_end: f64,
    epsilon_decay: f64,
    num_episodes: i32,
    max_steps_per_episode: i32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            hidden_dim: 128,
            buffer_size: 50_000,
            batch_size: 128,
            gamma: 0.99,
            lr: 1e-4,
            tau: 1e-3,
            update_every: 4,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.996,
            num_episodes: 50, // Match Python example
            max_steps_per_episode: 50, // Match Python env.max_steps
        }
    }
}

// --- Environment Simulation (StructuredTree) ---
#[derive(Debug)]
struct StructuredTreeEnv {
    state: Vec<f32>,
    rng: ThreadRng,
    steps_taken: i32,
    max_steps: i32,
}

impl StructuredTreeEnv {
    fn new(max_steps: i32) -> Self {
        println!(
            "Initializing StructuredTreeEnv (State Size: {}, Action Size: {}, Max Steps: {})",
            STATE_SIZE, ACTION_SIZE, max_steps
        );
        StructuredTreeEnv {
            state: vec![0.0; STATE_SIZE as usize],
            rng: rand::thread_rng(),
            steps_taken: 0,
            max_steps,
        }
    }

    fn reset(&mut self) -> Vec<f32> {
        self.state = (0..STATE_SIZE).map(|_| self.rng.gen::<f32>()).collect();
        // Specific initializations from Python
        self.state[1] = self.rng.gen::<f32>() * 0.5; // Lower urgency
        self.state[2] = 0.0; // No interaction flag
        self.steps_taken = 0;
        self.state.clone()
    }

    fn generate_next_base_state(&mut self) -> Vec<f32> {
        let mut next_state_base = vec![0.0; STATE_SIZE as usize];
        // Feature 0: Complexity decay
        next_state_base[0] = (self.state[0] * 0.9 + self.rng.gen::<f32>() * 0.1).clamp(0.0, 1.0);
        // Feature 1: Urgency random
        next_state_base[1] = self.rng.gen::<f32>();
        // Feature 2: Interaction Flag random
        next_state_base[2] = if self.rng.gen::<f32>() > 0.8 { 1.0 } else { 0.0 };
        // Feature 3+: Noise
        if STATE_SIZE > 3 {
            for i in 3..STATE_SIZE as usize {
                next_state_base[i] = self.rng.gen::<f32>() * 0.1;
            }
        }
        next_state_base
    }

    // Returns (next_state, reward, done)
    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) {
        if action >= ACTION_SIZE as usize {
            // In Python this raised ValueError, here we might panic or handle differently
            panic!("Invalid action {} for action_size {}", action, ACTION_SIZE);
        }

        self.steps_taken += 1;
        let mut reward = -0.05; // Base cost per step

        // --- Hypothetical Reward Logic ---
        if self.state[1] > 0.7 && action == 0 { // High urgency -> action 0
            reward += 0.8;
        } else if self.state[2] > 0.5 && action == 1 { // Interaction flag -> action 1
            reward += 0.6;
        } else if action == 2 { // Penalty for action 2
            reward -= 0.1;
        } else { // Small random reward otherwise
            reward += self.rng.gen::<f32>() * 0.1;
        }

        // --- State Transition ---
        let mut next_state_base = self.generate_next_base_state();
        // Action effects
        if action == 0 { // Reduce urgency
            next_state_base[1] *= 0.5;
        }
        if action == 1 { // Reset interaction flag
            next_state_base[2] = 0.0;
        }
        // Action influences complexity slightly
        next_state_base[0] *= 0.95 + (action as f32) * 0.01;

        // Clip and update state
        self.state = next_state_base.iter().map(|&v| v.clamp(0.0, 1.0)).collect();

        // --- Termination Condition ---
        let mut done = false;
        if reward > 0.7 { // End on high reward
            done = true;
        } else if self.steps_taken >= self.max_steps { // End if max steps reached
            done = true;
        }

        (self.state.clone(), reward, done)
    }
}

// --- Experience Struct ---
#[derive(Debug, Clone)]
struct Experience {
    state: Vec<f32>,
    action: usize,
    reward: f32,
    next_state: Vec<f32>,
    done: bool,
}

// --- Replay Buffer ---
#[derive(Debug)]
struct ReplayBuffer {
    memory: VecDeque<Experience>,
    capacity: usize,
    batch_size: usize,
    rng: ThreadRng,
    device: Device,
}

impl ReplayBuffer {
    fn new(capacity: usize, batch_size: usize, device: Device) -> Self {
        // println!("Initializing ReplayBuffer with size={}, batch_size={} on {:?}", capacity, batch_size, device);
        ReplayBuffer {
            memory: VecDeque::with_capacity(capacity),
            capacity,
            batch_size,
            rng: rand::thread_rng(),
            device,
        }
    }

    fn add(&mut self, exp: Experience) {
        if self.memory.len() == self.capacity {
            self.memory.pop_front();
        }
        self.memory.push_back(exp);
    }

    fn sample(&mut self) -> Option<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        if self.memory.len() < self.batch_size {
            return None;
        }
        let samples: Vec<_> = self.memory
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .choose_multiple(&mut self.rng, self.batch_size)
            .cloned()
            .collect();

        // Batch data for tensor conversion
        let states: Vec<f32> = samples.iter().flat_map(|e| e.state.clone()).collect();
        let actions: Vec<i64> = samples.iter().map(|e| e.action as i64).collect();
        let rewards: Vec<f32> = samples.iter().map(|e| e.reward).collect();
        let next_states: Vec<f32> = samples.iter().flat_map(|e| e.next_state.clone()).collect();
        let dones: Vec<f32> = samples.iter().map(|e| if e.done { 1.0 } else { 0.0 }).collect();

        // Convert to tensors
        let states_t = Tensor::of_slice(&states)
            .view([-1, STATE_SIZE])
            .to_kind(Kind::Float)
            .to(self.device);
        let actions_t = Tensor::of_slice(&actions)
            .view([-1, 1])
            .to_kind(Kind::Int64)
            .to(self.device);
        let rewards_t = Tensor::of_slice(&rewards).view([-1, 1]).to_kind(Kind::Float).to(self.device);
        let next_states_t = Tensor::of_slice(&next_states)
            .view([-1, STATE_SIZE])
            .to_kind(Kind::Float)
            .to(self.device);
        let dones_t = Tensor::of_slice(&dones).view([-1, 1]).to_kind(Kind::Float).to(self.device);

        Some((states_t, actions_t, rewards_t, next_states_t, dones_t))
    }

    fn len(&self) -> usize {
        self.memory.len()
    }
}

// --- Neural Network Definition (MLP using tch-rs) ---
#[derive(Debug)]
struct QNetwork {
    fc1: nn::Linear,
    fc2: nn::Linear,
    fc3: nn::Linear,
}

impl QNetwork {
    fn new(vs: &nn::Path, state_size: i64, action_size: i64, hidden_dim: i64) -> Self {
        let fc1 = nn::linear(vs / "fc1", state_size, hidden_dim, Default::default());
        let fc2 = nn::linear(vs / "fc2", hidden_dim, hidden_dim, Default::default());
        let fc3 = nn::linear(vs / "fc3", hidden_dim, action_size, Default::default());
        QNetwork { fc1, fc2, fc3 }
    }
}

impl nn::ModuleT for QNetwork {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        xs.apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
            .relu()
            .apply(&self.fc3)
    }
}

// --- DQN Agent (Renamed from Python's 'Tree' class) ---
struct DQNAgent {
    vs: nn::VarStore,
    q_local: QNetwork,
    q_target: QNetwork,
    optimizer: nn::Optimizer<nn::Dhruva>,
    buffer: ReplayBuffer,
    config: Config,
    epsilon: f64,
    time_step: u64,
    device: Device,
    rng: ThreadRng,
}

impl DQNAgent {
    fn new(config: Config, device: Device) -> Self {
        let vs = nn::VarStore::new(device);
        let root = vs.root(); // Get the root path
        let q_local = QNetwork::new(&(&root / "local"), STATE_SIZE, ACTION_SIZE, config.hidden_dim);
        let q_target = QNetwork::new(&(&root / "target"), STATE_SIZE, ACTION_SIZE, config.hidden_dim);
        let optimizer = nn::Dhruva::default().build(&vs, config.lr).unwrap();
        let buffer = ReplayBuffer::new(config.buffer_size, config.batch_size, device);

        // Initialize target same as local
        // Use `copy_` on tensors directly if QNetwork members are public,
        // or use VarStore's load/save or manual copying if needed.
        // `tch::no_grad` ensures this initialization doesn't track gradients.
        tch::no_grad(|| {
            for (target_p, local_p) in q_target.trainable_variables().iter()
                .zip(q_local.trainable_variables().iter()) {
                target_p.copy_(local_p);
            }
        });


        println!("Initialized DQNAgent (MLP based) on device: {:?}", device);
        println!("  State Size: {}", STATE_SIZE);
        println!("  Action Size: {}", ACTION_SIZE);

        DQNAgent {
            vs,
            q_local,
            q_target,
            optimizer,
            buffer,
            config,
            epsilon: config.epsilon_start,
            time_step: 0,
            device,
            rng: rand::thread_rng(),
        }
    }

    fn preprocess_state(&self, state: &[f32]) -> Tensor {
        Tensor::of_slice(state)
            .view([1, STATE_SIZE]) // Add batch dimension
            .to_kind(Kind::Float)
            .to(self.device)
    }

    fn select_action(&mut self, state: &[f32], use_epsilon: bool) -> usize {
        let state_t = self.preprocess_state(state);

        // Get action values from local network in no_grad mode
        let action_values = tch::no_grad(|| {
            self.q_local.forward_t(&state_t, false)
        });

        if use_epsilon && self.rng.gen::<f64>() < self.epsilon {
            // Exploration
            self.rng.gen_range(0..ACTION_SIZE as usize)
        } else {
            // Exploitation
            action_values
                .argmax(Some(1), false) // argmax along action dimension (dim=1)
                .int64_value(&[]) as usize
        }
    }

    fn store_experience(&mut self, state: Vec<f32>, action: usize, reward: f32, next_state: Vec<f32>, done: bool) {
        let exp = Experience { state, action, reward, next_state, done };
        self.buffer.add(exp);

        // Learn every UPDATE_EVERY time steps, if buffer is ready
        self.time_step = (self.time_step + 1) % self.config.update_every;
        if self.time_step == 0 && self.buffer.len() >= self.config.batch_size {
            if let Some(experiences) = self.buffer.sample() {
                self.learn(experiences);
            }
        }
    }

    fn learn(&mut self, experiences: (Tensor, Tensor, Tensor, Tensor, Tensor)) {
        let (states_t, actions_t, rewards_t, next_states_t, dones_t) = experiences;

        // --- Calculate Target Q-values (using target network in no_grad)---
        let q_targets_next = tch::no_grad(|| {
            self.q_target
                .forward_t(&next_states_t, false) // Use target network
                .max_dim(1, false) // Get max Q-value along action dimension
                .0 // Keep only the values
                .view([-1, 1])
        });

        // Q_targets = r + gamma * max_a' Q_target(s', a') * (1 - done)
        let q_targets = &rewards_t + self.config.gamma * q_targets_next * (1.0 - &dones_t);

        // --- Calculate Expected Q-values (using local network) ---
        let q_expected_all = self.q_local.forward_t(&states_t, true); // Train mode
        // Select the Q-value for the action actually taken using gather
        let q_expected = q_expected_all.gather(1, &actions_t, false);

        // --- Compute Loss ---
        let loss = q_expected.mse_loss(&q_targets, tch::Reduction::Mean);

        // --- Optimize the Model ---
        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();

        // --- Update Target Network ---
        self.soft_update_target();
    }

    // Soft update target network: target = tau*local + (1-tau)*target
    fn soft_update_target(&mut self) {
        let tau = self.config.tau;
        let mut target_params = self.q_target.trainable_variables();
        let local_params = self.q_local.trainable_variables();

        tch::no_grad(|| {
            for (target, local) in target_params.iter_mut().zip(local_params.iter()) {
                target.copy_(&(tau * local + (1.0 - tau) * &*target));
            }
        });
    }

    fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_end);
    }
}


// --- Main Execution ---
fn main() {
    println!("==============================================");
    println!(" Starting DQN Agent Training (Rust / tch-rs) ");
    println!("==============================================");
    // Ensure LibTorch is correctly set up!
    let device = Device::cuda_if_available();
    println!("Timestamp: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"));
    println!("Location Context: Bengaluru, Karnataka, India"); // From Python code

    let config = Config::default();
    let mut env = StructuredTreeEnv::new(config.max_steps_per_episode);
    let mut agent = DQNAgent::new(config.clone(), device);

    // --- Pre-populate Replay Buffer ---
    println!("\n--- Pre-populating Replay Buffer ---");
    let pre_populate_steps = config.batch_size * 5; // Match Python logic
    let mut state = env.reset();
    for _ in 0..pre_populate_steps {
        let action = agent.rng.gen_range(0..ACTION_SIZE as usize); // Random actions
        let (next_state, reward, done) = env.step(action);
        agent.buffer.add(Experience {
            state: state.clone(), action, reward, next_state: next_state.clone(), done
        });
        state = next_state;
        if done {
            state = env.reset();
        }
    }
    println!("Replay buffer pre-populated with {} experiences.", agent.buffer.len());

    // --- Training Loop ---
    println!("\n--- Starting Training Loop ---");
    let training_start_time = Instant::now();
    let mut scores_window: VecDeque<f32> = VecDeque::with_capacity(10); // Match Python window size
    let mut total_scores: Vec<f32> = Vec::new();

    for i_episode in 1..=config.num_episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;

        for _t in 0..config.max_steps_per_episode { // Use _t if loop counter isn't needed
            // 1. Agent selects action
            let action = agent.select_action(&state, true);

            // 2. Environment executes action
            let (next_state, reward, done) = env.step(action);

            // 3. Agent stores experience and learns (learn is called inside store_experience)
            agent.store_experience(state.clone(), action, reward, next_state.clone(), done);

            // Update state and reward
            state = next_state;
            episode_reward += reward;

            if done {
                break; // End episode early if done
            }
        } // End step loop

        // --- End of Episode ---
        total_scores.push(episode_reward);
        if scores_window.len() == 10 { // Maintain window size
            scores_window.pop_front();
        }
        scores_window.push_back(episode_reward);
        agent.decay_epsilon();

        let avg_score: f32 = if scores_window.is_empty() {
            0.0
        } else {
            scores_window.iter().sum::<f32>() / scores_window.len() as f32
        };


        // Print progress (mimic Python output)
        print!(
            "\rEpisode {}\tTotal Reward: {:.2}\tAverage Reward (last {}): {:.2}\tEpsilon: {:.3}   ",
            i_episode,
            episode_reward,
            scores_window.len(),
            avg_score,
            agent.epsilon
        );
        use std::io::{self, Write}; // Flush output
        io::stdout().flush().unwrap();

        if i_episode % 10 == 0 { // Newline every 10 episodes
            println!(
                "\rEpisode {}\tTotal Reward: {:.2}\tAverage Reward (last {}): {:.2}\tEpsilon: {:.3}   ",
                i_episode,
                episode_reward,
                scores_window.len(),
                avg_score,
                agent.epsilon
            );
        }

    } // End episode loop

    let training_duration = training_start_time.elapsed();
    println!("\n--- Training Finished ---");
    println!("Total training time: {:?}", training_duration);

    // env.close(); // Close method isn't strictly necessary here

    println!("\n==============================================");
    println!(" DQN Agent Training Finished ");
    println!("==============================================");
}