// src/main-simulation
use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Device, Kind, Tensor};
use rand::prelude::*;
use std::{collections::VecDeque, time::Instant};

const STATE_SIZE: i64 = 5;
const ACTION_SIZE: i64 = 4;

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
            hidden_dim: 64,
            buffer_size: 50_000, // Increased from Python example for better stability
            batch_size: 64,
            gamma: 0.99,
            lr: 5e-4,
            tau: 1e-3,
            update_every: 4,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            num_episodes: 500, // Increased for potentially longer learning in Rust/tch
            max_steps_per_episode: 200,
        }
    }
}

// --- Environment Simulation ---
#[derive(Debug)]
struct StructuredTreeEnv {
    state: Vec<f32>,
    rng: ThreadRng,
    action_names: [&'static str; ACTION_SIZE as usize],
}

impl StructuredTreeEnv {
    fn new() -> Self {
        println!(
            "Initializing StructuredTreeEnv (State Size: {}, Action Size: {})",
            STATE_SIZE, ACTION_SIZE
        );
        StructuredTreeEnv {
            state: vec![0.0; STATE_SIZE as usize],
            rng: rand::thread_rng(),
            action_names: ["Water", "Prune", "Fertilize", "Wait"],
        }
    }

    fn reset(&mut self) -> Vec<f32> {
        self.state = vec![
            self.rng.gen_range(0.5..2.0), // Height
            self.rng.gen_range(1..=5) as f32, // Branches
            self.rng.gen_range(20..=50) as f32, // Leaves
            self.rng.gen_range(0.7..0.9), // Health
            self.rng.gen_range(0..=3) as f32, // Age
        ];
        self.state.clone()
    }

    // Returns (next_state, reward, done)
    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) {
        if action >= ACTION_SIZE as usize {
            eprintln!("Warning: Invalid action {} received! Treating as Wait.", action);
            // action = 3; // Treat as Wait if needed, or handle differently
        }

        let mut height = self.state[0];
        let mut branches = self.state[1];
        let mut leaves = self.state[2];
        let mut health = self.state[3];
        let mut age = self.state[4];
        let mut reward = 0.0;

        match action {
            0 => { // Water
                let health_increase = self.rng.gen_range(0.05..0.15) * (1.1 - health);
                health = (health + health_increase).min(1.0);
                reward = 0.5 + health;
            }
            1 => { // Prune
                if branches > 1.0 && leaves > 15.0 {
                    let branches_removed = 1.0;
                    let leaves_lost = self.rng.gen_range((leaves * 0.1) as i32..(leaves * 0.3) as i32) as f32;
                    branches -= branches_removed;
                    leaves = (leaves - leaves_lost).max(10.0);
                    health = (health - 0.02).max(0.0);
                    reward = 0.3 * health;
                } else {
                    reward = -0.5;
                }
            }
            2 => { // Fertilize
                let height_increase = self.rng.gen_range(0.1..0.3) * health;
                let leaves_increase = self.rng.gen_range(5..=10) as f32 * health;
                height += height_increase;
                leaves += leaves_increase;
                reward = 0.6 + height_increase * 2.0;
            }
            3 | _ => { // Wait or Invalid Action fallback
                age += 0.1;
                let natural_growth = 0.05 * health * (1.0 - age / 20.0).max(0.0);
                height += natural_growth;
                if self.rng.gen::<f32>() < 0.1 * health { branches += 1.0; }
                leaves += self.rng.gen_range(1.0..5.0) * health;
                health = (health - 0.03).max(0.0);
                reward = -0.2 * (1.0 - health);
            }
        }

        self.state = vec![height, branches, leaves, health, age];
        let done = health <= 0.0;
        if done {
            reward = -10.0;
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
            .collect::<Vec<_>>() // Collect to Vec to sample
            .choose_multiple(&mut self.rng, self.batch_size) // Sample without replacement
            .cloned()
            .collect();

        let states: Vec<f32> = samples.iter().flat_map(|e| e.state.clone()).collect();
        let actions: Vec<i64> = samples.iter().map(|e| e.action as i64).collect();
        let rewards: Vec<f32> = samples.iter().map(|e| e.reward).collect();
        let next_states: Vec<f32> = samples.iter().flat_map(|e| e.next_state.clone()).collect();
        let dones: Vec<f32> = samples.iter().map(|e| if e.done { 1.0 } else { 0.0 }).collect();

        let states_t = Tensor::of_slice(&states)
            .view([-1, STATE_SIZE])
            .to(self.device);
        let actions_t = Tensor::of_slice(&actions)
            .view([-1, 1]) // Ensure shape is [batch_size, 1] for gather
            .to_kind(Kind::Int64) // Important for gather
            .to(self.device);
        let rewards_t = Tensor::of_slice(&rewards).view([-1, 1]).to(self.device);
        let next_states_t = Tensor::of_slice(&next_states)
            .view([-1, STATE_SIZE])
            .to(self.device);
        let dones_t = Tensor::of_slice(&dones).view([-1, 1]).to(self.device);

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

// --- DQN Agent ---
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
        let q_local = QNetwork::new(&vs.root(), STATE_SIZE, ACTION_SIZE, config.hidden_dim);
        let q_target = QNetwork::new(&vs.root(), STATE_SIZE, ACTION_SIZE, config.hidden_dim);
        let mut optimizer = nn::Dhruva::default().build(&vs, config.lr).unwrap();
        let buffer = ReplayBuffer::new(config.buffer_size, config.batch_size, device);

        // Initialize target same as local
        q_target.copy(&q_local);

        println!("Initialized DQNAgent (MLP based) on device: {:?}", device);

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
            .to(self.device)
    }

    fn select_action(&mut self, state: &[f32], use_epsilon: bool) -> usize {
        let state_t = self.preprocess_state(state);

        let action_values = self.vs.no_grad(|| { // Disable gradient calculation for inference
            self.q_local.forward_t(&state_t, false) // train=false
        });

        if use_epsilon && self.rng.gen::<f64>() < self.epsilon {
            // Exploration
            self.rng.gen_range(0..ACTION_SIZE as usize)
        } else {
            // Exploitation
            action_values
                .argmax(Some(1), false) // argmax along the action dimension (dim=1)
                .int64_value(&[]) as usize // Get the index as usize
        }
    }

    fn store_experience(&mut self, state: Vec<f32>, action: usize, reward: f32, next_state: Vec<f32>, done: bool) {
        let exp = Experience { state, action, reward, next_state, done };
        self.buffer.add(exp);

        self.time_step = (self.time_step + 1) % self.config.update_every;
        if self.time_step == 0 {
            if self.buffer.len() >= self.config.batch_size {
                if let Some(experiences) = self.buffer.sample() {
                    self.learn(experiences);
                }
            }
        }
    }

    fn learn(&mut self, experiences: (Tensor, Tensor, Tensor, Tensor, Tensor)) {
        let (states_t, actions_t, rewards_t, next_states_t, dones_t) = experiences;

        // --- Calculate Target Q-values ---
        let q_targets_next = self.vs.no_grad(|| {
            self.q_target
                .forward_t(&next_states_t, false) // Use target network
                .max_dim(1, false) // Get max Q-value along action dimension (returns values, indices)
                .0 // Keep only the values
                .view([-1, 1]) // Ensure shape [batch_size, 1]
        });

        // Q_targets = r + gamma * max_a' Q_target(s', a') * (1 - done)
        let q_targets = &rewards_t + self.config.gamma * q_targets_next * (1.0 - &dones_t);


        // --- Calculate Expected Q-values ---
        // Get Q-values from local network for all actions
        let q_expected_all = self.q_local.forward_t(&states_t, true); // train=true
        // Select the Q-value for the action actually taken
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

    fn soft_update_target(&mut self) {
        let tau = self.config.tau;
        let local_vars = self.q_local.variables(); // Assuming QNetwork exposes vars or we access via vs
        let target_vars = self.q_target.variables(); // Assuming QNetwork exposes vars

        self.vs.no_grad(|| {
            for (name, target_var) in target_vars.iter() {
                if let Some(local_var) = local_vars.get(name) {
                    target_var.copy_(&(tau * local_var + (1.0 - tau) * &*target_var));
                }
            }
        });
    }

    // Helper to get variables (might need adjustment based on how QNetwork is structured with vs)
    // This is a simplification - direct access might differ
    fn _variables(&self, network: &QNetwork) -> std::collections::HashMap<String, Tensor>{
        // This part is tricky with tch-rs, VarStore manages variables.
        // A common pattern is to access them via the VarStore path used to create the network.
        // This function is a placeholder concept.
        let mut vars = std::collections::HashMap::new();
        // Logic to extract named tensors associated with the network from self.vs
        // Example: vars.insert("fc1.weight".to_string(), self.vs.get("fc1.weight").unwrap());
        // This needs careful implementation based on tch-rs VarStore structure.
        // For soft_update, iterating through self.vs.trainable_variables() and filtering
        // based on names associated with q_local vs q_target paths might be needed.
        // Given the complexity, let's try a simplified soft_update using copy_ proportionaally
        let mut target_params = self.q_target.trainable_variables();
        let local_params = self.q_local.trainable_variables();

        tch::no_grad(|| {
            for (target, local) in target_params.iter_mut().zip(local_params.iter()) {
                target.copy_(&(tau * local + (1.0 - tau) * &*target));
            }
        });

        vars // Return empty for now as placeholder
    }

    fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_end);
    }
}

// --- Main Execution ---
fn main() {
    println!("==============================================");
    println!(" Starting DQN Agent Training for Tree Simulator (Rust / tch-rs) ");
    println!("==============================================");
    // NOTE: Ensure LibTorch is set up correctly!
    // Check for CUDA availability
    let device = Device::cuda_if_available();
    println!("Timestamp: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"));

    let config = Config::default();
    let mut env = StructuredTreeEnv::new();
    let mut agent = DQNAgent::new(config.clone(), device); // Clone config

    // --- Pre-populate Replay Buffer ---
    println!("\n--- Pre-populating Replay Buffer ---");
    let pre_populate_steps = config.batch_size * 10;
    let mut state = env.reset();
    for _ in 0..pre_populate_steps {
        let action = agent.rng.gen_range(0..ACTION_SIZE as usize); // Random actions
        let (next_state, reward, done) = env.step(action);
        // Use cloned state for storage
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
    let mut scores_window: VecDeque<f32> = VecDeque::with_capacity(100);
    let mut total_scores: Vec<f32> = Vec::new();

    for i_episode in 1..=config.num_episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;

        for t in 0..config.max_steps_per_episode {
            // 1. Agent selects action
            let action = agent.select_action(&state, true);

            // 2. Environment executes action
            let (next_state, reward, done) = env.step(action);

            // 3. Agent stores experience and learns
            agent.store_experience(state.clone(), action, reward, next_state.clone(), done);

            // Update state and reward
            state = next_state;
            episode_reward += reward;

            if done {
                break;
            }
        } // End step loop

        // --- End of Episode ---
        total_scores.push(episode_reward);
        if scores_window.len() == 100 {
            scores_window.pop_front();
        }
        scores_window.push_back(episode_reward);
        agent.decay_epsilon();

        let avg_score: f32 = scores_window.iter().sum::<f32>() / scores_window.len() as f32;

        print!(
            "\rEpisode {}\tAvg Reward (Last {}): {:.2}\tReward: {:.2}\tEpsilon: {:.3}   ",
            i_episode,
            scores_window.len(),
            avg_score,
            episode_reward,
            agent.epsilon
        );
        // Flush stdout to ensure progress is shown
        use std::io::{self, Write};
        io::stdout().flush().unwrap();

        if i_episode % 100 == 0 {
            println!(
                "\rEpisode {}\tAvg Reward (Last {}): {:.2}\tReward: {:.2}\tEpsilon: {:.3}   ",
                i_episode,
                scores_window.len(),
                avg_score,
                episode_reward,
                agent.epsilon
            );
        }

    } // End episode loop

    let training_duration = training_start_time.elapsed();
    println!("\n--- Training Finished ---");
    println!("Total training time: {:?}", training_duration);

    // env.close(); // Not strictly needed

    println!("\n==============================================");
    println!(" DQN Agent Training (Rust) Finished ");
    println!("==============================================");
}