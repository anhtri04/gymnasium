# 2D Top-Down 1v1 Football Game

## Introduction

A top-down 2D football game where two AI agents learn to play against each other through reinforcement learning. The field is viewed from above, like a tactical board game. Each player is a **rectangle with a facing direction**, meaning they must turn to face the ball before dribbling or kicking — adding a layer of realism and strategic depth. The agents start knowing nothing, and through thousands of self-play matches, they gradually learn to move, turn, chase the ball, and score goals. The entire learning process is driven by a simple principle: reward scoring, punish conceding, and let the math figure out the rest.

---

## Full Task List

### Phase 1 — Game Engine (No AI Yet)

- [ ] Set up project structure and install dependencies (pygame, torch, gymnasium)
- [ ] Create the game window and top-down field (green rectangle, white lines, center circle)
- [ ] Draw goals as openings on the left and right edges of the field
- [ ] Implement the player as a **rectangle** with a **facing direction** (stored as an angle)
- [ ] Implement player movement — move forward/backward along facing direction
- [ ] Implement player rotation — turn left/right to change facing direction
- [ ] Render the player rectangle rotated correctly using `pygame.transform.rotate()`
- [ ] Implement ball as a circle with position and velocity
- [ ] Implement ball physics — velocity, friction (ball slows down each frame), wall bounce
- [ ] Implement **kick arc** — a cone in front of the player where the ball can be kicked
- [ ] Implement kicking — when ball is inside the kick arc, apply velocity to ball in facing direction
- [ ] Implement dribbling — ball rolls ahead of player when close and aligned
- [ ] Implement player-wall collision (players can't leave the field)
- [ ] Implement goal detection (ball crosses left or right goal line opening)
- [ ] Implement game reset after a goal (players and ball return to starting positions)
- [ ] Add episode time limit (game ends after N seconds if no goal is scored)
- [ ] Add a simple scoreboard display
- [ ] Playtest manually with keyboard controls — make sure physics and rotation feel right

---

### Phase 2 — Gym Environment Wrapper

- [ ] Convert the game into a `gymnasium.Env` class
- [ ] Define the **observation space** — player 1 position, player 1 angle, player 2 position, player 2 angle, ball position, ball velocity (all normalized to 0–1)
- [ ] Define the **action space** — discrete: forward, backward, rotate left, rotate right, kick
- [ ] Implement `reset()` — resets game state and returns initial observation
- [ ] Implement `step(action)` — advances one frame, returns (obs, reward, done, info)
- [ ] Define the initial **reward function** — +1 for scoring, -1 for conceding
- [ ] Test the environment by stepping through it with random actions
- [ ] Verify observations and rewards are correct at each step

---

### Phase 3 — Single Agent Baseline

- [ ] Build the **Policy Network** (MLP: observation → action probabilities)
- [ ] Build the **Value Network** (MLP: observation → single value estimate)
- [ ] Implement PPO training loop, or integrate Stable-Baselines3
- [ ] Train one agent against a **fixed dummy opponent** (stands still or moves randomly)
- [ ] Verify the agent learns to score against the dummy
- [ ] Plot reward per episode over time to confirm learning is happening

---

### Phase 4 — Two Agent Self-Play

- [ ] Extend `step()` to accept two actions and return two separate rewards
- [ ] **Mirror the observation for Player 2** — flip field coordinates so both agents share the same perspective (always attacking right)
- [ ] Run two policy networks simultaneously, one per agent
- [ ] Implement self-play training loop — both agents update from the same match experience
- [ ] Save model checkpoints periodically
- [ ] Implement **opponent pool** — sample a random past checkpoint as the opponent each episode to prevent the agents from overfitting to each other

---

### Phase 5 — Reward Shaping

- [ ] Add reward for **rotating to face the ball**
- [ ] Add reward for **moving toward the ball**
- [ ] Add reward for **touching / dribbling the ball**
- [ ] Add reward for **ball moving toward opponent's goal**
- [ ] Add small penalty for **standing still** too long
- [ ] Add small penalty for **hitting the walls**
- [ ] Tune all reward weights — retrain and compare agent behavior before and after

---

### Phase 6 — Visualization & Evaluation

- [ ] Add a **render mode** to watch trained agents play in real time at normal speed
- [ ] Plot training curves — total reward, goals scored per episode, average episode length
- [ ] Run **evaluation episodes** — freeze policy weights, no training, measure true performance
- [ ] Watch the two final trained agents play a full match

---

### Optional Stretch Goals

- [ ] Add a **stamina system** — sprinting drains energy, affects movement speed
- [ ] Add a **tackle action** — knock the ball away from an opponent within range
- [ ] Expand to 5v5 with team coordination
- [ ] Let a human player compete against the trained AI
- [ ] Add a simple main menu and match replay system