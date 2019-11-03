## HW2

### Command
```
$python3 main.py
```

### Epsilon Value Decay

* As our traning progress, we would like to have a more deterministic policy. Therefore, we need to reduce the randomness in our policy. In the other word, we are going to reduce our epsilon as time goes on.

* The initial value of epsilon will be 1, and the smallest epsilon value will be 0.1.
* The decay used in my code in exponential decay. At beginning, I use the decay rate given in the template, but I later decrease the dacay rate. I reason I make this change will be discuss later.

* The implementation in python is as shown in the image below.

```python=
def epsilon_by_frame(frame_idx):
    epsilon = max(math.exp(-(1/epsilon_decay)*frame_idx), epsilon_final)
    return epsilon

```


### Epsilon Greedy Alogrithm

* First, we need to decide for this timestep we want to act greedily or randomly. We use random.random(), which gives a random float from 0.0 to 1.0 and compare it to our epsilon.

* If the random number is bigger than our epsilon, we act greedily. Otherwise, our agent will move randomly. We can see that if epsilon is 1, there's no way our random number is bigger than epsilon. In this situation, our agent will act 100% randomly. On the other hand, if our epsilon decays to a very small number, the probability of our random number bigger than epislon will be great. This way, there's a big chance that our agent will act greedily.

* The detailed implementation of Epsilon Greedy in python is as shown in the image below.

```python=
def act(state, epsilon):
    action = 0
    max = Q[state][0];
    print(epsilon)
    if random.random() > epsilon:
        for i in range (1, 4):
            if Q[state][i] > max:
                action = i;
    else:
        action = random.randint(0, 3)

    return action
```

### Learning

* First get the epsilon of the current time step.

* Use epsilon greedy to decide our action

* Use the generated action to interact with the environment and get reward ... etc.

* Use epsilon greedy to decide our next action given the new state (this action will be used the retreive the bootstrapped action value later)

* update the Q value using formula of SARSA.

* Code in python is shown below.

```python=

for frame_idx in range(1, num_frames + 1):
    # get epsilon
    epsilon = epsilon_by_frame(frame_idx)

    # forward
    action  = act(state, epsilon)

    # interact with environment
    env.render()
    next_state, reward, done, info = env.step(action)
    next_action = act(next_state, epsilon)

    # update Q table
    Q[state][action] = Q[state][action] + rate * (reward + gamma * Q[next_state][next_action] - Q[state][action])

    # go to next state
    state = next_state
    episode_reward += reward
```

### Problem encountered

* At first, I didn't know how to decay my epsilon value. After reviewing slides from the lecture, I decided to set epsilon to 1 / timestep. However, this failed train anything successful. I later find out late the reason is that my epsilon decayed way too quick, limiting the random exploration time of the agent. Since the reward in this problem is only given when our agent reach the goal. Accordig to that, I noticed if the agent can't reach the goal no update will be performed and the action value stored in array Q will all be zero. Because this failure to update the aciton value function, my agent was stuck at the starting point. This happened because our agent has stop exploring and will act greedily. However, since all Q are zero, it will always go in the default direction, which is left) I later changed my decay function to exponantial decay, and it did work better than the old method.

* When I changed to exponantial decay using the decay rate given in the template, sometimes i still get these failed result with all zero Q value. To cope with this, I increase epsilon decay, so epsilon decays much slower. This did decreases the number of divergent training result substaintially.

* Even with decreased decay rate, sometimes the agent still fail to train. I then increase the minimum epsilon from 0.01 to 0.1, so the agent will still perform some degree of exploration even after many timestep.
