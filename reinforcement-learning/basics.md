# Basics

## Action, Reward, Policy

### Reward

![](../.gitbook/assets/image%20%28118%29.png)

#### Cumulative Reward

Maximize the expected\(upcoming future\) cumulative reward. What is and how to get the future or expected reward?

![](../.gitbook/assets/image%20%28127%29.png)

![](../.gitbook/assets/image%20%2896%29.png)

#### Discounted return

The agent tries to select actions so that the sum of the discounted rewards it receives over the future is maximized.

Discounted rate: \[0,1\] give more weight to current or immediate rewards.  User design. It can be used to set limits of rewards not looking too far in continuous tasks.

![](../.gitbook/assets/image%20%2898%29.png)

### MDP: Markov Decision Process

MDP: Reinforcement framework. It works on both continuing and episodic task.

If the state signal has the Markov property, on the other hand, then the environment's response at t+1 depends only on the state and action representations at t, in which case the environment's dynamics can be de ned by specifying only

![](../.gitbook/assets/image%20%28140%29.png)

One-step Dynamics

* The environment considers only the state and action at the previous time step \(S\_t, A\_t\) at time t+1. 
* It does not care what States and  Actions were more than one step prior.  
  * {S\_0 ... S\_\(t-1\) },  {A\_0 ... A\_\(t-1\) }
* It does not consider any previous rewards of  to respond to the agent.
  *  {R\_0 ... R\_\(t\) }
* The environment decides the state and reward by

![](../.gitbook/assets/image%20%28114%29.png)

#### Pole-cart example

![](../.gitbook/assets/image%20%28105%29.png)

It is an MDP problem, but not finite MDP.  The  environment considers only the current action and states not the previous ones to get the next reward.

* Actions: 2 control input
  * move left
  * move right
* State: there are  4 observations
  * Cart position, cart velocity
  * Pole angle, Pole velocity
* Reward: 
  * Not fall +1,  fall -1

### Policy

![](../.gitbook/assets/image%20%28103%29.png)

Deterministic Policy

Stochastic Policy

![](../.gitbook/assets/image%20%28109%29.png)

![](../.gitbook/assets/image%20%28107%29.png)

![](../.gitbook/assets/image%20%28120%29.png)

### State-Value Function

#### Example: Episodic problem of finding the goal \(deterministic policy\)

The reward map

![](../.gitbook/assets/image%20%28137%29.png)

* Option 1:  An example of a bad policy. 
  * Starting at S\(1,1\) ,  cumulative reward score for this policy = -6
  * Starting at S\(1,2\) ,  cumulative reward score for this policy = -6

![](../.gitbook/assets/image%20%28129%29.png)

![](../.gitbook/assets/image%20%28130%29.png)

* * For every other state for this bad policy . 
  * This is a function of the environment state:  State-Value Function
  * Each state has a value: Expected return by following this policy starting at that state
  * 

![](../.gitbook/assets/image%20%2899%29.png)

![](../.gitbook/assets/image%20%28101%29.png)

For the given policy, the state-value function starting in state 's' returns the 'expected' reward

If the policy changes, the state-value function changes.

![](../.gitbook/assets/image%20%28102%29.png)

### Bellman Expectation Equation

In calculating for the values of state-value functions for given policy, we can effectively calculate the value  of any state,  using recursive property.

* Value of any state:  You only need the immediate reward and the value of the state that follows 
* > But in complicated worlds, the immediate reward and next state cannot be known with certainty.

![](../.gitbook/assets/image%20%28126%29.png)

![](../.gitbook/assets/image%20%28133%29.png)

![](../.gitbook/assets/image%20%28143%29.png)

### An Optimal Policy

Then, how to find the optimal policy? There are numbers of different policy. How can we define  if one policy is better than the other? 

* A better policy if **all** state has equal or better values
  * some policies may not be able to be compared
* Optimal policy may not be unique

![](../.gitbook/assets/image%20%28124%29.png)

![](../.gitbook/assets/image%20%28121%29.png)

### Action-Value function

At that state s, there could be multiple choices of action to take for the given policy.  

The optimal action-value function is denoted as :   q\*

![](../.gitbook/assets/image%20%2897%29.png)

![](../.gitbook/assets/image%20%28113%29.png)

## 

Example: starting at s\(0,0\), if action 'down' is chosen, then it follows the policy for all future time steps\(rest states\) to give the reward '0'.  It can continue for all other starting states and the action it takes at that state. 

* Starting in that state & taking the action --&gt; follows the policy for the rest steps

![](../.gitbook/assets/image%20%28111%29.png)

![](../.gitbook/assets/image%20%28104%29.png)

### Optimal Policy from Optimal action-value function

![](../.gitbook/assets/image%20%28142%29.png)

![](../.gitbook/assets/image%20%28147%29.png)

#### Example: 

That maximizes the action-value function for each state can be found from the table. The policy can give '1' probability to make sure this action is taken all the time. 

There can be multiple actions as in S3:  either a1 or a2.  We can give  probability p,q for each of them and '0' probability for the other actions. 

![](../.gitbook/assets/image%20%28112%29.png)

## Monte Carlo

How to interact with the environment to find the optimal policy? Which action is the best from each state?

 To gain the useful understanding of the environment, it needs many episodes of random actions. 

#### Q-Table

Q\(s,a\), q values from the action-value functions for given policy. 

![](../.gitbook/assets/image%20%28119%29.png)

#### MC Prediction

If the true action-value function is NOT known, we can predict \(approximate\) it with Q-tables from many episodes

It is estimating the q values of action-value function with a Q-table., with Monte Carlo approach.

In a single episode, the same action is selected from the same action multiple times.

* Every-visit MC Prediction:  average the returns of all visits to each state-action pair
* First-visit MC Prediction: consider only the first visit to the state-action pair

![](../.gitbook/assets/image%20%28144%29.png)

### Greedy Policy and Epsilon Greedy Policy

How to find an optimal policy from Q-table?

Construct the policy that is greedy for a better policy

![](../.gitbook/assets/image%20%28108%29.png)

![](../.gitbook/assets/image%20%28116%29.png)

Greedy policy always select the greedy action But, we want to **explore** all other possibilities. To give a probility of epsilon in selecting other action than the greedy action,  use:

![](../.gitbook/assets/image%20%28117%29.png)

### Exploitation - Exploration Dilemma

There is a trade-off between Exploitation and Exploration

* Exploitation:  action based on past experience. trusting more on experience \(epilon value is low\)
* Exploration: select range of various possibilities \(epsilon value is high\)

Make sense to favor exploration over exploitation initially, when the environment is not fully known. As the policy gradually becomes more greedy, it make sense to favor exploitation over exploration

* $e\_i &gt; 0$ for all step i  &  $e\_i$ decays to zero as i approaches inf. 

### When to update the Q-table?  

* Conventionally, the Q-table is updated once after all the episode iterations & Q-table have converged. 
* For more efficiency, update Q-table every episode iteration:  **Incremental Mean**
* \(1/N\) can be tuned as alpha
  * $$\alpha$$ is large,  trusting the most recent sampled\(experienced\) return  $$G_t$$ 
    * focus on more recent experience
    * faster learning, but may not converge
  * if $$\alpha$$ is very low,  tend not to update by the agent
    *  considering the longer history of returns

![](../.gitbook/assets/image%20%28125%29.png)

![](../.gitbook/assets/image%20%28106%29.png)

![](../.gitbook/assets/image%20%28128%29.png)

## Dynamic Programming

## TD Learning

TD learning is a combination of Monte Carlo ideas and dynamic programming \(DP\) ideas. Like Monte Carlo methods, TD methods can learn directly from raw experience without a model of the environment's dynamics. Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome \(they bootstrap\).

> MC:  update Q-table after an episode
>
> TD:  update Q-table at every step,  model-free,

![](../.gitbook/assets/image%20%28141%29.png)

![](../.gitbook/assets/image%20%28122%29.png)

### 

![](../.gitbook/assets/image%20%28146%29.png)

### Sarsa\(0\)

![](../.gitbook/assets/image%20%28134%29.png)

### Q-Learning \( Sarsamax\)

Chooses the Action, after updating the Q-table at each step.

![](../.gitbook/assets/image%20%28135%29.png)

![](../.gitbook/assets/image%20%28138%29.png)





![](../.gitbook/assets/image%20%28132%29.png)

## Resources

RL Cheat sheet- Udacity 

{% file src="../.gitbook/assets/cheatsheet.pdf" %}

## QnA

Difference between max and argmax?

> max is the unique value :  -x^2
>
> argmax is a set of values for the maximum value:  e.g.  sinX

