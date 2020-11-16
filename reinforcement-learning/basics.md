# Basics

### Reward

![](../.gitbook/assets/image%20%28108%29.png)

#### Cumulative Reward

Maximize the expected\(upcoming future\) cumulative reward. What is and how to get the future or expected reward?

![](../.gitbook/assets/image%20%28112%29.png)

![](../.gitbook/assets/image%20%2895%29.png)

#### Discounted return

The agent tries to select actions so that the sum of the discounted rewards it receives over the future is maximized.

Discounted rate: \[0,1\] give more weight to current or immediate rewards.  User design. It can be used to set limits of rewards not looking too far in continuous tasks.

![](../.gitbook/assets/image%20%2896%29.png)

### MDP: Markov Decision Process

MDP: Reinforcement framework. It works on both continuing and episodic task.

If the state signal has the Markov property, on the other hand, then the environment's response at t+1 depends only on the state and action representations at t, in which case the environment's dynamics can be de ned by specifying only

![](../.gitbook/assets/image%20%28119%29.png)

One-step Dynamics

* The environment considers only the state and action at the previous time step \(S\_t, A\_t\) at time t+1. 
* It does not care what States and  Actions were more than one step prior.  
  * {S\_0 ... S\_\(t-1\) },  {A\_0 ... A\_\(t-1\) }
* It does not consider any previous rewards of  to respond to the agent.
  *  {R\_0 ... R\_\(t\) }
* The environment decides the state and reward by

![](../.gitbook/assets/image%20%28106%29.png)

#### Pole-cart example

![](../.gitbook/assets/image%20%28102%29.png)

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

![](../.gitbook/assets/image%20%28101%29.png)

Deterministic Policy

Stochastic Policy

![](../.gitbook/assets/image%20%28104%29.png)

![](../.gitbook/assets/image%20%28103%29.png)

![](../.gitbook/assets/image%20%28109%29.png)

### Example: Episodic problem of finding the goal

The reward map

![](../.gitbook/assets/image%20%28117%29.png)

* Option 1:  An example of a bad policy. 
  * Starting at S\(1,1\) ,  cumulative reward score for this policy = -6
  * Starting at S\(1,2\) ,  cumulative reward score for this policy = -6

![](../.gitbook/assets/image%20%28113%29.png)

![](../.gitbook/assets/image%20%28114%29.png)

* * For every other state for this bad policy . 
  * This is a function of the environment state:  State-Value Function
  * Each state has a value: Expected return by following this policy starting at that state
  * 

![](../.gitbook/assets/image%20%2897%29.png)

![](../.gitbook/assets/image%20%2899%29.png)

For the given policy, the state-value function starting in state 's' returns the 'expected' reward

If the policy changes, the state-value function changes.

* Then, how to find the optimal policy?

![](../.gitbook/assets/image%20%28100%29.png)

### Bellman Expectation Equation

In calculating for the values of state-value functions for given policy, we can effectively calculate the value  of any state,  using recursive property.

* Value of any state:  You only need the immediate reward and the value of the state that follows 

![](../.gitbook/assets/image%20%28111%29.png)

![](../.gitbook/assets/image%20%28116%29.png)

![](../.gitbook/assets/image%20%28120%29.png)

### Monte Carlo

### Dynamic Programming

### TD Learning

TD learning is a combination of Monte Carlo ideas and dynamic programming \(DP\) ideas. Like Monte Carlo methods, TD methods can learn directly from raw experience without a model of the environment's dynamics. Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome \(they bootstrap\).

> model-free, update without the final outcome



