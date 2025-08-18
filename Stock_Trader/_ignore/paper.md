[cite_start]Check for [cite: 1]
[cite_start]Deep Reinforcement Learning for Automated Stock Trading: [cite: 2]
[cite_start]An Ensemble Strategy [cite: 3]
[cite_start]Hongyang Yang [cite: 4]
[cite_start]hy2500@columbia.edu [cite: 4]
AI4Finance LLC. [cite_start]& Columbia University [cite: 5]
[cite_start]New York City, New York [cite: 5]
[cite_start]Xiao-Yang Liu [cite: 25]
[cite_start]xl2427@columbia.edu [cite: 25]
[cite_start]Electrical Engineering, Columbia University [cite: 26]
[cite_start]New York City, New York [cite: 26]
[cite_start]Shan Zhong [cite: 7]
[cite_start]sz2495@columbia.edu [cite: 7]
[cite_start]Wormpex AI Research [cite: 7]
[cite_start]Bellevue, Washington [cite: 7]
[cite_start]Anwar Walid [cite: 27]
[cite_start]anwar.walid@nokia-bell-labs.com [cite: 27]
[cite_start]Nokia-Bell Labs [cite: 29]
[cite_start]Murray Hill, New Jersey [cite: 30]

[cite_start]**ABSTRACT** [cite: 6]

[cite_start]Stock trading strategies play a critical role in investment[cite: 8]. [cite_start]However, it is challenging to design a profitable strategy in a complex and dynamic stock market[cite: 8]. [cite_start]In this paper, we propose an ensemble strategy that employs deep reinforcement schemes to learn a stock trading strategy by maximizing investment return[cite: 9]. [cite_start]We train a deep reinforcement learning agent and obtain an ensemble trading strategy using three actor-critic based algorithms: Proximal Policy Optimization (PPO), Advantage Actor Critic (A2C), and Deep Deterministic Policy Gradient (DDPG)[cite: 10]. [cite_start]The ensemble strategy inherits and integrates the best features of the three algorithms, thereby robustly adjusting to different market situations[cite: 11]. [cite_start]In order to avoid the large memory consumption in training networks with continuous action space, we employ a load-on-demand technique for processing very large data[cite: 12]. [cite_start]We test our algorithms on the 30 Dow Jones stocks that have adequate liquidity[cite: 13]. [cite_start]The performance of the trading agent with different reinforcement learning algorithms is evaluated and compared with both the Dow Jones Industrial Average index and the traditional min-variance portfolio allocation strategy[cite: 14]. [cite_start]The proposed deep ensemble strategy is shown to outperform the three individual algorithms and two baselines in terms of the risk-adjusted return measured by the Sharpe ratio[cite: 15].

[cite_start]**CCS CONCEPTS** [cite: 16]

• **Computing methodologies → Machine learning; Neural networks; Markov decision processes; Reinforcement learning; Policy iteration; [cite_start]Value iteration.** [cite: 17]

[cite_start]**KEYWORDS** [cite: 28]

[cite_start]Deep reinforcement learning, Markov Decision Process, automated stock trading, ensemble strategy, actor-critic framework [cite: 31]

[cite_start]**ACM Reference Format:** [cite: 32]

Hongyang Yang, Xiao-Yang Liu, Shan Zhong, and Anwar Walid. 2020. [cite_start]Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy[cite: 33]. [cite_start]In ACM International Conference on Al in Finance (ICAIF 20), October 15-16, 2020, New York, NY, USA[cite: 34]. [cite_start]ACM, New York, NY, USA, 8 pages, [https://doi.org/10.1145/3383455.3422540](https://doi.org/10.1145/3383455.3422540) [cite: 35]

---

### [cite_start]**1 INTRODUCTION** [cite: 36]

[cite_start]Profitable automated stock trading strategy is vital to investment companies and hedge funds[cite: 37]. [cite_start]It is applied to optimize capital allocation and maximize investment performance, such as expected return[cite: 38]. [cite_start]Return maximization can be based on the estimates of potential return and risk[cite: 39]. [cite_start]However, it is challenging for analysts to consider all relevant factors in a complex and dynamic stock market [3, 23, 47][cite: 40].

[cite_start]Existing works are not satisfactory[cite: 41]. [cite_start]A traditional approach that employed two steps was described in [31][cite: 41]. [cite_start]First, the expected stock return and the covariance matrix of stock prices are computed[cite: 42]. [cite_start]Then, the best portfolio allocation strategy can be obtained by either maximizing the return for a given risk ratio or minimizing the risk for a pre-specified return[cite: 43]. [cite_start]This approach, however, is complex and costly to implement since the portfolio managers may want to revise the decisions at each time step, and take other factors into account, such as transaction cost[cite: 44]. [cite_start]Another approach for stock trading is to model it as a Markov Decision Process (MDP) and use dynamic programming to derive the optimal strategy [4, 5, 34, 35][cite: 45]. [cite_start]However, the scalability of this model is limited due to the large state spaces when dealing with the stock market[cite: 46].

[cite_start]In recent years, machine learning and deep learning algorithms have been widely applied to build prediction and classification models for the financial market[cite: 47]. [cite_start]Fundamentals data (earnings report) and alternative data (market news, academic graph data, credit card transactions, and GPS traffic, etc.) are combined with machine learning algorithms to extract new investment alphas or predict a company's future performance [10, 16, 45, 46][cite: 48]. [cite_start]Thus, a predictive alpha signal is generated to perform stock selection[cite: 49]. [cite_start]However, these approaches are only focused on picking high performance stocks rather than allocating trade positions or shares between the selected stocks[cite: 50, 64]. [cite_start]In other words, the machine learning models are not trained to model positions[cite: 65].

[cite_start]In this paper, we propose a novel ensemble strategy that combines three deep reinforcement learning algorithms and finds the optimal trading strategy in a complex and dynamic stock market[cite: 66]. [cite_start]The three actor-critic algorithms [24] are Proximal Policy Optimization (PPO) [28, 37], Advantage Actor Critic (A2C) [32, 48], and Deep Deterministic Policy Gradient (DDPG) [28, 29, 44][cite: 67]. [cite_start]Our deep reinforcement learning approach is described in Figure 1[cite: 68]. [cite_start]By applying the ensemble strategy, we make the trading strategy more robust and reliable[cite: 68]. [cite_start]Our strategy can adjust to different market situations and maximize return subject to risk constraint[cite: 69]. [cite_start]First, we build an environment and define action space, state space, and reward function[cite: 70]. [cite_start]Second, we train the three algorithms that take actions in the environment[cite: 71]. [cite_start]Third, we ensemble the three agents together using the Sharpe ratio that measures the risk-adjusted return[cite: 72]. [cite_start]The effectiveness of the ensemble strategy is verified by its higher Sharpe ratio than both the min-variance portfolio allocation strategy and the Dow Jones Industrial Average (DJIA)[cite: 73].

[cite_start]The remainder of this paper is organized as follows[cite: 74]. [cite_start]Section 2 introduces related works[cite: 74]. [cite_start]Section 3 provides a description of our stock trading problem[cite: 75]. [cite_start]In Section 4, we set up our stock trading environment[cite: 75]. [cite_start]In Section 5, we drive and specify the three actor-critic based algorithms and our ensemble strategy[cite: 76]. [cite_start]Section 6 describes the stock data preprocessing and our experimental setup, and presents the performance evaluation of the proposed ensemble strategy[cite: 77]. [cite_start]We conclude this paper in Section 7[cite: 78].

### [cite_start]**2 RELATED WORKS** [cite: 79]

[cite_start]Recent applications of deep reinforcement learning in financial markets consider discrete or continuous state and action spaces, and employ one of these learning approaches: critic-only approach, actor-only approach, or actor-critic approach [17][cite: 80]. [cite_start]Learning models with continuous action space provide finer control capabilities than those with discrete action space[cite: 81].

[cite_start]The critic-only learning approach, which is the most common, solves a discrete action space problem using, for example, Deep Q-learning (DQN) and its improvements, and trains an agent on a single stock or asset [9, 12, 21][cite: 85]. [cite_start]The idea of the critic-only approach is to use a Q-value function to learn the optimal action-selection policy that maximizes the expected future reward given the current state[cite: 86]. [cite_start]Instead of calculating a state-action value table, DON minimizes the error between estimated Q-value and target Q-value over a transition, and uses a neural network to perform function approximation[cite: 87]. [cite_start]The major limitation of the critic-only approach is that it only works with discrete and finite state and action spaces, which is not practical for a large portfolio of stocks, since the prices are of course continuous[cite: 88].

[cite_start]The actor-only approach has been used in [13, 22, 33][cite: 89]. [cite_start]The idea here is that the agent directly learns the optimal policy itself[cite: 90]. [cite_start]Instead of having a neural network to learn the Q-value, the neural network learns the policy[cite: 91]. [cite_start]The policy is a probability distribution that is essentially a strategy for a given state, namely the likelihood to take an allowed action[cite: 92]. [cite_start]Recurrent reinforcement learning is introduced to avoid the curse of dimensionality and improves trading efficiency in [33][cite: 93]. [cite_start]The actor-only approach can handle the continuous action space environments[cite: 94].

[cite_start]The actor-critic approach has been recently applied in finance [2, 26, 44, 48][cite: 95]. [cite_start]The idea is to simultaneously update the actor network that represents the policy, and the critic network that represents the value function[cite: 96]. [cite_start]The critic estimates the value function, while the actor updates the policy probability distribution guided by the critic with policy gradients[cite: 97]. [cite_start]Over time, the actor learns to take better actions and the critic gets better at evaluating those actions[cite: 98]. [cite_start]The actor-critic approach has proven to be able to learn and adapt to large and complex environments, and has been used to play popular video games, such as Doom [43][cite: 99]. [cite_start]Thus, the actor-critic approach is promising in trading with a large stock portfolio[cite: 100].

### [cite_start]**3 PROBLEM DESCRIPTION** [cite: 101]

[cite_start]We model stock trading as a Markov Decision Process (MDP), and formulate our trading objective as a maximization of expected return [20][cite: 102].

[cite_start]**3.1 MDP Model for Stock Trading** [cite: 103]

[cite_start]To model the stochastic nature of the dynamic stock market, we employ a Markov Decision Process (MDP) as follows: [cite: 103]
* [cite_start]**States $s=[p,h,b]$**: a vector that includes stock prices $p\in\mathbb{R}_{+}^{D}$. the stock shares $h\in\mathbb{Z}_{+}^{D}$, and the remaining balance $b\in\mathbb{R}_{+}$, where D denotes the number of stocks and $Z_{+}$ denotes non-negative integers[cite: 104].
* [cite_start]**Action a**: a vector of actions over D stocks[cite: 105]. [cite_start]The allowed actions on each stock include selling, buying, or holding, which result in decreasing, increasing, and no change of the stock shares h, respectively[cite: 105].
* [cite_start]**Reward $r(s,a,s^{\prime})$**: the direct reward of taking action a at state s and arriving at the new state $s^{\prime}$[cite: 106].
* [cite_start]**Policy $\pi(s)$**: the trading strategy at state s, which is the probability distribution of actions at state s[cite: 107].
* [cite_start]**Q-value $Q_{\pi}(s,a)$**: the expected reward of taking action a at state s following policy $\pi$[cite: 109].

[cite_start]The state transition of a stock trading process is shown in Figure 2[cite: 122]. [cite_start]At each state, one of three possible actions is taken on stock d (d=1,...,D) in the portfolio[cite: 122].
* [cite_start]Selling $k[d]\in[1,h[d]]$ shares results in $h_{t+1}[d]=h_{t}[d]-k[d]$, where $k[d]\in\mathbb{Z}_{+}$ and $d=1,...,D$[cite: 123].
* [cite_start]Holding, $h_{t+1}[d] = h_{t}[d]$[cite: 123].
* [cite_start]Buying k[d] shares results in $h_{t+1}[d]=h_{t}[d]+k[d]$[cite: 124].

[cite_start]At time t an action is taken and the stock prices update at t+1, accordingly the portfolio values may change from “portfolio value 0” to “portfolio value 1”, “portfolio value 2”, or “portfolio value 3” respectively, as illustrated in Figure 2[cite: 124]. [cite_start]Note that the portfolio value is $p^{T}h+b$[cite: 124].

[cite_start]**3.2 Incorporating Stock Trading Constraints** [cite: 125]

[cite_start]The following assumption and constraints reflect concerns for practice: transaction costs, market liquidity, risk-aversion, etc[cite: 125].
* [cite_start]**Market liquidity**: the orders can be rapidly executed at the close price[cite: 126]. [cite_start]We assume that stock market will not be affected by our reinforcement trading agent[cite: 127].
* [cite_start]**Nonnegative balance $b>0$**; the allowed actions should not result in a negative balance[cite: 128]. [cite_start]Based on the action at time t, the stocks are divided into sets for sell S, buying B, and holding H, where $S\cup\mathfrak{B\cup\mathcal{H}=\{1,\cdot\cdot\cdot,D\}$ and they are nonoverlapping[cite: 129]. [cite_start]Let $p_{t}^{B}=[p_{t}^{i}:i\in\mathcal{B}]$ and $k_{t}^{B}=[k_{t}^{i}:i\in\mathcal{B}]$ be the vectors of price and number of buying shares for the stocks in the buying set[cite: 130]. [cite_start]We can similarly define $p_{t}^{S}$ and $k_{t}^{S}$ for the selling stocks, and $p_{t}^{H}$ and $k_{t}^{H}$ for the holding stocks[cite: 131]. [cite_start]Hence, the constraint for non-negative balance can be expressed as [cite: 132]
    [cite_start]$b_{t+1}=b_{t}+(p_{t}^{S})^{T}k_{t}^{S}-(p_{t}^{B})^{T}k_{t}^{B}\ge0.$ (1) [cite: 133]
* [cite_start]**Transaction cost**: transaction costs are incurred for each trade[cite: 166]. [cite_start]There are many types of transaction costs such as exchange fees, execution fees, and SEC fees[cite: 167]. [cite_start]Different brokers have different commission fees[cite: 168]. [cite_start]Despite these variations in fees, we assume our transaction costs to be 0.1% of the value of each trade (either buy or sell) as in [45]: [cite: 168]
    $c_{t}=p^{T}k_{t}\times0.1\%$. (2) [cite_start][cite: 170]
* [cite_start]**Risk-aversion for market crash**: there are sudden events that may cause stock market crash, such as wars, collapse of stock market bubbles, sovereign debt default, and financial crisis[cite: 136]. [cite_start]To control the risk in a worst-case scenario like 2008 global financial crisis, we employ the financial turbulence index turbulence, that measures extreme asset price movements [25]: [cite: 137]
    [cite_start]$turbulence_{t}=(y_{t}-\mu)\Sigma^{-1}(y_{t}-\mu)^{\prime}\in\mathbb{R}$ (3) [cite: 138]
    [cite_start]where $y_{t}\in\mathbb{R}^{D}$ denotes the stock returns for current period t, $\mu\in\mathbb{R}^{D}$ denotes the average of historical returns, and $\Sigma\in\mathbb{R}^{D\times D}$ denotes the covariance of historical returns[cite: 140]. [cite_start]When turbulence, is higher than a threshold, which indicates extreme market conditions, we simply halt buying and the trading agent sells all shares[cite: 141]. [cite_start]We resume trading once the turbulence index returns under the threshold[cite: 142].

[cite_start]**3.3 Return Maximization as Trading Goal** [cite: 143]

[cite_start]We define our reward function as the change of the portfolio value when action a is taken at state s and arriving at new state $s^{\prime}$[cite: 143]. [cite_start]The goal is to design a trading strategy that maximizes the change of the portfolio value: [cite: 144]
[cite_start]$r(s_{t},a_{t},s_{t+1})=(b_{t+1}+p_{t+1}^{T}h_{t+1})-(b_{t}+p_{t}^{T}h_{t})-c_{t}$ (4) [cite: 145]
[cite_start]where the first and second terms denote the portfolio value at t+1 and t, respectively[cite: 146]. [cite_start]To further decompose the return, we define the transition of the shares $h_{t}$ is defined as [cite: 147]
[cite_start]$h_{t+1}=h_{t}-k_{t}^{S}+k_{t}^{B},$ (5) [cite: 148]
[cite_start]and the transition of the balance $b_{t}$ is defined in (1)[cite: 150]. [cite_start]Then (4) can be rewritten as [cite: 151]
[cite_start]$r(s_{t},a_{t},s_{t+1})=r_{H}-r_{S}+r_{B}-c_{t},$ (6) [cite: 152]
[cite_start]where [cite: 151]
[cite_start]$r_{H}=(p_{t+1}^{H}-p_{t}^{H})^{T}h_{t}^{H}$ (7) [cite: 154]
[cite_start]$r_{S}=(p_{t+1}^{S}-p_{t}^{S})^{T}h_{t}^{S}$ (8) [cite: 156]
[cite_start]$r_{B}=(p_{t+1}^{B}-p_{t}^{B})^{T}h_{t}^{B}$ (9) [cite: 158]
[cite_start]where $r_{H}$, $r_{S}$ and $r_{B}$ denote the change of the portfolio value comes from holding, selling, and buying shares moving from time t to t+1, respectively[cite: 160]. [cite_start]Equation (6) indicates that we need to maximize the positive change of the portfolio value by buying and holding the stocks whose price will increase at next time step and minimize the negative change of the portfolio value by selling the stocks whose price will decrease at next time step[cite: 161].

[cite_start]Turbulence index turbulence, is incorporated with the reward function to address our risk-aversion for market crash[cite: 162]. [cite_start]When the index in (3) goes above a threshold, Equation (8) becomes [cite: 163]
[cite_start]$r_{sell}=(p_{t+1}-p_{t})^{T}k_{t},$ (10) [cite: 164]
[cite_start]which indicates that we want to minimize the negative change of the portfolio value by selling all held stocks, because all stock prices will fall[cite: 169].

[cite_start]The model is initialized as follows[cite: 172]. [cite_start]$p_{0}$ is set to the stock prices at time 0 and $b_{0}$ is the amount of initial fund[cite: 172]. [cite_start]The hand $Q_{\pi}(s,a)$ are 0, and $\pi(s)$ is uniformly distributed among all actions for each state[cite: 173]. [cite_start]Then, $Q_{\pi}(s_{t},a_{t})$ is updated through interacting with the stock market environment[cite: 175]. [cite_start]The optimal strategy is given by the Bellman Equation, such that the expected reward of taking action $a_{t}$ at state $s_{t}$ is the expectation of the summation of the direct reward $r(s_{t},a_{t},s_{t+1})$ and the future reward in the next state $s_{t+1}$[cite: 176]. [cite_start]Let the future rewards be discounted by a factor of $0<\gamma<1$ for convergence purpose, then we have [cite: 176]
$Q_{\pi}(s_{t},a_{t})=\mathbb{B}_{s_{t+1}}[r(s_{t},a_{t},s_{t+1})+\gamma\mathbb{B}_{a_{t+1}-\pi(s_{t+1})}[Q_{\pi}(s_{t+1},a_{t+1})]]$. (11) [cite_start][cite: 177]
[cite_start]The goal is to design a trading strategy that maximizes the positive cumulative change of the portfolio value $r(s_{t},a_{t},s_{t+1})$ in the dynamic environment, and we employ the deep reinforcement learning method to solve this problem[cite: 178].

### [cite_start]**4 STOCK MARKET ENVIRONMENT** [cite: 179]

[cite_start]Before training a deep reinforcement trading agent, we carefully build the environment to simulate real world trading which allows the agent to perform interaction and learning[cite: 180]. [cite_start]In practical trading, various information needs to be taken into account, for example the historical stock prices, current holding shares, technical indicators, etc[cite: 181]. [cite_start]Our trading agent needs to obtain such information through the environment, and take actions defined in the previous section[cite: 181]. [cite_start]We employ OpenAI gym to implement our environment and train the agent [6, 14, 19][cite: 182].

[cite_start]**4.1 Environment for Multiple Stocks** [cite: 183]

[cite_start]We use a continuous action space to model the trading of multiple stocks[cite: 184]. [cite_start]We assume that our portfolio has 30 stocks in total[cite: 185].

[cite_start]**4.1.1 State Space.** [cite: 186] [cite_start]We use a 181-dimensional vector consists of seven parts of information to represent the state space of multiple stocks trading environment: $[b_{t},p_{t},h_{t},M_{t},R_{t},C_{t},X_{t}]$[cite: 186]. [cite_start]Each component is defined as follows: [cite: 186]
* [cite_start]$b_{t}\in\mathbb{R}_{+}$: available balance at current time step t[cite: 187].
* [cite_start]$p_{t}\in\mathbb{R}_{+}^{30}$: adjusted close price of each stock[cite: 187].
* [cite_start]$h_{t}\in\mathbb{Z}_{+}^{30}$: shares owned of each stock[cite: 188].
* [cite_start]$M_{t}\in\mathbb{R}^{30}$: Moving Average Convergence Divergence (MACD) is calculated using close price[cite: 189]. [cite_start]MACD is one of the most commonly used momentum indicator that identifies moving averages [11][cite: 190].
* [cite_start]$R_{t}\in\mathbb{R}_{+}^{30}$: Relative Strength Index (RSI) is calculated using close price[cite: 191]. [cite_start]RSI quantifies the extent of recent price changes[cite: 191]. [cite_start]If price moves around the support line, it indicates the stock is oversold, and we can perform the buy action[cite: 192]. [cite_start]If price moves around the resistance, it indicates the stock is overbought, and we can perform the selling action [11][cite: 193].
* [cite_start]$C_{t}\in\mathbb{R}_{+}^{30}$: Commodity Channel Index (CCI) is calculated using high, low and close price[cite: 194]. [cite_start]CCI compares current price to average price over a time window to indicate a buying or selling action [30][cite: 195].
* [cite_start]$X_{t}\in\mathbb{R}^{30}$: Average Directional Index (ADX) is calculated using high, low and close price[cite: 196]. [cite_start]ADX identifies trend strength by quantifying the amount of price movement [18][cite: 197].

[cite_start]**4.1.2 Action Space.** [cite: 198] [cite_start]For a single stock, the action space is defined as $\{-k,...,-1,0,1,...,k\}$ where k and -k presents the number of shares we can buy and sell, and $k\le h_{max}$ while $h_{max}$ is a predefined parameter that sets as the maximum amount of shares for each buying action[cite: 198]. [cite_start]Therefore the size of the entire action space is $(2k+1)^{30}$[cite: 217]. [cite_start]The action space is then normalized to [-1, 1], since the RL algorithms A2C and PPO define the policy directly on a Gaussian distribution, which needs to be normalized and symmetric [19][cite: 217].

[cite_start]**4.2 Memory Management** [cite: 218]

[cite_start]The memory consumption for training could grow exponentially with the number of stocks, data types, features of the state space, number of layers and neurons in the neural networks, and batch size[cite: 219]. [cite_start]To tackle the problem of memory requirements, we employ a load-on-demand technique for efficient use of memory[cite: 220]. [cite_start]As shown in Figure 3, the load-on-demand technique does not store all results in memory, rather, it generates them on demand[cite: 221]. [cite_start]The memory is only used when the result is requested, hence the memory usage is reduced[cite: 222].

### [cite_start]**5 TRADING AGENT BASED ON DEEP REINFORCEMENT LEARNING** [cite: 223]

[cite_start]We use three actor-critic based algorithms to implement our trading agent[cite: 224]. [cite_start]The three algorithms are A2C, DDPG, and PPO, respectively[cite: 224]. [cite_start]An ensemble strategy is proposed to combine the three agents together to build a robust trading strategy[cite: 225].

[cite_start]**5.1 Advantage Actor Critic (A2C)** [cite: 226]

[cite_start]A2C [32] is a typical actor-critic algorithm and we use it a component in the ensemble strategy[cite: 227]. [cite_start]A2C is introduced to improve the policy gradient updates[cite: 228]. [cite_start]A2C utilizes an advantage function to reduce the variance of the policy gradient[cite: 228]. [cite_start]Instead of only estimates the value function, the critic network estimates the advantage function[cite: 229]. [cite_start]Thus, the evaluation of an action not only depends on how good the action is, but also considers how much better it can be[cite: 230]. [cite_start]So that it reduces the high variance of the policy network and makes the model more robust[cite: 231]. [cite_start]A2C uses copies of the same agent to update gradients with different data samples[cite: 232]. [cite_start]Each agent works independently to interact with the same environment[cite: 232, 234]. [cite_start]In each iteration, after all agents finish calculating their gradients, A2C uses a coordinator to pass the average gradients over all the agents to a global network[cite: 234]. [cite_start]So that the global network can update the actor and the critic network[cite: 235]. [cite_start]The presence of a global network increases the diversity of training data[cite: 236]. [cite_start]The synchronized gradient update is more cost-effective, faster and works better with large batch sizes[cite: 237]. [cite_start]A2C is a great model for stock trading because of its stability[cite: 238]. [cite_start]The objective function for A2C is: [cite: 239]
[cite_start]$\nabla J_{\theta}(\theta)=\mathbb{E}[\sum_{t=1}^{T}\nabla_{\theta}log~\pi_{\theta}(a_{t}|s_{t})A(s_{t},a_{t})].$ (12) [cite: 240, 248]
[cite_start]where $\pi_{\theta}(a_{t}|s_{t})$ is the policy network, $A(s_{t},a_{t})$ is the Advantage function can be written as: [cite: 250]
[cite_start]$A(s_{t},a_{t})=Q(s_{t},a_{t})-V(s_{t})$ (13) [cite: 252, 257]
[cite_start]or [cite: 251]
[cite_start]$A(s_{t},a_{t})=r(s_{t},a_{t},s_{t+1})+yV(s_{t+1})-V(s_{t}).$ (14) [cite: 253, 258]

[cite_start]**5.2 Deep Deterministic Policy Gradient (DDPG)** [cite: 261]

[cite_start]DDPG [29] is used to encourage maximum investment return[cite: 261]. [cite_start]DDPG combines the frameworks of both Q-learning [40] and policy gradient [41], and uses neural networks as function approximators[cite: 262]. [cite_start]In contrast with DQN that learns indirectly through Q-values tables and suffers the curse of dimensionality problem [8], DDPG learns directly from the observations through policy gradient[cite: 263]. [cite_start]It is proposed to deterministically map states to actions to better fit the continuous action space environment[cite: 264]. [cite_start]At each time step, the DDPG agent performs an action $a_{t}$ at $s_{t}$, receives a reward $r_{t}$ and arrives at $s_{t+1}$[cite: 265]. [cite_start]The transitions ($(s_{t},a_{t},s_{t+1},r_{t})$) are stored in the replay buffer R[cite: 266]. [cite_start]A batch of N transitions are drawn from R and the Q-value $y_{i}$ is updated as: [cite: 266]
[cite_start]$y_{i} = r_{i}+\gamma Q^{\prime}(s_{i+1}, \mu^{\prime}(s_{i+1}|\theta^{\mu^{\prime}})), i = 1,..., N.$ (15) [cite: 267]
[cite_start]The critic network is then updated by minimizing the loss function $L(\theta^{Q})$ which is the expected difference between outputs of the target critic network $Q^{\prime}$ and the critic network Q, i.e, [cite: 267]
$L(\theta^{Q})=\mathbb{B}_{s_{t},a_{t},r_{t},s_{t+1}\sim buffer}[(y_{i}-Q(s_{t},a_{t}|\theta^{Q}))^{2}]$. (16) [cite_start][cite: 268]
[cite_start]DDPG is effective at handling continuous action space, and so it is appropriate for stock trading[cite: 268].

[cite_start]**5.3 Proximal Policy Optimization (PPO)** [cite: 269]

[cite_start]We explore and use PPO as a component in the ensemble method[cite: 270]. [cite_start]PPO [37] is introduced to control the policy gradient update and ensure that the new policy will not be too different from the previous one[cite: 271]. [cite_start]PPO tries to simplify the objective of Trust Region Policy Optimization (TRPO) by introducing a clipping term to the objective function [36, 37][cite: 272]. [cite_start]Let us assume the probability ratio between old and new policies is expressed as: [cite: 273]
[cite_start]$r_{t}(\theta) = \frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{old}}(a_{t}|s_{t})}$ (17) [cite: 274, 277]
[cite_start]The clipped surrogate objective function of PPO is: [cite: 275]
[cite_start]$J^{CLIP}(\theta)=\hat{B}_{t}[min(r_{t}(\theta)\hat{A}(s_{t},a_{t}), clip(r_{t}(\theta), 1-\epsilon, 1 + \epsilon)\hat{A}(s_{t}, a_{t}))],$ (18) [cite: 276, 278]
[cite_start]where $r_{t}(\theta)\hat{A}(s_{t},a_{t})$ is the normal policy gradient objective, and $\hat{A}(s_{t},a_{t})$ is the estimated advantage function[cite: 242]. [cite_start]The function $clip(r_{t}(\theta),1-\epsilon,1+\epsilon)$ clips the ratio $r_{t}(\theta)$ to be within $[1-\epsilon, 1+\epsilon]$[cite: 243]. [cite_start]The objective function of PPO takes the minimum of the clipped and normal objective[cite: 244]. [cite_start]PPO discourages large policy change move outside of the clipped interval[cite: 245]. [cite_start]Therefore, PPO improves the stability of the policy networks training by restricting the policy update at each training step[cite: 246]. [cite_start]We select PPO for stock trading because it is stable, fast, and simpler to implement and tune[cite: 247].

[cite_start]**5.4 Ensemble Strategy** [cite: 249]

[cite_start]Our purpose is to create a highly robust trading strategy[cite: 254]. [cite_start]So we use an ensemble strategy to automatically select the best performing agent among PPO, A2C, and DDPG to trade based on the Sharpe ratio[cite: 255]. [cite_start]The ensemble process is described as follows: [cite: 256]
* [cite_start]**Step 1.** We use a growing window of n months to retrain our three agents concurrently[cite: 259]. [cite_start]In this paper we retrain our three agents at every three months[cite: 260].
* [cite_start]**Step 2.** We validate all three agents by using a 3-month validation rolling window after training window to pick the best performing agent with the highest Sharpe ratio [39][cite: 279]. [cite_start]The Sharpe ratio is calculated as: [cite: 280]
    [cite_start]Sharpe ratio = $\frac{\overline{r}_{p}-r_{f}}{\sigma_{p}}$ (19) [cite: 281, 283]
    [cite_start]where $\overline{r}_{p}$ is the expected portfolio return, $r_{f}$ is the risk free rate, and $\sigma_{p}$ is the portfolio standard deviation[cite: 284]. [cite_start]We also adjust risk-aversion by using turbulence index in our validation stage[cite: 285].
* [cite_start]**Step 3.** After the best agent is picked, we use it to predict and trade for the next quarter[cite: 286].

[cite_start]The reason behind this choice is that each trading agent is sensitive to different type of trends[cite: 287]. [cite_start]One agent performs well in a bullish trend but acts bad in a bearish trend[cite: 288]. [cite_start]Another agent is more adjusted to a volatile market[cite: 289]. [cite_start]The higher an agent’s Sharpe ratio, the better its returns have been relative to the amount of investment risk it has taken[cite: 289]. [cite_start]Therefore, we pick the trading agent that can maximize the returns adjusted to the increasing risk[cite: 290].

### [cite_start]**6 PERFORMANCE EVALUATIONS** [cite: 291]

[cite_start]In this section, we present the performance evaluation of our proposed scheme[cite: 292]. [cite_start]We perform backtesting for the three individual agents and our ensemble strategy[cite: 293]. [cite_start]The result in Table 2 demonstrates that our ensemble strategy achieves higher Sharpe ratio than the three agents, Dow Jones Industrial Average and the traditional min-variance portfolio allocation strategy[cite: 294]. [cite_start]Our codes are available on Github 2[cite: 295].

[cite_start]**6.1 Stock Data Preprocessing** [cite: 296]

[cite_start]We select the Dow Jones 30 constituent stocks (at 01/01/2016) as our trading stock pool[cite: 297]. [cite_start]Our backtestings use historical daily data from 01/01/2009 to 05/08/2020 for performance evaluation[cite: 298]. [cite_start]The stock data can be downloaded from the Compustat database through the Wharton Research Data Services (WRDS) [38][cite: 299]. [cite_start]Our dataset consists of two periods: in-sample period and out-of-sample period[cite: 300]. [cite_start]In-sample period contains data for training and validation stages[cite: 301]. [cite_start]Out-of-sample period contains data for trading stage[cite: 317]. [cite_start]In the training stage, we train three agents using PPO, A2C, and DDPG, respectively[cite: 317]. [cite_start]Then, a validation stage is then carried out for validating the 3 agents by Sharpe ratio, and adjusting key parameters, such as learning rate, number of episodes, etc[cite: 318]. [cite_start]Finally, in the trading stage, we evaluate the profitability of each of the algorithms[cite: 318]. [cite_start]The whole dataset is split as shown in Figure 4[cite: 319]. [cite_start]Data from 01/01/2009 to 09/30/2015 is used for training, and the data from 10/01/2015 to 12/31/2015 is used for validation and tuning of parameters[cite: 319]. [cite_start]Finally, we test our agent’s performance on trading data, which is the unseen out-of-sample data from 01/01/2016 to 05/08/2020[cite: 320]. [cite_start]To better exploit the trading data, we continue training our agent while in the trading stage, since this will help the agent to better adapt to the market dynamics[cite: 321].

[cite_start]**6.2 Performance Comparisons** [cite: 322]

[cite_start]**6.2.1 Agent Selection.** [cite: 323] [cite_start]From Table 1, we can see that PPO has the best validation Sharpe ratio of 0.06 from 2015/10 to 2015/12, so we use PPO to trade for the next quarter from 2016/01 to 2016/03[cite: 323]. [cite_start]DDPG has the best validation Sharpe ratio of 0.61 from 2016/01 to 2016/03, so we use DDPG to trade for the next quarter from 2016/04 to 2016/06[cite: 324]. [cite_start]A2C has the best validation Sharpe ratio of -0.15 from 2020/01 to 2020/03, so we use A2C to trade for the next quarter from 2020/04 to 2020/05[cite: 325].

[cite_start]Five metrics are used to evaluate our results: [cite: 326]
* [cite_start]**Cumulative return**: is calculated by subtracting the portfolio’s final value from its initial value, and then dividing by the initial value[cite: 327].
* [cite_start]**Annualized return**: is the geometric average amount of money earned by the agent each year over the time period[cite: 328].
* [cite_start]**Annualized volatility**: is the annualized standard deviation of portfolio return[cite: 329].
* [cite_start]**Sharpe ratio**: is calculated by subtracting the annualized risk free rate from the annualized return, and the dividing by the annualized volatility[cite: 330].
* [cite_start]**Max drawdown**: is the maximum percentage loss during the trading period[cite: 331].

[cite_start]Cumulative return reflects returns at the end of trading stage[cite: 332]. [cite_start]Annualized return is the return of the portfolio at the end of each year[cite: 333]. [cite_start]Annualized volatility and max drawdown measure the robustness of a model[cite: 334]. [cite_start]The Sharpe ratio is a widely used metric that combines the return and risk together[cite: 335].

[cite_start]**Table 1: Sharpe Ratios over time.** [cite: 313]

| Trading Quarter | PPO | A2C | DDPG | Picked Model |
| :--- | :--- | :--- | :--- | :--- |
| 2016/01-2016/03 | 0.06 | 0.03 | 0.05 | PPO |
| 2016/04-2016/06 | 0.31 | 0.53 | 0.61 | DDPG |
| 2016/07-2016/09 | -0.02 | 0.01 | 0.05 | DDPG |
| 201610-2016//12 | 0.11 | 0.01 | 0.09 | PPO |
| 2017/01-2017/03 | 0.53 | 0.44 | 0.13 | PPO |
| 2017/04-2017/06 | 0.29 | 0.44 | 0.12 | A2C |
| 2017/07-2017/09 | 0.4 | 0.32 | 0.15 | PPO |
| 201710-2017//12 | -0.05 | -0.04 | 0.12 | DDPG |
| 2018/01-2018/03 | 0.71 | 0.63 | 0.62 | PPO |
| 2018/04-2018/06 | -0.08 | -0.02 | -0.01 | DDPG |
| 2018/07-2018/09 | -0.17 | 0.21 | -0.03 | A2C |
| 201810-2018//12 | 0.30 | 0.48 | 0.39 | A2C |
| 2019/01-2019/03 | -0.26 | -0.25 | -0.18 | DDPG |
| 2019/04-2019/06 | 0.38 | 0.29 | 0.25 | PPO |
| 2019/07-2019/09 | 0.53 | 0.47 | 0.52 | PPO |
| 2019/10-2019/12 | -0.22 | 0.11. | -0.22 | A2C |
| 2020/01-2020/03 | -0.36 | -0.13 | -0.22 | A2C |
| 2020/04-2020/05 | -0.42 | -0.15 | -0.58 | A2C |
[cite_start][cite: 314]

[cite_start]**6.2.2 Analysis of Agent Performance.** [cite: 336] [cite_start]From both Table 2 and Figure 5, we can observe that the A2C agent is more adaptive to risk[cite: 336]. [cite_start]It has the lowest annual volatility 10.4% and max drawdown -10.2% among the three agents[cite: 337]. [cite_start]So A2C is good at handling a bearish market[cite: 338]. [cite_start]PPO agent is good at following trend and acts well in generating more returns, it has the highest annual return 15.0% and cumulative return 83.0% among the three agents[cite: 339]. [cite_start]So PPO is preferred when facing a bullish market[cite: 340]. [cite_start]DDPG performs similar but not as good as PPO, it can be used as a complementary strategy to PPO in a bullish market[cite: 340]. [cite_start]All three agents’ performance outperform the two benchmarks, Dow Jones Industrial Average and min-variance portfolio allocation of DJIA, respectively[cite: 341].

[cite_start]**6.2.3 Performance under Market Crash.** [cite: 342] [cite_start]In Figure 6, we can see that our ensemble strategy and the three agents perform well in the 2020 stock market crash event[cite: 342]. [cite_start]When the turbulence index reaches a threshold, it indicates an extreme market situation[cite: 343]. [cite_start]Then our agents will sell off all currently held shares and wait for the market to return to normal to resume trading[cite: 344]. [cite_start]By incorporating the turbulence index, the agents are able to cut losses and successfully survive the stock market crash in March 2020[cite: 345]. [cite_start]We can tune the turbulence index threshold lower for higher risk aversion[cite: 345].

[cite_start]**6.2.4 Benchmark Comparison.** [cite: 346] [cite_start]Figure 5 demonstrates that our ensemble strategy significantly outperforms the DJIA and the min-variance portfolio allocation [45][cite: 346]. [cite_start]As can be seen from Table 2, the ensemble strategy achieves a Sharpe ratio 1.30, which is much higher than the Sharpe ratio of 0.47 for DJIA, and 0.45 for the min-variance portfolio allocation[cite: 347]. [cite_start]The annualized return of the ensemble strategy is also much higher, the annual volatility is much lower, indicating that the ensemble strategy beats both the DJIA and min-variance portfolio allocation in balancing risk and return[cite: 348]. [cite_start]The ensemble strategy also outperforms A2C with a Sharpe ratio of 1.12, PPO with a Sharpe ratio of 1.10, and DDPG with a Sharpe ratio of 0.87, respectively[cite: 349]. [cite_start]Therefore, our findings demonstrate that the proposed ensemble strategy can effectively develop a trading strategy that outperforms the three individual algorithms and the two baselines[cite: 350, 438].

[cite_start]**Table 2: Performance evaluation comparison.** [cite: 377]

| (2016/01/04-2020/05/08) | Ensemble (Ours) | PPO | A2C | DDPG | Min-Variance | DJIA |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cumulative Return** | [cite_start]70.4% [cite: 384] | [cite_start]83.0% [cite: 385] | [cite_start]60.0% [cite: 386] | [cite_start]54.8% [cite: 387] | [cite_start]31.7% [cite: 388] | [cite_start]38.6% [cite: 389] |
| **Annual Return** | [cite_start]13.0% [cite: 391] | [cite_start]15.0% [cite: 392] | [cite_start]11.4% [cite: 393] | [cite_start]10.5% [cite: 394] | [cite_start]6.5% [cite: 395] | [cite_start]7.8% [cite: 396] |
| **Annual Volatility** | [cite_start]9.7% [cite: 398] | [cite_start]13.6% [cite: 399] | [cite_start]10.4% [cite: 400] | [cite_start]12.3% [cite: 401] | [cite_start]17.8% [cite: 402] | [cite_start]20.1% [cite: 403] |
| **Sharpe Ratio** | [cite_start]1.30 [cite: 405] | [cite_start]1.10 [cite: 406] | [cite_start]1.12 [cite: 407] | [cite_start]0.87 [cite: 408] | [cite_start]0.45 [cite: 409] | [cite_start]0.47 [cite: 410] |
| **Max Drawdown** | [cite_start]-9.7% [cite: 412] | [cite_start]-23.7% [cite: 413] | [cite_start]-10.2% [cite: 413] | [cite_start]-14.8% [cite: 414] | [cite_start]-34.3% [cite: 415] | [cite_start]-37.1% [cite: 416] |

### [cite_start]**7 CONCLUSION** [cite: 441, 442]

[cite_start]In this paper, we have explored the potential of using actor-critic based algorithms which are Proximal Policy Optimization (PPO), Advantage Actor Critic (A2C), and Deep Deterministic Policy Gradient (DDPG) agents to learn stock trading strategy[cite: 439, 443]. [cite_start]In order to adjust to different market situations, we use an ensemble strategy to automatically select the best performing agent to trade based on the Sharpe ratio[cite: 440, 444]. [cite_start]Results show that our ensemble strategy outperforms the three individual algorithms, the Dow Jones Industrial Average and min-variance portfolio allocation method in terms of Sharpe ratio by balancing risk and return under transaction costs[cite: 444, 445, 448].

[cite_start]For future work, it will be interesting to explore more sophisticated model [42], solve empirical challenges [15], deal with large-scale data [7] such as S&P 500 constituent stocks[cite: 449]. [cite_start]We can also explore more features for the state space such as adding advanced transaction cost and liquidity model [1], incorporating fundamental analysis indicators [45], natural language processing analysis of financial market news [27], and ESG features [10] to our observations[cite: 450]. [cite_start]We are interested in directly using Sharpe ratio as the reward function, but the agents need to observe a lot more historical data, the state space will increase exponentially[cite: 451].

### [cite_start]**REFERENCES** [cite: 452]
[1] Wenhang Bao and Xiao-Yang Liu. 2019. [cite_start]Multi-agent deep reinforcement learning for liquidation strategy analysis. [cite: 453] [cite_start]ICML Workshop on Applications and Infrastructure for Multi-Agent Learning, 2019 (06 2019)[cite: 454].
[2] Stelios Bekiros. 2010. [cite_start]Heterogeneous trading strategies with adaptive fuzzy Actor-Critic reinforcement learning: A behavioral approach. [cite: 455] [cite_start]Journal of Economic Dynamics and Control 34 (06 2010), 1153-1170[cite: 456].
[3] Stelios D. Bekiros. 2010. [cite_start]Fuzzy adaptive decision-making for boundedly rational traders in speculative stock markets. [cite: 457] [cite_start]European Journal of Operational Research 202, 1 (April 2010), 285-293[cite: 458].
[4] Francesco Bertoluzzo and Marco Corazza. 2012. [cite_start]Testing different reinforcement learning configurations for financial trading: introduction and applications. [cite: 459] [cite_start]Procedia Economics and Finance 3 (12 2012), 68-77[cite: 460].
[5] Dimitri Bertsekas. 1995. Dynamic programming and optimal control. Vol. [cite_start]1[cite: 461].
[cite_start][6] Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba 2016. OpenAl Gym. [cite: 462] [cite_start]arXiv:arXiv:1606.01540[cite: 463].
[7] Yuri Burda, Harrison Edwards, Deepak Pathak, Amos Storkey, Trevor Darrell, and Alexei Efros. 2018. [cite_start]Large-scale study of curiosity-driven learning. [cite: 464] [cite_start]In 2019 Seventh International Conference on Learning Representations (ICLR) Poster[cite: 465].
[cite_start][8] Lucian Busoniu, Tim de Bruin, Domagoj Tolić, Jens Kober, and Ivana Palunko. [cite: 466] 2018. Reinforcement learning for control: Performance, stability, and deep approximators. [cite_start]Annual Reviews in Control (10 2018)[cite: 467].
[9] Lin Chen and Qiang Gao. 2019. [cite_start]Application of deep reinforcement learning on automated stock trading. [cite: 468] In 2019 IEEE 10th International Conference on Software Engineering and Service Science (ICSESS). [cite_start]29-33[cite: 469].
[10] Qian Chen and Xiao-Yang Liu. 2020. [cite_start]Quantifying ESG alpha using scholar big data: An automated machine learning approach. [cite: 470] [cite_start]ACM International Conference on Al in Finance, ICAIF 2020 (2020)[cite: 471].
[11] Terence Chong, Wing-Kam Ng, and Venus Liew. 2014. [cite_start]Revisiting the performance of MACD and RSI oscillators. [cite: 472] [cite_start]Journal of Risk and Financial Management 7 (03) 2014), 1-12[cite: 473].
[12] Quang-Vinh Dang, 2020. Reinforcement learning in stock trading. [cite_start]In Advanced Computational Methods for Knowledge Engineering. [cite: 474] [cite_start]ICCSAMA 2019. Advances in Intelligent Systems and Computing, vol 1121. Springer, Cham[cite: 475].
[cite_start][13] Yue Deng, Feng Bao, Youyong Kong, Zhiquan Ren, and Qionghai Dai. [cite: 476] 2016. [cite_start]Deep direct reinforcement learning for financial signal representation and trading. [cite: 477] [cite_start]IEEE Transactions on Neural Networks and Learning Systems 28 (02 2016), 1-12[cite: 478].
[cite_start][14] Prafulla Dhariwal, Christopher Hesse, Oleg Klimov, Alex Nichol, Matthias Plappert, Alec Radford, John Schulman, Szymon Sidor, Yuhuai Wu, and Peter Zhokhov. [cite: 479] 2017. [cite_start]OpenAl Baselines. [https://github.com/openai/baselines](https://github.com/openai/baselines)[cite: 480].
[cite_start][15] Gabriel Dulac-Arnold, N. Levine, Daniel J. Mankowitz, J. Li, Cosmin Paduraru, Sven Gowal, and T. Hester. [cite: 481] 2020. An empirical investigation of the challenges of real-world reinforcement learning. [cite_start]ArXiv abs/2003.11881 (2020)[cite: 482],
[16] Yunzhe Fang, Xiao-Yang Liu, and Hongyang Yang. 2019. [cite_start]Practical machine learning approach to capture the scholar data driven alpha in Al industry. [cite: 483] In 2019 IEEE International Conference on Big Data (Big Data) Special Session on Intelligent Data Mining. [cite_start]2230-2239[cite: 484].
[17] Thomas G. Fischer. 2018. [cite_start]Reinforcement learning in financial markets - a survey. [cite: 485] FAU Discussion Papers in Economics 12/2018. [cite_start]Friedrich-Alexander University Erlangen-Nuremberg, Institute for Economics[cite: 486].
[18] Ikhlaas Gurrib. 2018. [cite_start]Performance of the average directional index as a market timing tool for the most actively traded USD based currency pairs. [cite: 487] [cite_start]Banks and Bank Systems 13 (08.2018), 58-70[cite: 488].
[cite_start][19] Ashley Hill, Antonin Raffin, Maximilian Ernestus, Adam Gleave, Anssi Kanervisto, Rene Traore, Prafulla Dhariwal, Christopher Hesse, Oleg Klimov, Alex Nichol, Matthias Plappert, Alec Radford, John Schulman, Szymon Sidor, and Yuhuai Wu. [cite: 489] 2018. [cite_start]Stable baselines. [https://github.com/hill-a/stable-baselines](https://github.com/hill-a/stable-baselines)[cite: 490].
[20] A. Ilmanen. 2012. Expected Returns: An Investor's Guide to Harvesting Market Rewards. (05 2012) [cite_start][cite: 492].
[21] Gyeeun Jeong and Ha Kim. 2018. [cite_start]Improving financial trading decisions using deep Q-learning: predicting the number of shares, action strategies, and transfer leaming. [cite: 493] [cite_start]Expert Systems with Applications 117 (09 2018)[cite: 494].
[22] Zhengyao Jiang and Jinjun Liang. 2017. Cryptocurrency portfolio management with deep reinforcement learning. [cite_start]In 2017 Intelligent Systems Conference[cite: 495].
[cite_start][23] Youngmin Kim, Wonbin Ahn, Kyong Joo Oh, and David Enke. [cite: 496] 2017. [cite_start]An intelligent hybrid trading system for discovering trading rules for the futures market using rough sets and genetic algorithms. [cite: 497] [cite_start]Applied Soft Computing 55 (02 2017), 127-140[cite: 498].
[cite_start][24] Vijay Konda and John Tsitsiklis, 2001. Actor-critic algorithms. [cite: 498] [cite_start]Society for Industrial and Applied Mathematics 42 (04 2001)[cite: 499].
[25] Mark Kritzman and Yuanzhen Li. 2010, Skulls, financial turbulence, and risk management. [cite_start]Financial Analysts Journal 66 (10 2010)[cite: 500].
[26] Jinke Li, Ruonan Rao, and Jun Shi. 2018. [cite_start]Learning to Trade with Deep Actor Critic Methods. [cite: 501] [cite_start]2018 11th International Symposium on Computational Intelligence and Design (ISCID) 02 (2018), 66-71[cite: 502].
[cite_start][27] Xinyi Li, Yinchuan Li, Hongyang Yang, Liuqing Yang, and Xiao-Yang Liu. [cite: 503] 2019. [cite_start]DP-LSTM: Differential privacy-inspired LSTM for stock prediction using financial news. [cite: 504] [cite_start]33rd Conference on Neural Information Processing Systems (NeurIPS 2019) Workshop on Robust Al in Financial Services: Data, Fairness, Explainability, Trustworthiness, and Privacy, December 2019 (12 2019)[cite: 505].
[28] Zhipeng Liang, Kangkang Jiang, Hao Chen, Junhao Zhu, and Yanran Li. 2018. [cite_start]Adversarial deep reinforcement learning in portfolio management. [cite: 506] [cite_start]arXiv: Portfolio Management (2018)[cite: 507].
[cite_start][29] Timothy Lillicrap, Jonathan Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. [cite: 508] 2015. Continuous control with deep reinforcement learning. [cite_start]International Conference on Learning Representations (ICLR) 2016 (09 2015)[cite: 509].
[cite_start][30] Mansoor Maitah, Petr Procházka, Michal Čermák, and Karel Šrédl. [cite: 510] 2016. [cite_start]Commodity Channel index: evaluation of trading rule of agricultural Commodities. [cite: 511] [cite_start]International Journal of Economics and Financial Issues 6 (03 2016), 176-178[cite: 512].
[31] Harry Markowitz. 1952. Portfolio selection. [cite_start]Journal of Finance 7, 1 (1952), 77-91[cite: 513].
[cite_start][32] Volodymyr Mnih, Adrià Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. [cite: 514] 2016. Asynchronous methods for deep reinforcement learning. [cite_start]The 33rd International Conference on Machine Learning (02 2016)[cite: 515].
[33] John Moody and Matthew Saffell. 2001. [cite_start]Learning to trade via direct reinforcement. [cite: 516] [cite_start]IEEE Transactions on Neural Networks 12 (07 2001), 875-89[cite: 517].
[34] Ralph Neuneier. 1996. [cite_start]Optimal asset allocation using adaptive dynamic programming. [cite: 518] [cite_start]Conference on Neural Information Processing Systems, 1995 (05 1996)[cite: 519].
[35] Ralph Neuneier. 1997. Enhancing Q-learning for optimal asset allocation. [cite_start]Conference on Neural Information Processing Systems (NeurIPS), 1997[cite: 520].
[36] John Schulman, Sergey Levine, Philipp Moritz, Michael Jordan, and Pieter Abbeel. 2015. [cite_start]Trust region policy optimization. [cite: 521] [cite_start]In The 31st International Conference on Machine Learning[cite: 522].
[37] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. [cite_start]Proximal policy optimization algorithms. arXiv:1707.06347 (07 2017)[cite: 523].
[38] Wharton Research Data Service. 2015. Standard & Poor's Compustat. [cite_start]Data retrieved from Wharton Research Data Service[cite: 524]..
[39] W.F. Sharpe. 1994. The Sharpe ratio. [cite_start]Journal of Portfolio Management (01 1994)[cite: 525].
[cite_start][40] Richard Sutton and Andrew Barto, 1998. Reinforcement learning: an introduction. [cite: 526] [cite_start]IEEE Transactions on Neural Networks 9 (02 1998), 1054[cite: 527].
[41] Richard Sutton, David Mcallester, Satinder Singh, and Yishay Mansour. 2000. [cite_start]Policy gradient methods for reinforcement learning with function approximation. [cite: 528] [cite_start]Conference on Neural Information Processing Systems (NeurIPS), 1999 (02 2000)[cite: 529]..
[42] Lu Wang. [cite_start]Wei Zhang, Xiaofeng He, and Hongyuan Zha. [cite: 529] 2018. [cite_start]Supervised reinforcement learning with recurrent neural network for dynamic treatment recommendation. [cite: 530] [cite_start]In Conference on Knowledge Discovery and Data Mining (KDD), 2018. 2447-2456[cite: 531].
[43] Yuxin Wu and Yuandong Tian. 2017. [cite_start]Training agent for first-person shooter game with actor-critic curriculum learning. [cite: 532] [cite_start]In International Conference on Learning Representations (ICLR), 2017[cite: 533].
[cite_start][44] Zhuoran Xiong, Xiao-Yang Liu, Shan Zhong, Hongyang Yang, and A. Elwalid. [cite: 534] 2018. Practical deep reinforcement learning approach for stock trading. [cite_start]NeurIPS Workshop on Challenges and Opportunities for Al in Financial Services: the Impact of Fairness, Explainability, Accuracy, and Privacy, 2018. (2018)[cite: 535].
[45] Hongyang Yang, Xiao-Yang Liu, and Qingwei Wu. 2018. [cite_start]A practical machine learning approach for dynamic stock recommendation. [cite: 536] [cite_start]In IEEE TrustCom/BiDataSE, 2018. 1693-1697[cite: 537].
[46] Wenbin Zhang and Steven Skiena. 2010. [cite_start]Trading strategies to exploit blog and news sentiment.. In Fourth International AAAI Conference on Weblogs and Social Media, 2010[cite: 538].
[47] Yong Zhang and Xingyu Yang. 2016. [cite_start]Online portfolio selection strategy based on combining experts' advice. [cite: 539] [cite_start]Computational Economics 50 (05 2016)[cite: 540].
[48] Zihao Zhang. 2019. Deep reinforcement learning for trading. [cite_start]ArXiv 2019 (11 2019)[cite: 541].