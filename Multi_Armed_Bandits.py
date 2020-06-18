import numpy as np
import matplotlib.pyplot as plt

"""
Quick Theory -

	Multi Armed Bandits is a RL Algorithm that find
	best action out of `k` actions. In this, there is only one State.
	We incremently change our choise's distribution to get maximum
	cummulative gain.
	It also proposes dillemma of Exploration & Exploitation.
	Which can be solved by e-greedy, UBC or thompson sampling.

Problem - 
	
	Our Agent is at a Casino, with `k` number of slot machines in
	front of him. Each machine has different win probabilities.
	We would like to learn which is best machine to win max rewards.

Solution -

	Our Agent choses the best machine according to current estimate,
	and based on reward recieved change its estimate. We also force
	our Agent to explore different machines to reach optimum goal

"""	

K_ARMS = 100

class SlotMachines():

	def __init__(self,num_of_slots,seed):

		np.random.seed( seed )
		# We randomly generate "win probabilities" of all slot machines
		self.slot_probs = np.random.uniform(0.0,1.0,[1,num_of_slots])

	# Takes choice of machine and based on its
	# initialized probability give some reward.
	def play( self , choice ):
		# `choice` ~ [0,num_of_slot-1]

		tp = np.random.random()

		if tp<self.slot_probs[0,choice]:
			# Agent wins and get reward as 1
			return 1
		else:
			return 0



# Lets use simple still powerful ,method of exploration - "e-greedy"
# and solve this problem

class EGreedyBandit():

	def __init__(self,epsilon):

		self.epsilon = epsilon
		# Initialize optimistically
		self.arm_values = [1.0]*K_ARMS

		# To store number of times a arm is chosen
		self.num_trials = [0]*K_ARMS

	def make_choice(self):

		if np.random.random()<self.epsilon:
			# Take random exploratory action
			choice = np.random.randint( 0 , K_ARMS , size=() )
		else:
			choice = np.argmax( self.arm_values )

		self.num_trials[choice] += 1

		return choice


	def update_estimates(self,reward,choice):

		# Estimate update should be = (cummulative_rewards)/total_trials_of_arm 
		# Which can be mathematically reduced to this equation-
		self.arm_values[choice] +=  (reward-self.arm_values[choice])/(self.num_trials[choice])

		return

# Another Populer solution to exploration problem is "Thompson Sampling",
# Fun FACT : This technique was introduced in 1933 and was largely ignore,
# Untill recently after eight decade in 2010's, was shown to give good results.

# In "Thompson sampling" we sample an ARM's estimate from a Bayesian Distribution,
# Specifically "beta" distribution parameterized by (a,b)
# `a` and `b` act as pseudo counter of number of trials.

class ThompsonSampling():

	def __init__(self,init_a,init_b):

		self.a = [init_a]*K_ARMS
		self.b = [init_b]*K_ARMS


	def get_estimates(self):
		# To get array of estimates
		estimates = []
		for i in range(K_ARMS):
			estimates.append( self.a[i]/(self.a[i]+self.b[i]) )
		return estimates

	def make_choice(self):

		sampled_estimates = []
		for i in range(K_ARMS):
			sampled_estimates.append( np.random.beta(self.a[i],self.b[i]) )

		choice = np.argmax( np.array(sampled_estimates) )

		return choice


	def update_estimates(self,reward,choice):

		# We update `a` and `b`, and hence our estimates
		self.a[choice] += reward
		self.b[choice] += 1 - reward

		return



# Main loop
def main_loop(agent,max_steps=5000):

	reward_list = []
	avg_reward_list = []

	for i in range( max_steps ):

		choice = agent.make_choice()
		step_reward = casino.play(choice)
		agent.update_estimates( step_reward,choice )

		reward_list.append( step_reward )
		avg_reward_list.append( np.sum( np.array(reward_list)[-100:] ) )

		if i%int(max_steps/10)==0:
			print("Step :: {}".format(i))

	return avg_reward_list


casino = SlotMachines( num_of_slots = K_ARMS , seed = 1 )

greedyAgent = EGreedyBandit(0.1)
greedy_reward_list = main_loop( greedyAgent , 20000 )

TSAgent = ThompsonSampling(1.0,1.0)
ts_reward_list = main_loop( TSAgent , 20000 )


print("Agent Estimate - ")
print(greedyAgent.arm_values)
print("True Probs - ")
print(casino.slot_probs[0])

print("Difference - {}".format( greedyAgent.arm_values - casino.slot_probs[0] ))

print("------------------------------------------------------------")

print("Agent Estimate - ")
print(TSAgent.get_estimates())
print("True Probs - ")
print(casino.slot_probs[0])

print("Difference - {}".format( TSAgent.get_estimates() - casino.slot_probs[0] ))

plt.plot( greedy_reward_list , label = "Greedy" )
plt.plot( ts_reward_list , label = "Thompson" )

plt.xlabel('Episode')
plt.ylabel('Cummulative Reward') 
plt.legend()
plt.show()

"""
	Visually you can see that thompson sampling comes out
	to bes more superior and more stable.
	Concept of epsilon-greedy can also be made better using
	epsilon decay.
	Feel Free to Try different `K_ARMS` value and 'max_steps'.
"""



