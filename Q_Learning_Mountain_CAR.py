import gym
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt 
import pickle

problem = 'MountainCar-v0'
env = gym.make(problem)

nS = env.observation_space.shape[0]
nA = env.action_space.n
action = env.action_space.sample()

# *************************
MAX_EPSILON = 1.0
MIN_EPSILON = 0.05
l_rate=0.005
GAMMA = 0.99
LAMBDA = 0.0001
# *************************

# This model approx a q(a,s) function
def get_model():

	init = tf.keras.initializers.GlorotNormal(seed=3)

	inputs = tf.keras.layers.Input( shape=(nS) )
	out = tf.keras.layers.Dense(512,activation='relu',kernel_initializer=init)(inputs)
	out = tf.keras.layers.Dense(512,activation='relu',kernel_initializer=init)(out)
	outputs = tf.keras.layers.Dense(nA,kernel_initializer=init)(out)
	model = tf.keras.Model(inputs,outputs)

	model.compile(loss='mse',optimizer = tf.keras.optimizers.RMSprop( l_rate ))

	return model

def e_greedy_pi(epsilon,state):

	if np.random.random()<epsilon:
		action = np.random.choice( nA )
	else:		
		p_action = model(state)
		# Max of the all action , basic principle of Q-Learning
		action = tf.argmax( tf.squeeze(p_action) ).numpy()

	return action

model = get_model()

total_eps = 150
steps = 0
max_steps_per_ep = 4000

ep_reward_list = []

for ep in range(total_eps):

	if ep>50:
		MIN_EPSILON = 0.00001

	prev_state = env.reset()
	input_list = []
	target_list = []
	episodic_reward = 0

	ctr=0

	# Episodes can be large if agent perform bad so we limit then to 2000 for simplicity.
	# But it does not impact learning as Q-Learning is non-episodic.
	for t in range( max_steps_per_ep ):

		# Remove this if using in Jupyter Notebook
		env.render()

		# We reduce exploration exponentially
		epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * steps)

		tf_prev_state = tf.expand_dims( tf.convert_to_tensor( prev_state ) , 0 )

		action = e_greedy_pi( epsilon , tf_prev_state )
		state, reward, done, info = env.step(action)

		tf_state = tf.expand_dims( tf.convert_to_tensor( state ) , 0 )
		
		episodic_reward += reward
		inputs = tf_prev_state

		# keep all target same as prediction except for the action we choose
		targets = tf.squeeze( model( tf_prev_state ) )
		targets = np.array( targets.numpy() )

		# the condition is required to have long episodes 
		# else done will be True within 200 steps
		# and in these steps agent will never experience positive REWARD.
		if ( not info.get('TimeLimit.truncated',False) ) and done:
	
			print("Epsilon is == >  {}".format(epsilon))
			print("Success Reward ==> {}".format(reward))

			# as we are treating tuples of SARS' ( State , Action , Reward , Next_State)
			# And Q-Learning is not episodic there is not way for
			#  Agent to learn that max. episodic reward is good
			#  So we give +100 reward when agent reaches the goal
			#  NOTE : this will not be required for method like Monte Carlo

			reward = 100
			targets[action] = reward
			input_list.append( tf.squeeze(inputs) )
			target_list.append( targets )

			break
		
		else:

			# here we are computing only targets as we use use model.fit( )
			# but we can also use cost i.e td_error directly with tf.GradientTape()
			targets[action] = reward + GAMMA*tf.math.reduce_max( model(tf_state) )


			# so loss will be like this
			# td_error = reward + GAMMA*model(tf_state) - model(tf_prev_state)

			input_list.append( tf.squeeze(inputs) )
			target_list.append( targets )

			prev_state = state

		ctr += 1

		# epsidoes can be as long as "max_steps_per_ep"
		#  But we want agent to train even during episodes
		# So we use 128 steps as our training interval
		if ctr>=128:

			ctr = 0
			# preparing batch for training
			input_batch = tf.convert_to_tensor( input_list )
			target_batch = tf.convert_to_tensor( target_list )
			
			# now update our model by gradients we get for MSE loss
			model.fit( input_batch , target_batch )

			input_list = []
			target_list = []


		steps += 1

	
    # Agent will have high tendency to move as at each time step its getting -1 reward
    # So it will move a lot to increase this in positive direction.
    # Agent will reach the goal in first few epsidoes ~5 or even faster!!!
    #  You will get Great performance around ~100 episodes.
    # But of course it will fluctuate and can be made stable with experience relay.
	print("Episode * {} * ==> Total Reward is ==> {}".format(ep,episodic_reward))
	ep_reward_list.append(episodic_reward)

	# preparing batch for training of remaining samples
	input_batch = tf.convert_to_tensor( input_list )
	target_batch = tf.convert_to_tensor( target_list )
	
	# now update our model by gradients we get for MSE loss
	model.fit( input_batch , target_batch )


# Save with pickle
ql_results = open('results','wb')
pickle.dump( ep_reward_list , ql_results )                      

# Plot a Graph
# Episodes vs Rewards
plt.plot( ep_reward_list )
plt.xlabel('Episode')
plt.ylabel('Epsiodic Reward') 
plt.show()

env.close()

	

