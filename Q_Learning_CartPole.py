import gym
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt 
import pickle

problem = 'CartPole-v1'
env = gym.make(problem)

nS = env.observation_space.shape[0]
nA = env.action_space.n
action = env.action_space.sample()

# *************************
MAX_EPSILON = 0.5
MIN_EPSILON = 0.05
l_rate=0.01
GAMMA = 0.99
LAMBDA = 0.001
# *************************

# This model approx a q(a,s) function
def get_model():

	init = tf.keras.initializers.GlorotNormal(seed=3)

	inputs = tf.keras.layers.Input( shape=(nS) )
	out = tf.keras.layers.Dense(128,activation='relu',kernel_initializer=init)(inputs)
	out = tf.keras.layers.Dense(128,activation='relu',kernel_initializer=init)(out)
	outputs = tf.keras.layers.Dense(nA,kernel_initializer=init)(out)
	model = tf.keras.Model(inputs,outputs)

	model.compile(loss='mse',optimizer = tf.keras.optimizers.RMSprop( l_rate ))

	return model

def e_greedy_pi(epsilon,state):

	if np.random.random()<epsilon:
		action = np.random.choice( nA )
	else:		
		p_action = model(state)
		action = tf.argmax( tf.squeeze(p_action) ).numpy()

	return action

model = get_model()

total_eps = 400
steps = 0

ep_reward_list = []

for ep in range(total_eps):

	prev_state = env.reset()
	input_list = []
	target_list = []
	episodic_reward = 0

	while True:

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

		if done:
	
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

		steps += 1


	# Aroung 250~ episodes you will get great results
	# But results will not be that stable
	# and will fluctuate a lot! Experience Relay can make it lot stable.
	print("Episode * {} * ==> Total Reward is ==> {}".format(ep,episodic_reward))
	ep_reward_list.append(episodic_reward)

	# preparing batch for training
	input_batch = tf.convert_to_tensor( input_list )
	target_batch = tf.convert_to_tensor( target_list )

	# now update our model by gradients we get for MSE loss
	model.fit( input_batch , target_batch )


# Save as pickle
ql_results = open('results','wb')
pickle.dump( ep_reward_list , ql_results )                      

# Plot a Graph
# Episodes vs Rewards
plt.plot( ep_reward_list )
plt.xlabel('Episode')
plt.ylabel('Epsiodic Reward') 
plt.show()

env.close()

	

