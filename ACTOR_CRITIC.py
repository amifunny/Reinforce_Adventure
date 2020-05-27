import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

problem = 'CartPole-v1'
env = gym.make(problem)

nS = env.observation_space.shape[0]
nA = env.action_space.n
action = env.action_space.sample()


init = tf.keras.initializers.GlorotUniform(seed=3)
def get_actor():

	inputs = tf.keras.layers.Input( shape=(nS) )
	out = tf.keras.layers.Dense(512,activation='relu',kernel_initializer=init)(inputs)
	out = tf.keras.layers.Dense(512,activation='relu',kernel_initializer=init)(out)
	outputs = tf.keras.layers.Dense(nA,activation='softmax',kernel_initializer=init)(out)
	model = tf.keras.Model(inputs,outputs)

	return model

def get_critic():

	inputs = tf.keras.layers.Input( shape=(nS) )
	out = tf.keras.layers.Dense(512,activation='relu',kernel_initializer=init)(inputs)
	out = tf.keras.layers.Dense(512,activation='relu',kernel_initializer=init)(out)
	outputs = tf.keras.layers.Dense(1,kernel_initializer=init)(out)
	model = tf.keras.Model(inputs,outputs)

	return model

def policy(state):
	
	p_action = actor_model(state)
	act_prob = np.squeeze( p_action.numpy() )
	action = np.random.choice(nA,p=act_prob )
	return action,p_action

actor_model = get_actor()
critic_model = get_critic()
optimizer = tf.keras.optimizers.Adam( 0.001 )

total_eps = 800
GAMMA = 0.99

ep_reward_list = []
running_reward = 0

for ep in range(total_eps):

	prev_state = env.reset()
	episodic_r = 0

	actor_losses = []
	critic_losses = []

	avg_reward = 0

	while True:

		env.render()

		with tf.GradientTape(persistent=True) as tape:

			tf_prev_state = tf.expand_dims( tf.convert_to_tensor( prev_state ) , 0 )

			action,act_prob = policy( tf_prev_state )
			state, reward, done, info = env.step(action)

			tf_state = tf.expand_dims( tf.convert_to_tensor( state ) , 0 )

			td_error = reward - avg_reward + critic_model( tf_state )
			avg_reward = avg_reward + td_error*0.01

			# if done:
			# 	target = reward + GAMMA*critic_model(tf_state)
			# else:
			# 	target = reward
				
			# td_error = target - critic_model(tf_prev_state)

			log_prob = tf.math.log(  act_prob )

			critic_loss = td_error
			actor_loss =  -1.0*td_error * log_prob[0][action]

			critic_losses.append( critic_loss )
			actor_losses.append( actor_loss )
			
			
			episodic_r += reward

			if done:
				running_reward = 0.05 * episodic_r + (1 - 0.05) * running_reward
				print("ruuuunnningg rewaaaard is ===>  {}".format(running_reward))
				ep_reward_list.append( episodic_r )
				print("TD error ==> {}".format(td_error.numpy()))
				print("Episode * {} * ==> Total Reward is ==> {}".format(ep,episodic_r))

				summed_critic_loss = sum( critic_losses )
				summed_actor_loss = sum( actor_losses )

				break
			else:
				prev_state = state

	print(summed_critic_loss)
	print(summed_actor_loss)
	print( act_prob )

	critic_grad = tape.gradient( summed_critic_loss , critic_model.trainable_variables )	
	optimizer.apply_gradients( zip( critic_grad , critic_model.trainable_variables ) )

	actor_grad = tape.gradient( summed_actor_loss , actor_model.trainable_variables )	
	optimizer.apply_gradients( zip( actor_grad , actor_model.trainable_variables ) )



# # Save with pickle
ql_results = open('results','wb')
pickle.dump( ep_reward_list , ql_results )                      

# Plot a Graph
# Episodes vs Rewards
plt.plot( ep_reward_list )
plt.xlabel('Episode')
plt.ylabel('Epsiodic Reward') 
plt.show()

env.close()

env.close()



