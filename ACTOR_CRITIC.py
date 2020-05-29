import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

problem = 'CartPole-v1'
env = gym.make(problem)

nS = env.observation_space.shape[0]
nA = env.action_space.n

init = tf.keras.initializers.GlorotUniform(seed=6)
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
	return action,tf.squeeze(p_action)

actor_model = get_actor()
critic_model = get_critic()

actor_optimizer = tf.keras.optimizers.Adam( 0.003 )
critic_optimizer = tf.keras.optimizers.Adam( 0.003 )

total_eps = 600
GAMMA = 0.99

ep_reward_list = []

for ep in range(total_eps):

	prev_state = env.reset()
	episodic_r = 0

	reward_list = []
	values_state_list = []
	values_next_state_list = []
	log_prob_list = []

	with tf.GradientTape(persistent=True) as tape:

		while True:

			env.render()

			tf_prev_state = tf.expand_dims( tf.convert_to_tensor( prev_state ) , 0 )

			action,act_prob = policy( tf_prev_state )
			state, reward, done, info = env.step(action)

			log_prob = tf.math.log( act_prob[action] )
			value_state = tf.squeeze( critic_model(tf_prev_state) )

			tf_state = tf.expand_dims( tf.convert_to_tensor( state ) , 0 )

			value_next_state = tf.squeeze( critic_model(tf_state) )

			# lets store all vital info of each time step
			reward_list.append( reward )			
			values_state_list.append( value_state )
			values_next_state_list.append( value_next_state )
			log_prob_list.append( log_prob )

			episodic_r += reward

			if done:

				ep_reward_list.append( episodic_r )
				print("Episode * {} * ==> Total Reward is ==> {}".format(ep,episodic_r))
				break

			else:
				prev_state = state


		# Exactly as in theory we calculate ACTOR & CRITIC LOSSES
		advantage = tf.convert_to_tensor( tf.convert_to_tensor(reward_list) + GAMMA*tf.convert_to_tensor(values_next_state_list) - tf.convert_to_tensor(values_state_list) )
		tf_log_prob_list = tf.convert_to_tensor(log_prob_list)
		actor_loss = tf.math.reduce_mean(-1*tf_log_prob_list * advantage)
		critic_loss = tf.math.reduce_mean( 0.5 * tf.square( advantage ) )


	critic_grad = tape.gradient( critic_loss , critic_model.trainable_variables )	
	critic_optimizer.apply_gradients( zip( critic_grad , critic_model.trainable_variables ) )

	actor_grad = tape.gradient( actor_loss , actor_model.trainable_variables )	
	actor_optimizer.apply_gradients( zip( actor_grad , actor_model.trainable_variables ) )



# # Save with pickle
ql_results = open('results','wb')
pickle.dump( ep_reward_list , ql_results )                      

# Plot a Graph
# Episodes vs Rewards
# Now that what you call a beautiful graph
# You may notice Actor-Critic perform better than simple Q-Learning and Vanilla Policy Gradient
plt.plot( ep_reward_list )
plt.xlabel('Episode')
plt.ylabel('Epsiodic Reward') 
plt.show()

env.close()




