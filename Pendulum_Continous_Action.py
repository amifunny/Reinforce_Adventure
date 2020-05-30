import gym
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

problem = 'Pendulum-v0'
env = gym.make(problem)

nS = env.observation_space.shape[0]
print( nS )
nA = env.action_space
print( nA )

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print( upper_bound )
print( lower_bound )

# Now if you are familiar with DISCRETE Actor-Critic Method
# Then its all the same.
# Just now we simply treat pi(policy)
# as function of mu(mean) and sigma(variance)
# It will not affect the declaration of actor model.
# But simply the way we treat it.
def get_model():
	inputs = tf.keras.layers.Input( shape=(nS) )
	out = tf.keras.layers.Dense(512,activation='relu')(inputs)
	out = tf.keras.layers.Dense(512,activation='relu')(out)
	outputs = tf.keras.layers.Dense(1,activation='tanh')(out)
	model = tf.keras.Model(inputs,outputs)
	return model

def policy(state):
	
	sampled_number = actor_model(state)

	# now these sampled real number can be large +ve's and -ve's
	# So we clip them by the bounds legal for environment

	legal_action =  tf.clip_by_value( tf.squeeze(sampled_number*2.0) , lower_bound , upper_bound )

	return [legal_action]

actor_model = get_model()
critic_model = get_model()

critic_optimizer = tf.keras.optimizers.Adam( 0.003 )
actor_optimizer = tf.keras.optimizers.Adam( 0.003 )

total_eps = 100
max_time_steps = 1000
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

		for t in range(max_time_steps):

			env.render()

			tf_prev_state = tf.expand_dims( tf.convert_to_tensor( prev_state ) , 0 )

			# we sample a continous 'action' from policy
			action = policy( tf_prev_state )
			# Agent take that action , and enter a new 'state' and get a 'reward'
			state, reward, done, info = env.step(action)

			tf_state = tf.expand_dims( tf.convert_to_tensor( state ) , 0 )

			# td_error = reward + GAMMA*critic_model(tf_state) - critic_model(tf_prev_state)
			episodic_r += reward

			log_prob = tf.math.log( action )
			value_state = tf.squeeze( critic_model(tf_prev_state) )
			value_next_state = tf.squeeze( critic_model(tf_state) )


			reward_list.append( reward )			
			values_state_list.append( value_state )
			values_next_state_list.append( value_next_state )
			log_prob_list.append( log_prob )


			# Now this 'state' becomes our 'prev_state'
			prev_state = state


		print("Episode * {} * ==> Total Reward is ==> {}".format(ep,episodic_r))

		advantage = tf.convert_to_tensor( tf.convert_to_tensor(reward_list,dtype=tf.float32) + GAMMA*tf.convert_to_tensor(values_next_state_list) - tf.convert_to_tensor(values_state_list) )
		tf_log_prob_list = tf.convert_to_tensor(log_prob_list)
		actor_loss = tf.math.reduce_mean(-1*tf_log_prob_list * advantage)
		critic_loss = tf.math.reduce_mean( 0.5 * tf.square( advantage ) )

	critic_grad = tape.gradient( critic_loss , critic_model.trainable_variables )	
	critic_optimizer.apply_gradients( zip( critic_grad , critic_model.trainable_variables ) )

	actor_grad = tape.gradient( actor_loss , actor_model.trainable_variables )	
	actor_optimizer.apply_gradients( zip( actor_grad , actor_model.trainable_variables ) )


	ep_reward_list.append( episodic_r )






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



