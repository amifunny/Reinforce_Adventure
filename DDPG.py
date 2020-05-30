import gym
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy

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

def get_model():
	inputs = tf.keras.layers.Input( shape=(nS) )
	out = tf.keras.layers.Dense(512,activation='relu')(inputs)
	out = tf.keras.layers.Dense(512,activation='relu')(out)
	outputs = tf.keras.layers.Dense(1)(out)
	model = tf.keras.Model(inputs,outputs)
	return model

def policy(state):
	
	sampled_number = actor_model(state)

	print(sampled_number)

	noise = tf.random.uniform( () , minval=-1.0 , maxval = 1.0 )

	sampled_number = sampled_number + noise
	# now these sampled real number can be large +ve's and -ve's
	# So we clip them by the bounds legal for environment

	print(sampled_number)
	legal_action =  tf.clip_by_value( tf.squeeze(sampled_number*2.0) , lower_bound , upper_bound )

	return [legal_action]

actor_model = get_model()
critic_model = get_model()

target_actor = tf.keras.models.clone_model( actor_model )
target_critic = tf.keras.models.clone_model( critic_model )

critic_optimizer = tf.keras.optimizers.Adam( 0.003 )
actor_optimizer = tf.keras.optimizers.Adam( 0.003 )

total_eps = 100
max_time_steps = 100
GAMMA = 0.99
tau = 0.6

ep_reward_list = []

for ep in range(total_eps):

	prev_state = env.reset()
	episodic_r = 0

	reward_list = []
	state_list = []
	next_state_list = []
	action_list = []
	
	with tf.GradientTape(persistent=True) as tape:

		for t in range(max_time_steps):

			env.render()

			tf_prev_state = tf.expand_dims( tf.convert_to_tensor( prev_state ) , 0 )

			# we sample a continous 'action' from policy
			action = policy( tf_prev_state )
			# Agent take that action , and enter a new 'state' and get a 'reward'
			state, reward, done, info = env.step(action)

			tf_state = tf.expand_dims( tf.convert_to_tensor( state ) , 0 )

			episodic_r += reward

			reward_list.append( reward )			
			state_list.append( tf_prev_state )
			next_state_list.append( tf_state )
			action_list.append( action )


			# Now this 'state' becomes our 'prev_state'
			prev_state = state


		print("Episode * {} * ==> Total Reward is ==> {}".format(ep,episodic_r))

		reward_batch = tf.convert_to_tensor( reward_list , dtype=tf.float32 )
		state_batch = tf.convert_to_tensor( state_list )
		next_state_batch = tf.convert_to_tensor( next_state_list )
		action_batch = tf.convert_to_tensor( action_list )

		y = tf.convert_to_tensor( reward_batch + GAMMA*target_critic( next_state_batch ) - critic_model( state_batch ) )
		critic_loss = tf.math.reduce_mean( tf.square( y ) )

		actor_loss = tf.math.reduce_mean( critic_model(state_batch) * actor_model(next_state_batch) )


	# Here we update sampling ACTOR and CRITIC
	critic_grad = tape.gradient( critic_loss , critic_model.trainable_variables )	
	critic_optimizer.apply_gradients( zip( critic_grad , critic_model.trainable_variables ) )

	actor_grad = tape.gradient( actor_loss , actor_model.trainable_variables )	
	actor_optimizer.apply_gradients( zip( actor_grad , actor_model.trainable_variables ) )

	# Update Target Actor-Critic
	for i,layer in enumerate( target_critic.layers ):

		if layer.weights!=[]:

			new_W = critic_model.get_layer(index=i).get_weights()[0]*tau + layer.get_weights()[0]*(i-tau)
			new_b = critic_model.get_layer(index=i).get_weights()[1]*tau + layer.get_weights()[1]*(i-tau)

			layer.set_weights( [new_W,new_b] )

	for i,layer in enumerate( target_actor.layers ):

		if layer.weights!=[]:


			new_W = actor_model.get_layer(index=i).get_weights()[0]*tau + layer.get_weights()[0]*(i-tau)
			new_b = actor_model.get_layer(index=i).get_weights()[1]*tau + layer.get_weights()[1]*(i-tau)

			layer.set_weights( [new_W,new_b] )


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



