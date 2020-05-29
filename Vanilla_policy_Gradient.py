import gym
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

problem = 'CartPole-v1'
env = gym.make(problem)

nS = env.observation_space.shape[0]
nA = env.action_space.n
action = env.action_space.sample()


# Approx a policy function
def get_model():
	inputs = tf.keras.layers.Input( shape=(nS) )
	out = tf.keras.layers.Dense(128,activation='relu')(inputs)
	out = tf.keras.layers.Dense(128,activation='relu')(out)
	outputs = tf.keras.layers.Dense(nA,activation='softmax')(out)
	model = tf.keras.Model(inputs,outputs)


	return model

def policy(state):
	
	p_action = actor_model(state)
	act_prob = np.squeeze( p_action.numpy() )
	most_prob_action = np.random.choice(nA,p=act_prob )

	log_prob = tf.math.log( tf.clip_by_value(tf.squeeze( p_action )[most_prob_action] , 1e-8 , 1-1e-8 ) )

	return most_prob_action,log_prob

actor_model = get_model()

optimizer = tf.keras.optimizers.Adam( 0.0003 )

total_eps = 400
GAMMA = 0.99
ep_reward_list = []

for ep in range(total_eps):

	prev_state = env.reset()
	episodic_r = 0
	time_step_rewards = []
	time_step_log_prob = []

	with tf.GradientTape() as tape:

		while True:

			env.render()

			tf_prev_state = tf.expand_dims( tf.convert_to_tensor( prev_state ) , 0 )

			action,log_prob = policy( tf_prev_state )
			state, reward, done, info = env.step(action)

			tf_state = tf.convert_to_tensor( state )

			time_step_rewards.append( reward )
			time_step_log_prob.append( log_prob )

			episodic_r += reward

			if done:
				ep_reward_list.append( episodic_r )
				print("Episode * {} * ==> Total Reward is ==> {}".format(ep,episodic_r))
				break
			else:
				prev_state = state



		# Here to calculate Gt we use Monte Carlo return
		each_step_return = []
		for time_step in range(len(time_step_rewards)):

			# now for each time step we calculate discounted future rewards
			Gt=0
			pw = 0
			for reward in time_step_rewards[time_step:]:
				Gt = Gt + (GAMMA**pw)*reward
				pw += 1

			each_step_return.append(Gt)

		each_step_loss = []
		for i,Gt in enumerate( each_step_return ):
			each_step_loss.append( -time_step_log_prob[i]*Gt )

		summed_loss = tf.math.reduce_sum( tf.convert_to_tensor( each_step_loss ) )

	actor_grad = tape.gradient( summed_loss , actor_model.trainable_variables )
	optimizer.apply_gradients( zip(actor_grad,actor_model.trainable_variables ) )


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

	




