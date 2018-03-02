import tensorflow as tf
import numpy as np


def Latent_Similarity_Network(n_inputs, n_outputs, batch_size, inputs, outputs):


	def tfloat32(x, name=''):
		return tf.Variable(tf.cast(np.asarray(x, dtype=np.float32), tf.float32), name=name, trainable=False)
	M_t0 = tfloat32(1e-6*np.random.rand(batch_size,n_outputs,n_inputs),'M_t0')
	h_t0 = tfloat32(np.zeros((batch_size,n_outputs)),'h_t0')


	def forward(st8_tm1, Xy_t):

		M_tm1, h_tm1 = st8_tm1['M'], st8_tm1['h']
		X_t, y_t = Xy_t['X'], Xy_t['y']

		# Cosine Distance
		X_norm = tf.norm(tf.expand_dims(X_t,1),axis=2,keep_dims=True)
		M_norm = tf.norm(M_tm1,axis=2,keep_dims=True)
		inner_prod = tf.matmul(tf.expand_dims(X_t,1),tf.transpose(M_tm1,perm=[0,2,1]))
		norm_prod = tf.matmul(X_norm,tf.transpose(M_norm,perm=[0,2,1]))
		h_t = tf.squeeze(tf.divide(inner_prod,norm_prod),1)

		# Update Latent Memory Definitions
		u_t = tf.matmul(tf.expand_dims(X_t,2),tf.expand_dims(y_t,1))
		M_t = M_tm1 + tf.tanh(tf.transpose(u_t,perm=[0,2,1]))

		st8_t = { 'M': M_t, 'h': h_t }
		return st8_t

	elems = { 'X': inputs, 'y': outputs }
	st8_t0 = { 'M': M_t0, 'h': h_t0 }
	st8_tf = tf.scan(forward, elems=elems, initializer=st8_t0)

	return st8_tf