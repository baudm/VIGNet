from keras import backend as K
from keras import activations, initializers
from keras.layers import RNN, Layer, InputSpec, add
from keras.utils import conv_utils
import numpy as np

class RNN3D(RNN):
	"""
	    # Arguments
        cell: A RNN cell instance. A RNN cell is a class that has:
            - a `call(input_at_t, states_at_t)` method, returning
                `(output_at_t, states_at_t_plus_1)`. The call method of the
                cell can also take the optional argument `constants`, see
                section "Note on passing external constants" below.
            - a `state_size` attribute. This can be a single integer
                (single state) in which case it is
                the number of channels of the recurrent state
                (which should be the same as the number of channels of the cell output).
                This can also be a list/tuple of integers
                (one size per state). In this case, the first entry
                (`state_size[0]`) should be the same as
                the size of the cell output.
        return_sequences: Boolean. Whether to return the last output.
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        input_shape: Use this argument to specify the shape of the
            input when this layer is the first one in a model.
    # Input shape
        1D tensor with shape:
        (samples, encoded)
    # Output shape
    	5D tensor with shape:
		(samples, depth, width, height, filters)
	"""

	def __init__(self, cell,
					return_sequences=False,
					return_state=False,
					go_backwards=False,
					stateful=False,
					**kwargs):
		if isinstance(cell, (list, tuple)):
			raise TypeError('It is not possible at the moment to stack RNN3D cellls')
		super(RNN3D, self).__init__(cell,
									return_sequences,
									return_state,
									go_backwards,
									stateful,
									**kwargs)
		self.input_spec = [InputSpec(ndim=3)]
		# self.state_spec = [InputSpec(ndim=5), InputSpec(ndim=5)]

	def compute_output_shape(self, input_shape):
		if isinstance(input_shape, list):
			input_shape = input_shape[0]

		cell = self.cell

		if cell.data_format == 'channels_first':
			output_shape = (input_shape[0], cell.filters) + cell.spatial_resolution
		elif cell.data_format == 'channels_last':
			output_shape = (input_shape[0],) + cell.spatial_resolution + (cell.filters,)

		return output_shape

	def build(self, input_shape):
		if isinstance(input_shape, list):
			input_shape = input_shape[0]

		batch_size = input_shape[0] if self.stateful else None
		self.input_spec[0] = InputSpec(shape=(batch_size, None)+input_shape[2:])

		if isinstance(self.cell, Layer):
			self.cell.build(input_shape)

		if hasattr(self.cell.state_size, '__len__'):
			state_size = list(self.cell.state_size)
		else:
			state_size = [self.cell.state_size]

		if self.state_spec is not None:
            # initial_state was passed in call, check compatibility
			if self.cell.data_format == 'channels_first':
				ch_dim = 1
			elif self.cell.data_format == 'channels_last':
				ch_dim = -1
			if not [spec.shape[ch_dim] for spec in self.state_spec] == state_size:
				raise ValueError(
					'An initial_state was passed that is not compatible with '
					'`cell.state_size`. Received `state_spec`={}; '
					'However `cell.state_size` is '
					'{}'.format([spec.shape for spec in self.state_spec], self.cell.state_size))
		else:
			if self.cell.data_format == 'channels_first':
				self.state_spec = [InputSpec(shape=(None, dim, None, None, None))
									for dim in state_size]
			elif self.cell.data_format == 'channels_last':
				self.state_spec = [InputSpec(shape=(None, None, None, None, dim))
									for dim in state_size]

		if self.stateful:
			self.return_states()

		self.built = True

	def get_initial_state(self, inputs):
		#input is in the form of (samples, time, sequence)
		dummy_inputs = K.zeros_like(inputs)
		# dummy_inputs = K.zeros((self.cell.batch_size,)+
		#weight is in the form of (N, N, N, sequence, filters)
		dummy_weights_shape = self.cell.spatial_resolution + (inputs.shape[2],) + (self.cell.filters,)
		dummy_weights = K.zeros(dummy_weights_shape)
		#initial state is the dot product
		initial_state = K.dot(dummy_inputs, dummy_weights)
		#Remove the time variable
		initial_state = initial_state[:,0]
		# initial_state = K.zeros((32, 4, 4, 4, 128))
		return [initial_state, initial_state]

	# def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
	# 	inputs, initial_state, constants = self._standardize_args(inputs, initial_state, constants)

	# 	if initial_state is None and constants is None:
	# 		return super(RNN3D, self).__call__(inputs, **kwargs)

	def call(self, inputs, mask=None, training=None, initial_state=None, constants=None, **kwargs):
		if isinstance(inputs, list):
			inputs = inputs[0]

		initial_state = self.get_initial_state(inputs)

		def step(inputs, states):
			return self.cell.call(inputs, states, **kwargs)
		
		last_output = initial_state[0]
		last_output, outputs, states = K.rnn(step,
											inputs,
											initial_state,
											constants=constants,
											go_backwards=self.go_backwards,
											mask=mask)
		return last_output

class LSTM3DCell(Layer):

	def __init__(self, filters,
				kernel_size=(3, 3, 3),
				strides=(1, 1, 1),
				spatial_resolution=(4, 4, 4),
				padding='same',
				data_format=None,
				activation='tanh',
				recurrent_activation='hard_sigmoid',
				use_bias=True,
				w_initializer='random_uniform',
				u_initializer='orthogonal',
				b_initializer='zeros',
				**kwargs):
		super(LSTM3DCell, self).__init__(**kwargs)
		self.filters = filters
		self.kernel_size = conv_utils.normalize_tuple(kernel_size, 3, 'kernel_size')
		self.strides = conv_utils.normalize_tuple(strides, 3, 'strides')
		self.spatial_resolution = conv_utils.normalize_tuple(spatial_resolution, 3, 'spatial_resolution')
		self.padding = conv_utils.normalize_padding(padding)
		self.data_format = conv_utils.normalize_data_format(data_format)
		self.activation = activations.get(activation)
		self.recurrent_activation = activations.get(recurrent_activation)
		self.use_bias = use_bias
		self.state_size = (self.filters, self.filters)

		self.w_initializer = initializers.get(w_initializer)
		self.u_initializer = initializers.get(u_initializer)
		self.b_initializer = initializers.get(b_initializer)

	def build(self, input_shape):
		weight_shape = self.spatial_resolution + (input_shape[2],) + (self.filters,)
		u_shape = self.kernel_size + (self.filters,) + (self.filters,)
		bias_shape = (self.filters,)

		self.Wf = self.add_weight(name='Wf',
									shape=weight_shape,
									initializer=self.w_initializer,
									trainable=True)
		self.Uf = self.add_weight(name='Uf',
									shape=u_shape,
									initializer=self.u_initializer,
									trainable=True)
		self.Wi = self.add_weight(name='Wi',
									shape=weight_shape,
									initializer=self.w_initializer,
									trainable=True)
		self.Ui = self.add_weight(name='Ui',
									shape=u_shape,
									initializer=self.u_initializer,
									trainable=True)
		self.Ws = self.add_weight(name='Ws',
									shape=weight_shape,
									initializer=self.w_initializer,
									trainable=True)
		self.Us = self.add_weight(name='Us',
									shape=u_shape,
									initializer=self.u_initializer,
									trainable=True)

		if self.use_bias:
			self.bf = self.add_weight(name='bf',
									shape=bias_shape,
									initializer=self.b_initializer,
									trainable=True)
			self.bi = self.add_weight(name='bi',
									shape=bias_shape,
									initializer=self.b_initializer,
									trainable=True)
			self.bs = self.add_weight(name='bs',
									shape=bias_shape,
									initializer=self.b_initializer,
									trainable=True)

		else:
			self.bf = None
			self.bi = None
			self.bs = None

		self.built = True

	def call(self, inputs, states, **kwargs):
		h_t = states[0]
		s_t = states[1]
		
		f = self.recurrent_activation(K.bias_add(K.dot(inputs, self.Wf) +
			K.conv3d(h_t, self.Uf, strides=self.strides, padding=self.padding), self.bf))
		i = self.recurrent_activation(K.bias_add(K.dot(inputs, self.Wi) +
			K.conv3d(h_t, self.Ui, strides=self.strides, padding=self.padding), self.bi))
		s = f*s_t + i*self.activation(K.bias_add(K.dot(inputs, self.Ws) +
			K.conv3d(h_t, self.Us, strides=self.strides, padding=self.padding), self.bs))
		h = self.activation(s)
		
		return h, [h, s]

class LSTM3D(RNN3D):
	def __init__(self, filters,
				kernel_size=(3, 3, 3),
				strides=(1, 1, 1),
				spatial_resolution=(4, 4, 4),
				padding='same',
				activation='tanh',
				recurrent_activation='hard_sigmoid',
				use_bias=True,
				w_initializer='random_uniform',
				u_initializer='orthogonal',
				b_initializer='zeros',
				**kwargs):
		cell = LSTM3DCell(filters=filters,
							kernel_size=kernel_size,
							strides=strides,
							spatial_resolution=spatial_resolution)
		super(LSTM3D, self).__init__(cell, **kwargs)

	def call(self, inputs):
		return super(LSTM3D, self).call(inputs)

class GRU3DCell(Layer):

	def __init__(self, filters,
				kernel_size=(3, 3, 3),
				strides=(1, 1, 1),
				spatial_resolution=(4, 4, 4),
				padding='same',
				data_format=None,
				activation='tanh',
				recurrent_activation='hard_sigmoid',
				use_bias=True,
				w_initializer='random_uniform',
				u_initializer='orthogonal',
				b_initializer='zeros',
				**kwargs):
		super(GRU3DCell, self).__init__(**kwargs)
		self.filters = filters
		self.kernel_size = conv_utils.normalize_tuple(kernel_size, 3, 'kernel_size')
		self.strides = conv_utils.normalize_tuple(strides, 3, 'strides')
		self.spatial_resolution = conv_utils.normalize_tuple(spatial_resolution, 3, 'spatial_resolution')
		self.padding = conv_utils.normalize_padding(padding)
		self.data_format = conv_utils.normalize_data_format(data_format)
		self.activation = activations.get(activation)
		self.recurrent_activation = activations.get(recurrent_activation)
		self.use_bias = use_bias
		self.state_size = (self.filters, self.filters)

		self.w_initializer = initializers.get(w_initializer)
		self.u_initializer = initializers.get(u_initializer)
		self.b_initializer = initializers.get(b_initializer)

	def build(self, input_shape):
		weight_shape = self.spatial_resolution + (input_shape[2],) + (self.filters,)
		u_shape = self.kernel_size + (self.filters,) + (self.filters,)
		bias_shape = (self.filters,)

		self.Wu = self.add_weight(name='Wu',
									shape=weight_shape,
									initializer=self.w_initializer,
									trainable=True)
		self.Uu = self.add_weight(name='Uu',
									shape=u_shape,
									initializer=self.u_initializer,
									trainable=True)
		self.Wi = self.add_weight(name='Wi',
									shape=weight_shape,
									initializer=self.w_initializer,
									trainable=True)
		self.Ui = self.add_weight(name='Ui',
									shape=u_shape,
									initializer=self.u_initializer,
									trainable=True)
		self.Wh = self.add_weight(name='Wh',
									shape=weight_shape,
									initializer=self.w_initializer,
									trainable=True)
		self.Uh = self.add_weight(name='Uh',
									shape=u_shape,
									initializer=self.u_initializer,
									trainable=True)

		if self.use_bias:
			self.bu = self.add_weight(name='bu',
									shape=bias_shape,
									initializer=self.b_initializer,
									trainable=True)
			self.bi = self.add_weight(name='bi',
									shape=bias_shape,
									initializer=self.b_initializer,
									trainable=True)
			self.bh = self.add_weight(name='bh',
									shape=bias_shape,
									initializer=self.b_initializer,
									trainable=True)

		else:
			self.bf = None
			self.bi = None
			self.bs = None

		self.built = True

	def call(self, inputs, states, **kwargs):
		h_t = states[0]
		
		u = self.recurrent_activation(K.bias_add(K.dot(inputs, self.Wu) +
			K.conv3d(h_t, self.Uu, strides=self.strides, padding=self.padding), self.bu))
		r = self.recurrent_activation(K.bias_add(K.dot(inputs, self.Wi) +
			K.conv3d(h_t, self.Ui, strides=self.strides, padding=self.padding), self.bi))
		h = (K.ones_like(u)-u)*h_t + u*self.activation(K.bias_add(K.dot(inputs, self.Wh) +
			K.conv3d(r*h_t, self.Uh, strides=self.strides, padding=self.padding), self.bh))
		
		return h, [h, h]

class GRU3D(RNN3D):
	def __init__(self, filters,
				kernel_size=(3, 3, 3),
				strides=(1, 1, 1),
				spatial_resolution=(4, 4, 4),
				padding='same',
				activation='tanh',
				recurrent_activation='hard_sigmoid',
				use_bias=True,
				w_initializer='random_uniform',
				u_initializer='orthogonal',
				b_initializer='zeros',
				**kwargs):
		cell = GRU3DCell(filters=filters,
							kernel_size=kernel_size,
							strides=strides,
							spatial_resolution=spatial_resolution)
		super(GRU3D, self).__init__(cell, **kwargs)

	def call(self, inputs):
		return super(GRU3D, self).call(inputs)