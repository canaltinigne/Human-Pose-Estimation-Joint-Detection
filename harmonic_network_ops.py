"""
Core Harmonic Convolution Implementation
"""

import numpy as np
import tensorflow as tf


def h_conv(X, W, strides=(1,1,1,1), padding='VALID', max_order=1, name='h_conv'):
    """Inter-order (cross-stream) convolutions can be implemented as single
    convolution. For this we store data as 6D tensors and filters as 8D
    tensors, at convolution, we reshape down to 4D tensors and expand again.

    X: tensor shape [mbatch,h,w,order,complex,channels]
    Q: tensor dict---reshaped to [h,w,in,in.comp,in.ord,out,out.comp,out.ord]
    P: tensor dict---phases
    strides: as per tf convention (default (1,1,1,1))
    padding: as per tf convention (default VALID)
    filter_size: (default 3)
    max_order: (default 1)
    name: (default h_conv)
    """
    with tf.name_scope('hconv'+str(name)) as scope:
        # Build data tensor: reshape it as [mbatch,h,w,order*complex*channels]
        Xsh = X.get_shape().as_list()
        X_ = tf.reshape(X, tf.concat(axis=0,values=[Xsh[:3],[-1]]))

        # The script below constructs the stream-convolutions as one big filter
        # W_. For each output order, run through each input order and
        # copy-paste the filter for that convolution.
        W_ = []
        for output_order in range(max_order+1):
            # For each output order build input
            Wr = []
            Wi = []
            for input_order in range(Xsh[3]):
                # Difference in orders is the convolution order
                weight_order = output_order - input_order
                weights = W[np.abs(weight_order)]
                sign = np.sign(weight_order)
                # Choose a different filter depending on whether input is real.
                # We have the arbitrary convention that negative orders use the
                # conjugate weights.
                if Xsh[4] == 2:
                    Wr += [weights[0],-sign*weights[1]]
                    Wi += [sign*weights[1],weights[0]]
                else:
                    Wr += [weights[0]]
                    Wi += [weights[1]]
            W_ += [tf.concat(axis=2, values=Wr), tf.concat(axis=2, values=Wi)]
        W_ = tf.concat(axis=3, values=W_)

        # Convolve
        Y = tf.nn.conv2d(X_, W_, strides=strides, padding=padding, name=name)
        # Reshape result into appropriate format
        Ysh = Y.get_shape().as_list()
        new_shape = tf.concat(axis=0, values=[Ysh[:3],[max_order+1,2],[Ysh[3]//(2*(max_order+1))]])
        return tf.reshape(Y, new_shape)


def h_range_conv(X, W, strides=(1,1,1,1), padding='VALID', in_range=(0,1),
                      out_range=(0,1), name='r_conv'):
    """Inter-order (cross-stream) convolutions can be implemented as single
    convolution. For this we store data as 6D tensors and filters as 8D
    tensors, at convolution, we reshape down to 4D tensors and expand again.

    X: tensor shape [mbatch,h,w,order,complex,channels]
    Q: tensor dict---reshaped to [h,w,in,in.comp,in.ord,out,out.comp,out.ord]
    P: tensor dict---phases
    strides: as per tf convention (default (1,1,1,1))
    padding: as per tf convention (default VALID)
    filter_size: (default 3)
    in_range: (default (0,1))
    out_range: (default (0,1))
    name: (default r_conv)
    """
    with tf.name_scope('hconv'+str(name)) as scope:
        # Build data tensor: reshape it as [mbatch,h,w,order*complex*channels]
        Xsh = X.get_shape().as_list()
        X_ = tf.reshape(X, tf.concat(axis=0,values=[Xsh[:3],[-1]]))

        # The script below constructs the stream-convolutions as one big filter
        # W_. For each output order, run through each input order and copy-paste
        # the filter for that convolution.
        W_ = []
        for output_order in range(out_range[0], out_range[1]+1):
            # For each output order build input
            Wr = []
            Wi = []
            for input_order in range(in_range[0], in_range[1]+1):
                # Difference in orders is the convolution order
                weight_order = output_order - input_order
                weights = W[weight_order]
                # Choose a different filter depending on whether input is real. We
                # have the arbitrary convention that negative orders use the
                # conjugate weights.
                if Xsh[4] == 2:
                    Wr += [weights[0],-weights[1]]
                    Wi += [weights[1], weights[0]]
                else:
                    Wr += [weights[0]]
                    Wi += [weights[1]]
            W_ += [tf.concat(axis=2, values=Wr), tf.concat(axis=2, values=Wi)]
        W_ = tf.concat(axis=3, values=W_)

        # Convolve
        Y = tf.nn.conv2d(X_, W_, strides=strides, padding=padding, name=name)
        # Reshape result into appropriate format
        Ysh = Y.get_shape().as_list()
        diff = out_range[1] - out_range[0] + 1
        new_shape = tf.concat(axis=0, values=[Ysh[:3],[diff,2],[Ysh[3]//(2*diff)]])
        return tf.reshape(Y, new_shape)



##### NONLINEARITIES #####
def h_nonlin(X, fnc, eps=1e-12, name='b'):
    """Apply the nonlinearity described by the function handle fnc: R -> R+ to
    the magnitude of X. CAVEAT: fnc must map to the non-negative reals R+.

    Output U + iV = fnc(R+b) * (A+iB)
    where  A + iB = Z/|Z|

    X: dict of channels {rotation order: (real, imaginary)}
    fnc: function handle for a nonlinearity. MUST map to non-negative reals R+
    eps: regularization since grad |Z| is infinite at zero (default 1e-8)
    """
    magnitude = stack_magnitudes(X, eps)
    msh = magnitude.get_shape()
    b = tf.get_variable('b'+name, shape=[1,1,1,msh[3],1,msh[5]])

    Rb = tf.add(magnitude, b)
    c = tf.div(fnc(Rb), magnitude)
    return c*X


def h_batch_norm(X, fnc, train_phase, decay=0.99, eps=1e-12, name='hbn'):
    """Batch normalization for the magnitudes of X

    X: dict of channels {rotation order: (real, imaginary)}
    fnc: function handle for a nonlinearity. MUST map to non-negative reals R+
    train_phase: boolean flag True: training mode, False: test mode
    decay: decay rate: 0 is memory-less, 1 no updates (default 0.99)
    eps: regularization since grad |Z| is infinite at zero (default 1e-8)
    name: (default complexBatchNorm)
    """
    with tf.name_scope(name) as scope:
        magnitude = stack_magnitudes(X, eps)
        Rb = bn(magnitude, train_phase, decay=decay, name=name)
        c = tf.div(fnc(Rb), magnitude)
        return c*X


def bn(X, train_phase, decay=0.99, name='batchNorm'):
    """Batch normalization module.

    X: tf tensor
    train_phase: boolean flag True: training mode, False: test mode
    decay: decay rate: 0 is memory-less, 1 no updates (default 0.99)
    name: (default batchNorm)

    Source: bgshi @ http://stackoverflow.com/questions/33949786/how-could-i-use-
    batch-normalization-in-tensorflow"""
    Xsh = X.get_shape().as_list()
    n_out = Xsh[-3:]

    with tf.name_scope(name) as scope:
        beta = tf.get_variable(name+'_beta', dtype=tf.float32, shape=n_out,
                                      initializer=tf.constant_initializer(0.0))
        gamma = tf.get_variable(name+'_gamma', dtype=tf.float32, shape=n_out,
                                        initializer=tf.constant_initializer(1.0))
        pop_mean = tf.get_variable(name+'_pop_mean', dtype=tf.float32,
                                            shape=n_out, trainable=False)
        pop_var = tf.get_variable(name+'_pop_var', dtype=tf.float32,
                                          shape=n_out, trainable=False)
        batch_mean, batch_var = tf.nn.moments(X, np.arange(len(Xsh)-3), name=name+'moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        pop_mean_op = tf.assign(pop_mean, ema.average(batch_mean))
        pop_var_op = tf.assign(pop_var, ema.average(batch_var))

        with tf.control_dependencies([ema_apply_op, pop_mean_op, pop_var_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(train_phase, mean_var_with_update,
                lambda: (pop_mean, pop_var))
    normed = tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-3)
    return normed


def mean_pooling(x, ksize=(1,1,1,1), strides=(1,1,1,1)):
    """Implement mean pooling on complex-valued feature maps. The complex mean
    on a local receptive field, is performed as mean(real) + i*mean(imag)

    x: tensor shape [mbatch,h,w,order,complex,channels]
    ksize: kernel size 4-tuple (default (1,1,1,1))
    strides: stride size 4-tuple (default (1,1,1,1))
    """
    Xsh = x.get_shape()
    # Collapse output the order, complex, and channel dimensions
    X_ = tf.reshape(x, tf.concat(axis=0,values=[Xsh[:3],[-1]]))
    Y = tf.nn.avg_pool(X_, ksize=ksize, strides=strides, padding='VALID',
                       name='mean_pooling')
    Ysh = Y.get_shape()
    new_shape = tf.concat(axis=0, values=[Ysh[:3],Xsh[3:]])
    return tf.reshape(Y, new_shape)


def stack_magnitudes(X, eps=1e-12, keep_dims=True):
    """Stack the magnitudes of each of the complex feature maps in X.

    Output U = concat(|X_i|)

    X: dict of channels {rotation order: (real, imaginary)}
    eps: regularization since grad |Z| is infinite at zero (default 1e-12)
    """
    R = tf.reduce_sum(tf.square(X), axis=[4], keep_dims=keep_dims)
    return tf.sqrt(tf.maximum(R,eps))


##### CREATING VARIABLES #####
def to_constant_float(Q):
    """Converts a numpy tensor to a tf constant float

    Q: numpy tensor
    """
    Q = tf.Variable(Q, trainable=False)
    return tf.to_float(Q)


def get_weights(filter_shape, W_init=None, std_mult=0.4, name='W'):
    """Initialize weights variable with He method

    filter_shape: list of filter dimensions
    W_init: numpy initial values (default None)
    std_mult: multiplier for weight standard deviation (default 0.4)
    name: (default W)
    device: (default /cpu:0)
    """
    if W_init == None:
        stddev = std_mult*np.sqrt(2.0 / np.prod(filter_shape[:3]))
        W_init = tf.random_normal_initializer(stddev=stddev)
    return tf.get_variable(name, dtype=tf.float32, shape=filter_shape,
            initializer=W_init)


##### FUNCTIONS TO CONSTRUCT STEERABLE FILTERS #####
def get_interpolation_weights(filter_size, m, n_rings=None):
    """Resample the patches on rings using Gaussian interpolation"""
    if n_rings is None:
        n_rings = np.maximum(filter_size/2, 2)
    radii = np.linspace(m!=0, n_rings-0.5, n_rings) #<-------------------------look into m and n-rings-0.5
    # We define pixel centers to be at positions 0.5
    foveal_center = np.asarray([filter_size, filter_size])/2.
    # The angles to sample
    N = n_samples(filter_size)
    lin = (2*np.pi*np.arange(N))/N
    # Sample equi-angularly along each ring
    ring_locations = np.vstack([-np.sin(lin), np.cos(lin)])
    # Create interpolation coefficient coordinates
    coords = L2_grid(foveal_center, filter_size)
    # Sample positions wrt patch center IJ-coords
    radii = radii[:,np.newaxis,np.newaxis,np.newaxis]
    ring_locations = ring_locations[np.newaxis,:,:,np.newaxis]
    diff = radii*ring_locations - coords[np.newaxis,:,np.newaxis,:]
    dist2 = np.sum(diff**2, axis=1)
    # Convert distances to weightings
    bandwidth = 0.5
    weights = np.exp(-0.5*dist2/(bandwidth**2))
    # Normalize
    return weights/np.sum(weights, axis=2, keepdims=True)


def get_filters(R, filter_size, P=None, n_rings=None):
    """Perform single-frequency DFT on each ring of a polar-resampled patch"""
    k = filter_size
    filters = {}
    N = n_samples(k)
    from scipy.linalg import dft
    for m, r in R.items():
        rsh = r.get_shape().as_list()
        # Get the basis matrices
        weights = get_interpolation_weights(k, m, n_rings=n_rings)
        DFT = dft(N)[m,:]
        LPF = np.dot(DFT, weights).T

        cosine = np.real(LPF).astype(np.float32)
        sine = np.imag(LPF).astype(np.float32)
        # Reshape for multiplication with radial profile
        cosine = tf.constant(cosine)
        sine = tf.constant(sine)
        # Project taps on to rotational basis
        r = tf.reshape(r, tf.stack([rsh[0],rsh[1]*rsh[2]]))
        ucos = tf.reshape(tf.matmul(cosine, r), tf.stack([k, k, rsh[1], rsh[2]]))
        usin = tf.reshape(tf.matmul(sine, r), tf.stack([k, k, rsh[1], rsh[2]]))
        if P is not None:
            # Rotate basis matrices
            ucos_ = tf.cos(P[m])*ucos + tf.sin(P[m])*usin
            usin = -tf.sin(P[m])*ucos + tf.cos(P[m])*usin
            ucos = ucos_
        filters[m] = (ucos, usin)
    return filters


def n_samples(filter_size):
    return np.maximum(np.ceil(np.pi*filter_size),101) ############## <--- One source of instability


def L2_grid(center, shape):
    # Get neighbourhoods
    lin = np.arange(shape)+0.5
    J, I = np.meshgrid(lin, lin)
    I = I - center[1]
    J = J - center[0]
    return np.vstack((np.reshape(I, -1), np.reshape(J, -1)))


def get_weights_dict(shape, max_order, std_mult=0.4, n_rings=None, name='W'):
    """Return a dict of weights.

    shape: list of filter shape [h,w,i,o] --- note we use h=w
    max_order: returns weights for m=0,1,...,max_order, or if max_order is a
    tuple, then it returns orders in the range.
    std_mult: He init scaled by std_mult (default 0.4)
    name: (default 'W')
    dev: (default /cpu:0)
    """
    if isinstance(max_order, int):
        orders = range(-max_order, max_order+1)
    else:
        diff = max_order[1]-max_order[0]
        orders = range(-diff, diff+1)
    weights_dict = {}
    for i in orders:
        if n_rings is None:
            n_rings = np.maximum(shape[0]/2, 2)
        sh = [n_rings,] + shape[2:]
        nm = name + '_' + str(i)
        weights_dict[i] = get_weights(sh, std_mult=std_mult, name=nm)
    return weights_dict


def get_phase_dict(n_in, n_out, max_order, name='b'):
    """Return a dict of phase offsets"""
    if isinstance(max_order, int):
        orders = range(-max_order, max_order+1)
    else:
        diff = max_order[1]-max_order[0]
        orders = range(-diff, diff+1)
    phase_dict = {}
    for i in orders:
        init = np.random.rand(1,1,n_in,n_out) * 2. *np.pi
        init = np.float32(init)
        phase = tf.get_variable(name+'_'+str(i), dtype=tf.float32,
                                shape=[1,1,n_in,n_out],
            initializer=tf.constant_initializer(init))
        phase_dict[i] = phase
    return phase_dict