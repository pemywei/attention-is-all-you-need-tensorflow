# nn.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import tensorflow as tf


def linear(inputs, output_size, bias, concat=True, dtype=None, scope=None):
    with tf.variable_scope(scope or "linear"):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        input_size = [item.get_shape()[-1].value for item in inputs]

        if len(inputs) != len(input_size):
            raise RuntimeError("inputs and input_size unmatched!")

        output_shape = tf.concat([tf.shape(inputs[0])[:-1], [output_size]], axis=0)

        # Flatten to 2D
        inputs = [tf.reshape(inp, [-1, inp.shape[-1].value]) for inp in inputs]

        results = []

        if concat:
            input_size = sum(input_size)
            inputs = tf.concat(inputs, 1)
            
            shape = [input_size, output_size]
            matrix = tf.get_variable("matrix", shape, dtype=dtype)
            results.append(tf.matmul(inputs, matrix))
        else:
            for i in range(len(input_size)):
                shape = [input_size[i], output_size]
                name = "matrix_%d" % i
                matrix = tf.get_variable(name, shape, dtype=dtype)
                results.append(tf.matmul(inputs[i], matrix))

        output = tf.add_n(results)

        if bias:
            shape = [output_size]
            bias = tf.get_variable("bias", shape, dtype=dtype)
            output = tf.nn.bias_add(output, bias)

        output = tf.reshape(output, output_shape)
        return output

def layer_norm(inputs, epsilon=1e-6, dtype=None, scope=None):
    with tf.variable_scope(scope or "layer_norm", dtype=dtype):
        channel_size = inputs.get_shape().as_list()[-1]

        scale = tf.get_variable("scale", shape=[channel_size], 
                                initializer=tf.ones_initializer())
        offset = tf.get_variable("offset", shape=[channel_size], 
                                 initializer=tf.zeros_initializer())

        mean = tf.reduce_mean(inputs, -1, True)
        variance = tf.reduce_mean(tf.square(inputs - mean), -1, True)

        norm_inputs = (inputs - mean) * tf.rsqrt(variance + epsilon)

        return norm_inputs * scale + offset

def smoothed_softmax_cross_entropy_with_logits(**kwargs):
    logits = kwargs.get("logits")
    labels = kwargs.get("labels")
    smoothing = kwargs.get("smoothing") or 0.0
    normalize = kwargs.get("normalize")
    scope = kwargs.get("scope")

    if logits is None or labels is None:
        raise ValueError("Both logits and labels must be provided")

    with tf.name_scope(scope or "smoothed_softmax_cross_entropy_with_logits"):
        if not smoothing:
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            return ce

        # label smoothing
        vocab_size = tf.shape(logits)[1]

        n = tf.to_float(vocab_size - 1)
        p = 1.0 - smoothing
        q = smoothing / n

        soft_targets = tf.one_hot(labels, depth=vocab_size, on_value=p, off_value=q)
        ce = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=soft_targets)
        
        if normalize is False:
            return ce

        normalizing = -(p * tf.log(p) + n * q * tf.log(q + 1e-20))

        return ce - normalizing

def maxout(inputs, output_size, maxpart=2, use_bias=True, dtype=None, scope=None):
    candidate = linear(inputs, output_size * maxpart, use_bias,
                       dtype=dtype, scope=scope or "maxout")
    shape = tf.concat([tf.shape(candidate)[:-1], [output_size, maxpart]], axis=0)
    
    value = tf.reshape(candidate, shape)
    output = tf.reduce_max(value, -1)

    return output
