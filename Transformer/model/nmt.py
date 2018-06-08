# rnnsearch.py
# author: Wei Xiangpeng
# email: weixiangpeng@iie.ac.cn

import ops
import numpy as np
import tensorflow as tf
import math

from utils import function
from search import beam, select_nbest


def add_position_embedding(x, min_timescale=1.0, max_timescale=1.0e4, name=None):
    """
    This function adds a bunch of sinusoids of different frequencies to a Tensor.
    :param x: A tensor with shape [batch, length, channels]
    :param min_timescale: A floating point number
    :param max_timescale: A floating point number
    :param name: An optional string
    
    :returns: a Tensor the same shape as x.
    """
    with tf.name_scope(name or "add_position_embedding"):
        length = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2

        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1)
        )
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment
        )

        scaled_time = (tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0))
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])
        
        return x + signal


def residual_fn(x, y, keep_prob=1.0):
    if keep_prob < 1.0:
        y = tf.nn.dropout(y, keep_prob)
    return x + y


def ffn_layer(inputs, hidden_size, output_size, keep_prob=1.0, dtype=None, scope=None):
    with tf.variable_scope(scope or "ffn_layer", dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = ops.nn.linear(inputs, hidden_size, True, True)
            hidden = tf.nn.relu(hidden)

        if keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)
        
        with tf.variable_scope("output_layer"):
            output = ops.nn.linear(hidden, output_size, True, True)

        return output


def split_heads(inputs, num_heads, name=None):
    with tf.name_scope(name or "split_heads"):
        x = inputs
        n = num_heads
        old_shape = x.get_shape().dims
        ndims = x.shape.ndims

        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
        ret.set_shape(new_shape)
        perm = [0, ndims - 1] + [i for i in range(1, ndims - 1)] + [ndims]
        return tf.transpose(ret, perm)


def combine_heads(inputs, name=None):
    """ Combine heads
    :param inputs: A tensor with shape [batch, heads, step, hidden]
    :param name: An optional string
    :returns: A tensor with shape [batch, step, heads * hidden]
    """
    with tf.name_scope(name or "combine_heads"):
        x = inputs
        x = tf.transpose(x, [0, 2, 1, 3])
        old_shape = x.get_shape().dims
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        x = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
        x.set_shape(new_shape)

        return x


def attention_mask(inputs, mode, inf=-1e9, name=None):
    """
    A mask tensor used in attention mechanism
    """
    with tf.name_scope(name or "attention_mask"):
        if mode == "causal":
            # for masked multihead attention of decoder
            length = inputs
            lower_triangle = tf.matrix_band_part(
                tf.ones([length, length]), -1, 0
            )
            ret = inf * (1.0 - lower_triangle)
            return tf.reshape(ret, [1, 1, length, length])
        elif mode == "masking":
            mask = inputs
            ret = (1.0 - mask) * inf
            return tf.expand_dims(tf.expand_dims(ret, 1), 1)
        elif mode == "proximal":
            length = inputs
            r = tf.to_float(tf.range(length))
            diff = tf.expand_dims(r, 0) - tf.expand_dims(r, 1)
            m = tf.expand_dims(tf.expand_dims(-tf.log(1 + tf.abs(diff)), 0), 0)
            return m
        elif mode == "distance":
            length, distance = inputs
            distance = tf.where(distance > length, 0, distance)
            distance = tf.cast(distance, tf.int64)
            lower_triangle = tf.matrix_band_part(
                tf.ones([length, length]), -1, 0
            )
            mask_triangle = 1.0 - tf.matrix_band_part(
                tf.ones([length, length]), distance - 1, 0
            )
            ret = inf * (1.0 - lower_triangle + mask_triangle)
            return tf.reshape(ret, [1, 1, length, length])
        else:
            raise ValueError("Unknown mode %s" % mode)


def multiplicative_attention(queries, keys, values, mask, keep_prob=1.0, scope=None):
    with tf.name_scope(scope or "multiplicative_attention"):
        # shape: [batch, heads, step, step]
        logits = tf.matmul(queries, keys, transpose_b=True)
        if mask is not None:
            logits += mask

        weights = tf.nn.softmax(logits, name="attention_weights")

        if keep_prob < 1.0:
            weights = tf.nn.dropout(weights, keep_prob)
        
        outputs = tf.matmul(weights, values)
        
        return {"weights": weights, "outputs": outputs}


def multihead_attention(queries, memories, mask, num_heads, key_size, value_size, output_size, 
                        keep_prob=1.0, output=True, state=None, dtype=None, scope=None):
    """ Multi-head scaled-dot-product attention with input/output transformations.
    :param queries: A tensor with shape [batch, step, hidden]
    :param memories: A tensor with shape [batch, step, hidden]
    :param mask: A tensor, for masking attention
    :param output: Whether to use output transformation
    :param state: An optional dictionary used for incremental decoding

    :returns: A dict with the following keys:
        weights: A tensor with shape [batch, heads, step, step]
        outputs: A tensor with shape [batch, step, hidden]
    """
    
    if key_size % num_heads != 0:
        raise ValueError("Key size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (key_size, num_heads))

    if value_size % num_heads != 0:
        raise ValueError("Value size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (value_size, num_heads))

    with tf.variable_scope(scope or "multihead_attention", dtype=dtype):
        next_state = {}

        if memories is None:
            # self attention
            size = key_size * 2 + value_size
            combined = ops.nn.linear(queries, size, True, True, scope="qkv_transform")
            q, k, v = tf.split(combined, [key_size, key_size, value_size], axis=-1)

            if state is not None:
                # for decoding
                k = tf.concat([state["key"], k], axis=1)
                v = tf.concat([state["value"], v], axis=1)
                next_state["key"] = k
                next_state["value"] = v
        else:
            q = ops.nn.linear(queries, key_size, True, True, scope="q_transform")
            combined = ops.nn.linear(memories, key_size + value_size, True, scope="kv_transform")
            k, v = tf.split(combined, [key_size, value_size], axis=-1)

        # split heads
        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)
        
        # scale query
        key_depth_per_head = key_size // num_heads
        q *= key_depth_per_head ** -0.5

        # attention
        results = multiplicative_attention(q, k, v, mask, keep_prob)

        # combine heads
        weights = results["weights"]
        x = combine_heads(results["outputs"])

        if output:
            outputs = ops.nn.linear(x, output_size, True, True, scope="output_transform")
        else:
            outputs = x

        outputs = {"weights": weights, "outputs": outputs}

        if state is not None:
            outputs["state"] = next_state
        
        return outputs


def transformer_encoder(inputs, mask, num_layers, num_heads, hidden_size, attention_dropout, 
                        residual_dropout, filter_size, relu_dropout, dtype=None, scope=None):
    with tf.variable_scope(scope or "encoder", dtype=dtype):
        x = inputs
        for layer in range(num_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = multihead_attention(
                        x,
                        None,
                        mask,
                        num_heads,
                        hidden_size,
                        hidden_size,
                        hidden_size,
                        1.0 - attention_dropout
                    )
                    y = y["outputs"]
                    x = residual_fn(x, y, 1.0 - residual_dropout)
                    x = ops.nn.layer_norm(x)

                with tf.variable_scope("feed_forward"):
                    y = ffn_layer(
                        x,
                        filter_size,
                        hidden_size,
                        1.0 - relu_dropout,
                    )
                    x = residual_fn(x, y, 1.0 - residual_dropout)
                    x = ops.nn.layer_norm(x)
        
        outputs = x

        return outputs


def transformer_decoder(inputs, memory, mask, mem_mask, num_layers, num_heads, hidden_size, 
                        attention_dropout, residual_dropout, filter_size, relu_dropout, 
                        state=None, dtype=None, scope=None):
    with tf.variable_scope(scope or "decoder", dtype=dtype):
        x = inputs
	next_state = {}
        for layer in range(num_layers):
	    layer_name = "layer_%d" % layer
            with tf.variable_scope(layer_name):
		layer_state = state[layer_name] if state is not None else None
				
                with tf.variable_scope("self_attention"):
                    y = multihead_attention(
                        x,
                        None,
                        mask,
                        num_heads,
                        hidden_size,
                        hidden_size,
                        hidden_size,
                        1.0 - attention_dropout,
                        state=layer_state
                    )
                    
                    if layer_state is not None:
                        next_state[layer_name] = y["state"]

                    y = y["outputs"]
                    x = residual_fn(x, y, 1.0 - residual_dropout)
                    x = ops.nn.layer_norm(x)

                with tf.variable_scope("encdec_attention"):
                    y = multihead_attention(
                        x,
                        memory,
                        mem_mask,
                        num_heads,
                        hidden_size,
                        hidden_size,
                        hidden_size,
                        1.0 - attention_dropout,
                    )

                    y = y["outputs"]
                    x = residual_fn(x, y, 1.0 - residual_dropout)
                    x = ops.nn.layer_norm(x)

                with tf.variable_scope("feed_forward"):
                    y = ffn_layer(
                        x,
                        filter_size,
                        hidden_size,
                        1.0 - relu_dropout,
                    )

                    x = residual_fn(x, y, 1.0 - residual_dropout)
                    x = ops.nn.layer_norm(x)
        
        outputs = x
        
        if state is not None:
            return outputs, next_state

        return outputs


class NMT:

    def __init__(self, num_layers, num_heads, attention_dropout, residual_dropout, relu_dropout, 
                 emb_size, hidden_size, filter_size, l1_vocab_size, l2_vocab_size, l1_word2vec, 
                 l2_word2vec, **option):

        if "initializer" in option:
            initializer = option["initializer"]
        else:
            initializer = None

        def prediction(inputs, output_size):
            
            features = [inputs]
            logits = ops.nn.linear(features, output_size, True, True, scope="logits")
            
            logits = tf.reshape(logits, [-1, output_size])

            return logits

        # training graph
        with tf.variable_scope("rnnsearch", initializer=initializer):
            l1_seq = tf.placeholder(tf.int32, [None, None], "l1_sequence")
            l1_len = tf.placeholder(tf.int32, [None], "l1_length")

            l2_seq = tf.placeholder(tf.int32, [None, None], "l2_sequence")
            l2_len = tf.placeholder(tf.int32, [None], "l2_length")

            with tf.device("/cpu:0"):
                l1_embedding = tf.get_variable("l1_embedding",
                                               initializer=l1_word2vec,
                                               dtype=tf.float32)
                l2_embedding = tf.get_variable("l2_embedding",
                                               initializer=l2_word2vec,
                                               dtype=tf.float32)
                l1_inputs = tf.gather(l1_embedding, l1_seq) # shape=[batch, step, dim]
                l2_inputs = tf.gather(l2_embedding, l2_seq) # shape=[batch, step, dim]
            
            # encoder
            l1_mask = tf.sequence_mask(l1_len, maxlen=tf.shape(l1_seq)[1], dtype=tf.float32)

            with tf.variable_scope("encoder"):
                emb_bias = tf.get_variable("emb_bias", [emb_size])
            
            l1_inputs = l1_inputs * (emb_size ** 0.5)
            l1_inputs = l1_inputs * tf.expand_dims(l1_mask, -1)
            l1_inputs = tf.nn.bias_add(l1_inputs, emb_bias)
            
            l1_inputs = add_position_embedding(l1_inputs)
            
            enc_attn_mask = attention_mask(l1_mask, "masking")
            
            if residual_dropout > 0.0:
                keep_prob = 1.0 - residual_dropout
                l1_inputs = tf.nn.dropout(l1_inputs, keep_prob)
            
            annotation = transformer_encoder(l1_inputs, enc_attn_mask, num_layers, num_heads, 
                                             hidden_size, attention_dropout, residual_dropout, 
                                             filter_size, relu_dropout)

	    # decoder
            l2_mask = tf.sequence_mask(l2_len, maxlen=tf.shape(l2_seq)[1], dtype=tf.float32)
            l2_inputs = l2_inputs * (emb_size ** 0.5)
            l2_inputs = l2_inputs * tf.expand_dims(l2_mask, -1)
            
            dec_attn_mask = attention_mask(tf.shape(l2_seq)[1], "causal")

            shift_inputs = tf.pad(l2_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
            shift_inputs = add_position_embedding(shift_inputs)

            if residual_dropout > 0.0:
                keep_prob = 1.0 - residual_dropout
                shift_inputs = tf.nn.dropout(shift_inputs, keep_prob)

            decoder_output = transformer_decoder(shift_inputs, annotation, dec_attn_mask,
                                                 enc_attn_mask, num_layers, num_heads, 
                                                 hidden_size, attention_dropout, 
                                                 residual_dropout, filter_size, relu_dropout)
            
            with tf.variable_scope("decoder"):
                logits = prediction(decoder_output, l2_vocab_size)

            labels = tf.reshape(l2_seq, [-1])
            ce = ops.nn.smoothed_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels,
                smoothing=0.1,
                normalize=True
            )
            ce = tf.reshape(ce, tf.shape(l2_seq))
            cost = tf.reduce_sum(ce * l2_mask) / tf.reduce_sum(l2_mask)

        # ******************************* Training Graph End *********************************
        train_inputs = [l1_seq, l1_len, l2_seq, l2_len]
        train_outputs = [cost]
        
        # ******************************** Decoding L1 -> L2 *********************************
        with tf.variable_scope("rnnsearch", reuse=True):
            partial_translation = tf.placeholder(tf.int32, [None, None], "partial_translation")
            prev_state = {
                "layer_%d" % i: {
                    "key": tf.placeholder(tf.float32, [None, None, hidden_size], 
                                          "layer_%d" % i + "_key"),
                    "value": tf.placeholder(tf.float32, [None, None, hidden_size], 
                                            "layer_%d" % i + "_value")
                } for i in range(num_layers)
            }

            with tf.device("/cpu:0"):
                l1_embedding = tf.get_variable("l1_embedding",
                                                initializer=l1_word2vec,
                                                dtype=tf.float32)
                l2_embedding = tf.get_variable("l2_embedding",
                                                initializer=l2_word2vec,
                                                dtype=tf.float32)

                l1_inputs = tf.gather(l1_embedding, l1_seq)
                l2_inputs = tf.gather(l2_embedding, partial_translation)
	    
            cond = tf.equal(partial_translation, 0)
            cond = tf.cast(cond, tf.float32)
            l2_inputs = l2_inputs * (1.0 - tf.expand_dims(cond, -1))

            # encoder
            l1_mask = tf.sequence_mask(l1_len, maxlen=tf.shape(l1_seq)[1], dtype=tf.float32)
            with tf.variable_scope("encoder"):
                emb_bias = tf.get_variable("emb_bias", [emb_size])
            
            l1_inputs = l1_inputs * (emb_size ** 0.5)
            l1_inputs = l1_inputs * tf.expand_dims(l1_mask, -1)
            l1_inputs = tf.nn.bias_add(l1_inputs, emb_bias)
            l1_inputs = add_position_embedding(l1_inputs)

            enc_attn_mask = attention_mask(l1_mask, "masking")
            
            annotation = transformer_encoder(l1_inputs, enc_attn_mask, num_layers, num_heads,
                                             hidden_size, 0.0, 0.0, filter_size, 0.0)
            
            # decoder
            l2_inputs = l2_inputs * (emb_size ** 0.5)
            l2_inputs = add_position_embedding(l2_inputs)

            query = l2_inputs[:, -1:, :]
            
            decoder_outputs = transformer_decoder(query, annotation, None, enc_attn_mask, 
                                                  num_layers, num_heads, hidden_size, 
                                                  attention_dropout, residual_dropout, 
                                                  filter_size, relu_dropout, state=prev_state)
	    decoder_output, decoder_state = decoder_outputs
            decoder_output = decoder_output[:, -1:, :]
	    with tf.variable_scope("decoder"):
                logits = prediction(decoder_output, l2_vocab_size)
		probs = tf.nn.softmax(logits)
            
        encoding_inputs = [l1_seq, l1_len]
        encoding_outputs = [annotation, enc_attn_mask]
        encode = function(encoding_inputs, encoding_outputs)

        prediction_inputs = [partial_translation, prev_state, annotation, enc_attn_mask]
        prediction_outputs = [probs, decoder_state]
        predict = function(prediction_inputs, prediction_outputs)

        self.cost = cost

        self.inputs = train_inputs
        self.outputs = train_outputs
        self.encode = encode
        self.predict = predict
        self.option = option


def beamsearch(model, seq, seqlen=None, beamsize=10, normalize=False,
               maxlen=None, minlen=None):
    size = beamsize
    vocabulary = model.option["vocabulary"]
    eos_symbol = model.option["eos"]
    
    hidden_size = model.option["hidden"]
    num_layers = model.option["num_layers"]

    encode = model.encode
    predict = model.predict
    vocab = vocabulary[1][1]
    eosid = vocabulary[1][0][eos_symbol]
    
    batch_dim = 0
    time_dim = 1

    if seqlen is None:
        seq_len = np.array([seq.shape[time_dim]])
    else:
        seq_len = seqlen

    if maxlen is None:
        maxlen = seq_len[0] * 3

    if minlen is None:
        minlen = seq_len[0] / 2

    annotation, enc_attn_mask = encode(seq, seq_len)
    batch = annotation.shape[batch_dim]
    state = {
        "layer_%d" % i: {
            "key": np.zeros([batch, 0, hidden_size], "float32"),
            "value": np.zeros([batch, 0, hidden_size], "float32")
        } for i in range(num_layers)
    }

    initial_beam = beam(size)
    initial_beam.candidate = [[eosid]]
    initial_beam.alignment = [[-1]]
    initial_beam.score = np.zeros([1], "float32")

    hypo_list = []
    beam_list = [initial_beam]

    for k in range(maxlen):
        if size == 0:
            break

        prev_beam = beam_list[-1]
        candidate = prev_beam.candidate
        num = len(prev_beam.candidate)
        #last_words = np.array(map(lambda t: t[-1], candidate), "int32")
        partial_translation = np.array(candidate, "int32")

        batch_annot = np.repeat(annotation, num, batch_dim)
        batch_mask = np.repeat(enc_attn_mask, num, batch_dim)

        prob_dist, state = predict(partial_translation, state, batch_annot, batch_mask)
        
	logprobs = np.log(prob_dist)
	
	# force to don't select eos
        if k < minlen:
            logprobs[:, eosid] = -np.inf

        # force to add eos symbol
        if k == maxlen - 1:
            # copy
            eosprob = logprobs[:, eosid].copy()
            logprobs[:, :] = -np.inf
            logprobs[:, eosid] = eosprob

        next_beam = beam(size)
        outputs = next_beam.prune(logprobs, lambda x: x[-1] == eosid, prev_beam)

        hypo_list.extend(outputs[0])
        batch_indices, word_indices = outputs[1:]
        size -= len(outputs[0])
        
        new_state = {}
        for key, value in state.items():
            state_k = select_nbest(value["key"], batch_indices)
            state_v = select_nbest(value["value"], batch_indices)
            
            new_state[key] = {
                "key": state_k,
                "value": state_v
            }

        state = new_state
        beam_list.append(next_beam)

    # postprocessing
    if len(hypo_list) == 0:
        score_list = [0.0]
        hypo_list = [[eosid]]
    else:
        score_list = [item[1] for item in hypo_list]
        # exclude bos symbol
        hypo_list = [item[0][1:] for item in hypo_list]

    for i, (trans, score) in enumerate(zip(hypo_list, score_list)):
        count = len(trans)
        if count > 0:
            if normalize:
                length_penalty = (float(5. + count)) ** 1.0 / (5. + 1.)
                score_list[i] = score / length_penalty
            else:
                score_list[i] = score

    # sort
    hypo_list = np.array(hypo_list)[np.argsort(score_list)]
    score_list = np.array(sorted(score_list))

    output = []

    for trans, score in zip(hypo_list, score_list):
        trans = map(lambda x: vocab[x], trans)
        output.append((trans, score))

    return output
