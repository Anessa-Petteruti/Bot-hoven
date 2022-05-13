import tensorflow as tf
import numpy as np

class Relative_multi_head_attention(tf.keras.layers.Layer):

    def __init__(self, d_model, n_heads, name='mha'):
        super(Relative_multi_head_attention, self).__init__(name=name)

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.sqrt_dk = tf.math.sqrt(self.d_model * 1.0)

        self.w_q = tf.keras.layers.Dense(d_model, use_bias=False, name='w_q')
        self.w_k_e = tf.keras.layers.Dense(
            d_model, use_bias=False, name='w_k_e')
        self.w_k_r = tf.keras.layers.Dense(
            d_model, use_bias=False, name='w_k_r')
        self.w_v = tf.keras.layers.Dense(d_model, use_bias=False, name='w_v')

        self.final = tf.keras.layers.Dense(
            d_model, use_bias=False, activation='gelu', name='final')

        u_init = tf.random_normal_initializer()
        self.u_param = tf.Variable(
            initial_value=u_init(shape=(1, 1, self.n_heads, self.d_head), dtype="float32"), trainable=True, name='u_param')

        v_init = tf.random_normal_initializer()
        self.v_param = tf.Variable(
            initial_value=v_init(shape=(1, 1, self.n_heads, self.d_head), dtype="float32"), trainable=True, name='v_param'
        )

    def call(self, inputs, seq_len, mask, rel_enc):

        x_tilde = inputs

        batch_size = x_tilde.shape[0]
        full_len = x_tilde.shape[1]

        x = x_tilde[:, -seq_len:, :]

        full_len = x_tilde.shape[1]

        q = self.w_q(x)
        k = self.w_k_e(x_tilde)
        v = self.w_v(x_tilde)

        q = tf.reshape(q, [batch_size, seq_len, self.n_heads, self.d_head])
        k = tf.reshape(k, [batch_size, full_len, self.n_heads, self.d_head])
        v = tf.reshape(v, [batch_size, full_len, self.n_heads, self.d_head])

        A_C = tf.einsum('bsnd,bfnd->bnsf', q + self.u_param, k)

        Q = self.w_k_r(rel_enc)
        Q = tf.reshape(Q, [full_len, self.n_heads, self.d_head])

        B_D_hat = tf.einsum('bsnd, fnd->bnsf', q + self.v_param, Q)

        B_D = self.rel_enc_shift(B_D_hat)

        attention_score = A_C + B_D
        attention_score = attention_score / self.sqrt_dk

        attention_score += (mask * -1e10)

        attention_weights = tf.nn.softmax(attention_score, axis=-1)
        max_weights = tf.math.reduce_max(attention_weights, axis=-1)
        max_weights = tf.math.reduce_max(max_weights, axis=-1)
        attention_loss = tf.math.reduce_mean(max_weights)

        attention_output = tf.einsum('bnsf,bfnd->bsnd', attention_weights, v)
        attention_output = tf.reshape(
            attention_output, [batch_size, seq_len, self.d_model])

        output = self.final(attention_output)

        return output, attention_weights, attention_loss


class Transformer_block(tf.keras.layers.Layer):

    def __init__(self, d_model, n_heads, dropout_rate, gating_type=None):
        super(Transformer_block, self).__init__()

        assert 0.0 <= dropout_rate < 1

        self.d_model = d_model

        self.rmha = Relative_multi_head_attention(
            d_model=self.d_model, n_heads=n_heads)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.linear1 = tf.keras.layers.Dense(
            self.d_model, activation='gelu', name='linear1')
        self.linear2 = tf.keras.layers.Dense(
            self.d_model, activation='gelu', name='linear2')

    def call(self, inputs, mem, mask, rel_enc, training):

        seq_len = inputs.shape[1]

        if mem is None:
            x_tilde = inputs
        else:
            x_tilde = tf.concat((tf.stop_gradient(mem), inputs), axis=1)

        x_tilde = self.layer_norm1(x_tilde)

        rmha_output, weight_list, attention_loss = self.rmha(
            x_tilde, seq_len, mask, rel_enc)
        rmha_output = self.dropout1(rmha_output, training=training)

        rmha_output = self.gating_layer1((inputs, rmha_output))

        output = self.layer_norm2(rmha_output)
        output = self.linear1(output)
        output = self.linear2(output)
        output = self.dropout2(output, training=training)

        output = self.gating_layer2((rmha_output, output))

        return output, weight_list, attention_loss


class Music_transformer(tf.keras.Model):

    def __init__(self, d_sound, d_delta, n_heads_sound, n_heads_delta, n_heads_combined,
                 n_layers_sound, n_layers_delta, n_layers_combined,
                 n_sounds, n_deltas, dropout_rate, pad_idx,
                 weights_sound=None, weights_delta=None, max_seq_len=2048, gating_type=None):

        super(Music_transformer, self).__init__()

        assert d_sound % n_heads_sound == 0
        assert d_delta % n_heads_delta == 0
        assert (d_sound + d_delta) % n_heads_combined == 0
        assert 0.0 <= dropout_rate < 1.0

        self.d_sound = d_sound
        self.d_delta = d_delta
        self.d_combined = d_sound + d_delta
        self.n_heads_sound = n_heads_sound
        self.n_heads_delta = n_heads_delta
        self.n_heads_combined = n_heads_combined
        self.n_layers_sound = n_layers_sound
        self.n_layers_delta = n_layers_delta
        self.n_layers_combined = n_layers_combined
        self.n_layers_total = n_layers_sound + n_layers_delta + n_layers_combined
        self.n_sounds = n_sounds
        self.n_deltas = n_deltas
        self.dropout_rate = dropout_rate
        self.pad_idx = pad_idx

        if not weights_sound is None:
            weights_sound = tf.constant(weights_sound, dtype=tf.float32)
            assert weights_sound.shape == (self.n_sounds,)

        self.weights_sound = weights_sound

        if not weights_delta is None:
            weights_delta = tf.constant(weights_delta, dtype=tf.float32)
            assert weights_delta.shape == (self.n_deltas,)

        self.weights_delta = weights_delta

        self.emb_layer_sound = tf.keras.layers.Embedding(
            self.n_sounds, self.d_sound)
        self.emb_layer_delta = tf.keras.layers.Embedding(
            self.n_deltas, self.d_delta)
        self.pos_enc = get_pos_encoding(max_seq_len, self.d_combined)

        self.layer_list_sound = []
        for _ in range(self.n_layers_sound):

            layer = Transformer_block(
                self.d_sound, self.n_heads_sound, self.dropout_rate, gating_type)
            self.layer_list_sound.append(layer)

        self.layer_list_delta = []
        for _ in range(self.n_layers_delta):

            layer = Transformer_block(
                self.d_delta, self.n_heads_delta, self.dropout_rate, gating_type)
            self.layer_list_delta.append(layer)

        self.layer_list_combined = []
        for _ in range(self.n_layers_combined):

            layer = Transformer_block(
                self.d_combined, self.n_heads_combined, self.dropout_rate, gating_type)
            self.layer_list_combined.append(layer)

        self.dropout1 = tf.keras.layers.Dropout(
            self.dropout_rate, name='dropout1')
        self.hidden = tf.keras.layers.Dense(
            self.d_combined, activation='gelu', name='hidden')
        self.dropout2 = tf.keras.layers.Dropout(
            self.dropout_rate, name='dropout2')
        self.final_sound = tf.keras.layers.Dense(
            self.n_sounds, name='final_sound')
        self.final_delta = tf.keras.layers.Dense(
            self.n_deltas, name='final_deltas')

    def call(self, inputs, mem_list, next_mem_len, training):

        sounds, deltas = inputs

        batch_size = sounds.shape[0]
        seq_len = sounds.shape[1]

        if mem_list is None:
            mem_len = 0
            mem_list = [None] * self.n_layers_total
        else:
            mem_len = mem_list[0].shape[1]

        full_len = seq_len + mem_len

        mask = self.get_look_ahead_mask(seq_len, mem_len)

        rel_enc_sound = self.pos_enc[:full_len, :self.d_sound]
        rel_enc_sound = tf.reverse(rel_enc_sound, axis=[0])

        rel_enc_delta = self.pos_enc[:full_len, :self.d_delta]
        rel_enc_delta = tf.reverse(rel_enc_delta, axis=[0])

        rel_enc_combined = self.pos_enc[:full_len, :]
        rel_enc_combined = tf.reverse(rel_enc_combined, axis=[0])

        next_mem_list = []
        attention_weight_list = []
        attention_loss_list = []

        sounds = self.emb_layer_sound(sounds)
        sounds = sounds * tf.math.sqrt(tf.cast(self.d_sound, tf.float32))

        for idx, layer in enumerate(self.layer_list_sound):

            next_mem = self.get_next_mem(mem_list[idx], sounds, next_mem_len)
            next_mem_list.append(next_mem)

            sounds, attention_weights, attention_loss = layer(
                sounds, mem_list[idx], mask, rel_enc_sound, training)
            attention_weight_list.append(attention_weights)
            attention_loss_list.append(attention_loss)

        deltas = self.emb_layer_delta(deltas)
        deltas = deltas * tf.math.sqrt(tf.cast(self.d_delta, tf.float32))

        for idx, layer in enumerate(self.layer_list_delta, self.n_layers_sound):

            next_mem = self.get_next_mem(mem_list[idx], deltas, next_mem_len)
            next_mem_list.append(next_mem)

            deltas, attention_weights, attention_loss = layer(
                deltas, mem_list[idx], mask, rel_enc_delta, training)
            attention_weight_list.append(attention_weights)
            attention_loss_list.append(attention_loss)

        x = tf.concat((sounds, deltas), axis=-1)

        for idx, layer in enumerate(self.layer_list_combined, self.n_layers_sound + self.n_layers_delta):

            next_mem = self.get_next_mem(mem_list[idx], x, next_mem_len)
            next_mem_list.append(next_mem)

            x, attention_weights, attention_loss = layer(
                x, mem_list[idx], mask, rel_enc_combined, training)
            attention_weight_list.append(attention_weights)
            attention_loss_list.append(attention_loss)

        x = self.dropout1(x, training=training)
        x = self.hidden(x)
        x = self.dropout2(x, training=training)

        logits_sound = self.final_sound(x)

        logits_delta = self.final_delta(x)

        return logits_sound, logits_delta, next_mem_list, attention_weight_list, attention_loss_list

