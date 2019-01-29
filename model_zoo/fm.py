
import tensorflow as tf

class FMModelParams(object):
  """ class for initializing weights """
  def __init__(self, feature_size, embedding_size):
    self._feature_size = feature_size
    self._embedding_size = embedding_size

  def initialize_weights(self):
    """ init fm  weights
    Returns
    weights:
      feature_embeddings:  vi, vj second order params
      weights_first_order: wi first order params
      bias: b  bias
    """

    weights = dict()
    weights_initializer=tf.glorot_normal_initializer()
    bias_initializer=tf.constant_initializer(0.0)
    weights["feature_embeddings"] = tf.get_variable(
        name='weights',
        dtype=tf.float32,
        initializer=weights_initializer,
        shape=[self._feature_size, self._embedding_size])
    weights["weights_first_order"] = tf.get_variable(
        name='vectors',
        dtype=tf.float32,
        initializer=weights_initializer,
        shape=[self._feature_size, 1])
    weights["fm_bias"] = tf.get_variable(
        name='bias',
        dtype=tf.float32,
        initializer=bias_initializer,
        shape=[1])
    return weights

class FMModel(object):
  """ FM implementation """

  @staticmethod
  def fm_model_fn(features, labels, mode, params):
    """ build tf model"""

    #parse params
    embedding_size = params['embedding_size']
    feature_size = params['feature_size']
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    field_size = params['field_size']
    optimizer_used = params['optimizer']

    #parse features
    feature_idx = features["feature_idx"]
    feature_idx = tf.reshape(feature_idx, shape=[batch_size, field_size])
    labels = tf.reshape(labels, shape=[batch_size, 1])
    feature_values = features["feature_values"]
    feature_values = tf.reshape(feature_values, shape=[batch_size, field_size, 1])

    # tf fm weights
    tf_model_params = FMModelParams(feature_size, embedding_size)
    weights = tf_model_params.initialize_weights()
    embeddings = tf.nn.embedding_lookup(
        weights["feature_embeddings"],
        feature_idx
        )
    weights_first_order = tf.nn.embedding_lookup(
        weights["weights_first_order"],
        feature_idx
        )
    bias = weights['fm_bias']

    #build function
    ##first order
    first_order = tf.multiply(feature_values, weights_first_order)
    first_order = tf.reduce_sum(first_order, 2)
    first_order = tf.reduce_sum(first_order, 1, keepdims=True)

    ##second order
    ### feature * embeddings
    f_e_m = tf.multiply(feature_values, embeddings)
    ###  square(sum(feature * embedding))
    f_e_m_sum = tf.reduce_sum(f_e_m, 1)
    f_e_m_sum_square = tf.square(f_e_m_sum)
    ###  sum(square(feature * embedding))
    f_e_m_square = tf.square(f_e_m)
    f_e_m_square_sum = tf.reduce_sum(f_e_m_square, 1)

    second_order = f_e_m_sum_square - f_e_m_square_sum

    second_order = tf.reduce_sum(second_order, 1, keepdims=True)

    ##final objective function
    logits = second_order + first_order + bias
    predicts = tf.sigmoid(logits)

    ##loss function
    sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    sigmoid_loss = tf.reduce_mean(sigmoid_loss)
    loss = sigmoid_loss

    #train op
    if optimizer_used == 'adagrad':
      optimizer = tf.train.AdagradOptimizer(
          learning_rate=learning_rate,
          initial_accumulator_value=1e-8)
    elif optimizer_used == 'adam':
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    else:
      raise Exception("unknown optimizer", optimizer_used)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())


    #metric
    eval_metric_ops = {
        "auc": tf.metrics.auc(labels, predicts)
        }
    predictions = {"prob": predicts}

    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        predictions=predicts,
        loss=loss,
        eval_metric_ops=eval_metric_ops,
        train_op=train_op)
