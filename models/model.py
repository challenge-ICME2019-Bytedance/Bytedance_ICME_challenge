""" import necessary packages"""
import tensorflow as tf
from data_io.data_parser import PosShifts, LineParser
from model_zoo.fm import FMModel


class RecommendModelHandler(object):
  """ class for setup recommend model """
  def __init__(self, train_dataset_path, val_dataset_path, save_model_dir,  \
      learning_rate=0.1, num_threads=1, num_epochs=100, batch_size=40,  \
      embedding_size=100, optimizer='adam', task="finish", track=2):
    """ init basic params"""
    self._learning_rate = learning_rate
    self._num_threads = num_threads
    self._num_epochs = num_epochs
    self._batch_size = batch_size
    self._train_dataset_path = train_dataset_path
    self._val_dataset_path = val_dataset_path
    self._save_model_dir = save_model_dir
    self._embedding_size = embedding_size
    self._optimizer = optimizer
    self._task = task
    self._track = track


  def build_model(self):
    """ build recommend model framework"""
    config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'CPU':self._num_threads}),
        log_step_count_steps=20)
    PosShifts(self._track)
    feature_size = PosShifts.get_features_num()
    params={
        'feature_size': feature_size,
        'embedding_size': self._embedding_size,
        'learning_rate': self._learning_rate,
        'field_size': 5,
        'batch_size': self._batch_size,
        'optimizer': self._optimizer}

    model = tf.estimator.Estimator(
        model_fn=FMModel.fm_model_fn,
        model_dir=self._save_model_dir,
        params=params,
        config=config)
    return model

  def prepare_data_fn(self, data_mode='train'):
    """ prepare train, val fn"""
    if data_mode == 'train':
      dataset = tf.data.TextLineDataset(self._train_dataset_path)
    elif data_mode == 'val':
      dataset = tf.data.TextLineDataset(self._val_dataset_path)
    else:
      raise Exception("unknown data_mode", data_mode)

    if self._task == "finish":
      dataset = dataset.map(LineParser.parse_finish_line)
    elif self._task == "like":
      dataset = dataset.map(LineParser.parse_like_line)
    else:
      raise Exception("unknown task", task)

    dataset = dataset.shuffle(buffer_size=300)
    dataset = dataset.repeat(self._num_epochs)
    dataset = dataset.batch(self._batch_size)
    data_iterator = dataset.make_one_shot_iterator()
    idx, features, labels = data_iterator.get_next()
    feature_infos = {}
    feature_infos['feature_idx'] = idx
    feature_infos['feature_values'] = features
    tf.logging.info(labels)
    return feature_infos, labels


  def train(self):
    """
    Train model
    """
    model = self.build_model()
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: self.prepare_data_fn(data_mode='train'))
    val_spec = tf.estimator.EvalSpec(input_fn=lambda: self.prepare_data_fn(data_mode='val'))
    tf.estimator.train_and_evaluate(model, train_spec, val_spec)
