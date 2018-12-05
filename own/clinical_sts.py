# coding=utf-8


"""Clinical Semantic text Similarity Problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import modality
from tensor2tensor.utils import metrics
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import registry

import tensorflow as tf

class SymbolModality(modality.Modality):
  """Modality for sets of discrete symbols.

  Input:
    Embedding.

  Output:
    Linear transformation + softmax.
  """

  @property
  def name(self):
    return "symbol_modality_%d_%d" % (self._vocab_size, self._body_input_depth)

  @property
  def top_is_pointwise(self):
    return True

  @property
  def targets_weights_fn(self):
    weights_fn = common_layers.weights_nonzero

    hp = self._model_hparams
    if hp and hp.prepend_mode != "none":
      assert (hp.prepend_mode == "prepend_inputs_masked_attention" or
              hp.prepend_mode == "prepend_inputs_full_attention")

      if (
          # In masked attention mode, during training, the network try to
          # autoregressively predicting the inputs portion, while the
          # evaluation is only done on the output
          hp.prepend_mode != "prepend_inputs_masked_attention" or
          hp.mode != tf.estimator.ModeKeys.TRAIN):
        weights_fn = common_layers.weights_prepend_inputs_to_targets

    return weights_fn

  def _get_weights(self, hidden_dim=None):
    """Create or get concatenated embedding or softmax variable.

    Args:
      hidden_dim: dim of the variable. Defaults to self._body_input_depth

    Returns:
       a list of self._num_shards Tensors.
    """
    if hidden_dim is None:
      hidden_dim = self._body_input_depth
    num_shards = self._model_hparams.symbol_modality_num_shards
    shards = []
    for i in range(num_shards):
      shard_size = (self._vocab_size // num_shards) + (
          1 if i < self._vocab_size % num_shards else 0)
      var_name = "weights_%d" % i
      shards.append(
          tf.get_variable(
              var_name, [shard_size, hidden_dim],
              initializer=tf.random_normal_initializer(0.0, hidden_dim**-0.5)))
    if num_shards == 1:
      ret = shards[0]
    else:
      ret = tf.concat(shards, 0)
    # Convert ret to tensor.
    if not tf.contrib.eager.in_eager_mode():
      ret = common_layers.convert_gradient_to_tensor(ret)
    return ret

  def bottom_simple(self, x, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
      # Ensure the inputs are 3-D
      if len(x.get_shape()) == 4:
        x = tf.squeeze(x, axis=3)
      while len(x.get_shape()) < 3:
        x = tf.expand_dims(x, axis=-1)

      var = self._get_weights()
      x = common_layers.dropout_no_scaling(
          x, 1.0 - self._model_hparams.symbol_dropout)
      ret = common_layers.gather(var, x)
      if self._model_hparams.multiply_embedding_mode == "sqrt_depth":
        ret *= self._body_input_depth**0.5
      ret *= tf.expand_dims(tf.to_float(tf.not_equal(x, 0)), -1)
      return ret

  def bottom(self, x):
    if (self._model_hparams.shared_embedding_and_softmax_weights or
        self._model_hparams.get("shared_embedding")):
      return self.bottom_simple(x, "shared", reuse=None)
    return self.bottom_simple(x, "input_emb", reuse=None)

  def targets_bottom(self, x):
    if (self._model_hparams.shared_embedding_and_softmax_weights or
        self._model_hparams.get("shared_embedding")):
      try:
        return self.bottom_simple(x, "shared", reuse=True)
      except ValueError:
        # perhaps there were no inputs, and this is a new variable.
        return self.bottom_simple(x, "shared", reuse=None)
    else:
      return self.bottom_simple(x, "target_emb", reuse=None)

  def top(self, body_output, _):
    """Generate logits.

    Args:
      body_output: A Tensor with shape [batch, p0, p1, body_input_depth]
    Returns:
      logits: A Tensor with shape  [batch, p0, p1, ?, vocab_size].
    """
    if self._model_hparams.symbol_modality_skip_top:
      return tf.expand_dims(body_output, 3)

    if self._model_hparams.shared_embedding_and_softmax_weights:
      scope_name = "shared"
      reuse = tf.AUTO_REUSE
    else:
      scope_name = "softmax"
      reuse = False

    with tf.variable_scope(scope_name, reuse=reuse):
      body_output_shape = common_layers.shape_list(body_output)
      var = self._get_weights(body_output_shape[-1])
      if (self._model_hparams.factored_logits and
          self._model_hparams.mode == tf.estimator.ModeKeys.TRAIN):
        # insert channels dimension
        body_output = tf.expand_dims(body_output, 3)
        return common_layers.FactoredTensor(body_output, var)
      else:
        body_output = tf.reshape(body_output, [-1, body_output_shape[-1]])
        logits = tf.matmul(body_output, var, transpose_b=True)
        return tf.reshape(logits,
                          body_output_shape[:-1] + [1, self._vocab_size])


class RealL2LossModality(modality.Modality):
  """Modality for real (i.e. float) vectors with L2 (Gaussian) loss."""
  
  @property
  def top_is_pointwise(self):
    return True

  def bottom(self, x):
    with tf.variable_scope("real"):
      return tf.layers.dense(
          tf.to_float(x), self._body_input_depth, name="bottom")

  def top(self, body_output, _):
    with tf.variable_scope("real"):
      output = tf.layers.dense(body_output, self._vocab_size, name="top")
      print(output)
      return output

  def loss(self, top_out, targets):
    predictions = top_out
    if (len(common_layers.shape_list(top_out)) != len(
        common_layers.shape_list(targets))):
      predictions = tf.squeeze(top_out, axis=[-1])
    with tf.name_scope("l2"):
      weights = self.targets_weights_fn(targets)
      l2 = tf.pow(predictions - targets, 2)
      return tf.reduce_sum(l2 * weights), tf.reduce_sum(weights)


class Text2ScoreProblem(text_problems.Text2TextProblem):
  """Base class for text regression problems."""
  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generate samples of text and label pairs.

    Each yielded dict will be a single example. The inputs should be raw text.
    The score should be an float in [0, self.max_score).

    Args:
      data_dir: final data directory. Typically only used in this method to copy
        over user-supplied vocab files (for example, if vocab_type ==
        VocabType.TOKEN).
      tmp_dir: temporary directory that you can use for downloading and scratch.
      dataset_split: problem.DatasetSplit, which data split to generate samples
        for (for example, training and evaluation).

    Yields:
      {"inputs": text, "score": float}
    """
    raise NotImplementedError()

  # START: Additional subclass interface
  @property
  def max_score(self):
    """The max score."""
    raise NotImplementedError()

  def score_encoder(self, s):
    return s

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    for i, sample in enumerate(
        self.generate_samples(data_dir, tmp_dir, problem.DatasetSplit.TRAIN)):
      yield sample["inputs"]
      if self.max_samples_for_vocab and (i + 1) >= self.max_samples_for_vocab:
        break

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    encoder = self.get_or_create_vocab(data_dir, tmp_dir)
    for sample in generator:
      inputs = encoder.encode(sample["inputs"])
      inputs.append(text_encoder.EOS_ID)
      score = sample["score"]
      yield {"inputs": inputs, "targets": [score]}

  def feature_encoders(self, data_dir):
    encoder = self.get_or_create_vocab(data_dir, None, force_get=True)

    return {
        "inputs": encoder,
        "targets": self.score_encoder
    }

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {"inputs": SymbolModality,
                  "targets": RealL2LossModality}
    p.vocab_size = {"inputs": self._encoders["inputs"].vocab_size,
                    "targets": self.max_score}

  def example_reading_spec(self):
    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.FixedLenFeature([1], tf.float32),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)


class TextSimilarityProblem(Text2ScoreProblem):
  """Base class for text classification problems with multiple inputs.

  For problems where there are multiple input sentences and we wish to concat
  these inputs with a special delimiter. See, for example, NLI tasks.
  """
  CONCAT_TOKEN = "$"

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    for i, sample in enumerate(
        self.generate_samples(data_dir, tmp_dir, problem.DatasetSplit.TRAIN)):
      for inp in sample["inputs"]:
        yield inp
        if self.max_samples_for_vocab and (i + 1) >= self.max_samples_for_vocab:
          break

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    encoder = self.get_or_create_vocab(data_dir, tmp_dir)
    for sample in generator:
      inputs = []
      for idx, inp in enumerate(sample["inputs"]):
        inputs += encoder.encode(inp)
        if idx < len(sample["inputs"]) - 1:
          inputs.append(encoder.encode(self.CONCAT_TOKEN)[0])
      inputs.append(text_encoder.EOS_ID)
      score = sample["score"]
      yield {"inputs": inputs, "targets": [score]}


@registry.register_problem
class Clinical(TextSimilarityProblem):
  """"""

  @property
  def is_generate_per_split(self):
    return True

  @property
  def dataset_splits(self):
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 10,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def approx_vocab_size(self):
    return 2**13      # 8k vocab suffices for this small dataset.

  @property
  def max_score(self):
    """The max score."""
    return 5.0

  def example_generator(self, filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    for idx, line in enumerate(lines):
      lines = line.strip().split('\t')
      if len(lines) == 3:
        score = float(lines[2])
      else:
        score = None
      inputs = [lines[0], lines[1]]
      yield {
          "inputs": inputs,
          "score": score
      }

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    if dataset_split == problem.DatasetSplit.TRAIN:
      data_dir = os.path.join(data_dir, "train")
    else:
      data_dir = os.path.join(data_dir, "test")

    for f in os.listdir(data_dir):
      filename = os.path.join(data_dir, f)
      for example in self.example_generator(filename):
        yield example
