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
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import registry

import tensorflow as tf


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
    p.modality = {"inputs": modalities.SymbolModality,
                  "targets": modalities.RealL2LossModality}
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

  def eval_metrics(self):
    eval_metrics = [
      metrics.Metrics.PEARSON
    ]
    return eval_metrics