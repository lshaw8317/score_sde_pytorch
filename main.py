# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation"""

import run_lib
import argparse
import sys
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import torch
import tensorflow as tf
import MLMC
FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string("eval_folder",None,"The folder name for storing evaluation results")
flags.DEFINE_enum("payoff",'images',['images','activations','variance'],"Payoff functions for MLMC")
flags.DEFINE_list("acc",[0.1,.05,.01,.005,.001,.0005,.0001],"Accuracies for MLMC")
flags.DEFINE_enum('MLMCsampler','EM',['EM','DDIM'],"Sampler to use for MLMC")
flags.mark_flags_as_required(["workdir", "config", "mode","eval_folder"])

def main(argv):
  if FLAGS.mode == "train":
    # Create the working directory
    tf.io.gfile.makedirs(FLAGS.workdir)
    # Set logger so that it outputs to both console and file
    # Make logging work for both disk and Google Cloud Storage
    gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    # Run the training pipeline
    run_lib.train(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == "eval":
    # Run the evaluation pipeline
    print(f'DDIM eta={FLAGS.config.mlmc.DDIM_eta}')
    MLMC.mlmc_test(FLAGS.config,FLAGS.eval_folder,FLAGS.workdir,FLAGS.payoff,[float(a) for a in FLAGS.acc],FLAGS.MLMCsampler)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
