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
flags.DEFINE_enum("mode", None, ["MLMC", "MC"], "Running mode: MLMC or MC")
flags.DEFINE_string("eval_folder",None,"The folder name for storing evaluation results")
flags.DEFINE_enum("payoff",'images',['images','activations','variance'],"Payoff functions for MLMC")
flags.DEFINE_list("acc",[],"Accuracies for MLMC")
flags.DEFINE_float("DDIMeta",0.,"DDIM eta")
flags.DEFINE_enum('MLMCsampler','EM',['EM','DDIM','TEM','EXPINT','SKROCK'],"Sampler to use for MLMC")
flags.DEFINE_boolean('adaptive',False,"Use adaptive (EM) sampling")
flags.mark_flags_as_required(["workdir", "config", "mode","eval_folder"])

def main(argv):
  print(f'DDIM eta={FLAGS.DDIMeta}')
  if FLAGS.mode == "MLMC":
    # Run the evaluation pipeline
    MLMC.mlmc_test(FLAGS.config,FLAGS.eval_folder,FLAGS.workdir,FLAGS.payoff,[float(a) for a in FLAGS.acc],
                   FLAGS.MLMCsampler,FLAGS.adaptive,FLAGS.DDIMeta,MLMC_=True)
  elif FLAGS.mode == "MC":
    MLMC.mlmc_test(FLAGS.config,FLAGS.eval_folder,FLAGS.workdir,FLAGS.payoff,sampler=FLAGS.MLMCsampler,DDIMeta=FLAGS.DDIMeta,MLMC_=False)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
