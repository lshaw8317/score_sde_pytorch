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
from easydict import EasyDict as edict

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["MLMC", "MC"], "Running mode: MLMC or MC")
flags.DEFINE_string("eval_folder",None,"The folder name for storing evaluation results")

flags.DEFINE_enum("payoff",'mean',['mean','activations','secondmoment'],"Payoff functions for MLMC")
flags.DEFINE_list("acc",[],"Accuracies for MLMC")
flags.DEFINE_list("abg",[-1,-1,-1],"Convergence exponents alpha,beta,gamma")
flags.DEFINE_integer("Lmax",11,"Maximum allowed L")
flags.DEFINE_integer("M",2,"Mesh refinement factor")
flags.DEFINE_integer("Lmin",2,"Starting l0")
flags.DEFINE_enum('MLMCsampler','EXPINT',['EM','TEM','EXPINT'],"Sampler to use for MLMC")
flags.DEFINE_boolean('adaptive',False,"Use adaptive (EM) sampling")
flags.DEFINE_boolean('probflow',False,"Use probflow ODE for sampling")
flags.DEFINE_float("accsplit",1./1.41,"accsplit for var-bias split")
flags.DEFINE_boolean("conditional",False,"Denoising problem")
flags.DEFINE_float("conditional_noise",.4,"Denoising problem noise")

flags.mark_flags_as_required(["workdir", "config", "mode","eval_folder"])

def main(argv):
  if FLAGS.conditional:
    from torchvision.datasets import CIFAR10
    #Load an image from the Cifar10 dataset                                                                                                                 
    x0 = CIFAR10(root='./data', train=True, download=True).__get_item__(42)/255.
    torch.seed(42)
    v=x0+FLAGS.conditional_noise*torch.randn_like(x0)
    denoisedir=os.path.join(FLAGS.eval_folder,f'Denoising_{FLAGS.conditional_noise}')
    os.mkdir(denoisedir)
    with open(os.path.join(denoisedir, "x0.pt"), "wb") as f:
      torch.save(x0.cpu(),f)
    with open(os.path.join(denoisedir, "v.pt"), "wb") as f:
      torch.save(v.cpu(),f) 
    conditional=edict(obsV=v,noise=FLAGS.conditional_noise)
  else:
    conditional=None


  if FLAGS.mode == "MLMC":
    # Run the evaluation pipeline
    MLMC.mlmc_test(FLAGS.config,FLAGS.eval_folder,FLAGS.workdir,FLAGS.payoff,
                   [float(a) for a in FLAGS.acc],FLAGS.M,FLAGS.Lmin,FLAGS.Lmax,
                   FLAGS.MLMCsampler,FLAGS.adaptive,FLAGS.probflow,MLMC_=True,accsplit=FLAGS.accsplit,
                   abg=(float(a) for a in FLAGS.abg),conditional=conditional)
  elif FLAGS.mode == "MC":
    MLMC.mlmc_test(FLAGS.config,FLAGS.eval_folder,FLAGS.workdir,FLAGS.payoff,
                   M=FLAGS.M,Lmax=FLAGS.Lmax,sampler=FLAGS.MLMCsampler,
                   adaptive=FLAGS.adaptive,probflow=FLAGS.probflow,MLMC_=False)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
