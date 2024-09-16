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
import pickle
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
import MLMC as MLMC
from easydict import EasyDict as edict

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("eval_folder",None,"The folder name for storing evaluation results")

flags.DEFINE_enum("payoff",'secondmoment',['mean','secondmoment'],"Payoff functions for MLMC")
flags.DEFINE_list("acc",[],"Accuracies for MLMC")
flags.DEFINE_list("abL",None,"Convergence exponents alpha,beta and Lmin")
flags.DEFINE_integer("Lmax",11,"Maximum allowed L")
flags.DEFINE_integer("M",2,"Mesh refinement factor")
flags.DEFINE_enum('MLMCsampler','EXPINT',['EM','EXPINT'],"Sampler to use for MLMC")
flags.DEFINE_boolean('probflow',False,"Use probflow ODE for sampling")
flags.DEFINE_float("accsplit",1./1.41,"accsplit for var-bias split")
flags.DEFINE_boolean("conditional",False,"Denoising problem")
flags.DEFINE_float("conditional_noise",.4,"Denoising problem noise")

flags.mark_flags_as_required(["workdir", "config","eval_folder"])

def main(argv):
  if FLAGS.conditional:
    import numpy as np
    #Load an image from the Cifar10 dataset
    with open(os.path.join('data',"cifar-10-batches-py/", "data_batch_1"),'rb') as f:
      x0 = pickle.load(f,encoding='latin1')['data'].reshape(-1, 3, 32, 32)[42]/255.
    torch.manual_seed(0)
    x0=2*torch.tensor(x0)-1.
    v=x0+FLAGS.conditional_noise*torch.randn_like(x0)
    denoisedir=os.path.join(FLAGS.eval_folder,f'Denoising_{FLAGS.conditional_noise}')
    os.makedirs(denoisedir,exist_ok=True)
    with open(os.path.join(denoisedir, "x0.pt"), "wb") as f:
      torch.save(x0.cpu(),f)
    with open(os.path.join(denoisedir, "v.pt"), "wb") as f:
      torch.save(v.cpu(),f)
    try:
      with open(os.path.join(denoisedir, "mask.pt"), "rb") as f:
        MASK=torch.load(mask)
    except:
      print('No mask for conditional sampling found. Defaulting to identity.')
      MASK=torch.ones_like(v)
    conditional=edict(obsV=v,noise=FLAGS.conditional_noise,MASK=MASK)

  else:
    conditional=None


  # Run the evaluation pipeline
  MLMC.mlmc_test(FLAGS.config,FLAGS.eval_folder,FLAGS.workdir,FLAGS.payoff,
                   [float(a) for a in FLAGS.acc],FLAGS.M,FLAGS.Lmax,
                   FLAGS.MLMCsampler,FLAGS.probflow,accsplit=FLAGS.accsplit,
                   abL=FLAGS.abL,conditional=conditional)

if __name__ == "__main__":
  app.run(main)
