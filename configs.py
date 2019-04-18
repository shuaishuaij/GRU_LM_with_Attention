__AUTHOR__ = 'KALAMITE'
"""THIS SCRIPT CONTAINS CONFIGURATIONS FOR DIFFERENT MODELS. WHEN ACQUIRED,
SIMPLY IMPORT THE CORRESPONDING CONFIGS FROM THIS FILE.
"""
import os


# &&&&&&&&&&&&&&&&&    1    &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# MODEL SETTING
class ModelConfig1(object):
    model_name = 'Yidhra_debug1'
    decoder_attn_method = 'dot'  # 'dot','general','concat'
    hidden_size = 500
    teacher_forcing_ratio = 1.0
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64


# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000


# TRAINING/OPTIMIZATION SETTING
class ExpConfig1(object):
    clip = 50.0
    teacher_forcing_ratio = 1.0
    learning_rate = 0.001
    decoder_learning_ratio = 5.0
    n_iteration = 20
    print_every = 1
    save_every = 5
    save_dir = os.path.join("data", "save")


# &&&&&&&&&&&&&&&&&    2    &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# MODEL SETTING
class ModelConfig2(object):
    model_name = 'Yidhra_2'
    decoder_attn_method = 'dot'  # 'dot','general','concat'
    hidden_size = 1024
    teacher_forcing_ratio = 1.0
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64


# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000


# TRAINING/OPTIMIZATION SETTING
class ExpConfig2(object):
    clip = 50.0
    teacher_forcing_ratio = 1.0
    learning_rate = 0.001
    decoder_learning_ratio = 5.0
    n_iteration = 500
    print_every = 1
    save_every = 100
    save_dir = os.path.join("data", "save")
