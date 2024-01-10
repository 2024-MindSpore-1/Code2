import numpy as np


################# ACQUIRE NUMBER OF WEIGHTS #################
def gimme_params(model):
    return sum([p.size for p in model.trainable_params()])


################# SAVE TRAINING PARAMETERS IN NICE STRING #################
def gimme_save_string(opt):
    varx = vars(opt)
    base_str = ''
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key], dict):
            for sub_key, sub_item in varx[key].items():
                base_str += '\n\t' + str(sub_key) + ": " + str(sub_item)
        else:
            base_str += '\n\t' + str(varx[key])
        base_str += '\n\n'
    return base_str

