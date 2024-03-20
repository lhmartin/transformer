import numpy as np
import matplotlib.pyplot as plt

def visualise_pos_encoding(pos_embedings):
    """Credit to: https://github.com/gordicaleksa/pytorch-original-transformer/tree/main
    """
    embed_np = pos_embedings.numpy()

    shape = embed_np.shape
    data_type = embed_np.dtype

    width_mult = 9  # make it almost square
    positional_encodings_img = np.zeros((shape[0], width_mult*shape[1]), dtype=data_type)
    for i in range(width_mult):
        positional_encodings_img[:, i::width_mult] = embed_np

    plt.imshow(positional_encodings_img)