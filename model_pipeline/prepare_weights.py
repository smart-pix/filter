from fxpmath import Fxp
import os
import csv
import pandas as pd
import numpy as np


def bits_to_fxp_value(bits):
    bin_str = ''.join(str(b) for b in bits)
    raw = int(bin_str, 2)
    if bits[0] == 1:
        raw -= 16
    return raw * 0.125

# convert code from hsl4ml style output to chip style input
def prepareWeights(path):
    data_fxp = Fxp(None, signed=True, n_word=4, n_int=0)
    data_fxp.rounding = 'around'
    def to_fxp(val):
        return data_fxp(val)

    b5_data = pd.read_csv(os.path.join(path, 'b5.txt'), header=None)
    w5_data = pd.read_csv(os.path.join(path, 'w5.txt'), header=None)
    b2_data = pd.read_csv(os.path.join(path, 'b2.txt'), header=None)
    w2_data = pd.read_csv(os.path.join(path, 'w2.txt'), header=None)
    # print(b5_data)

    b5_data_list = []
    w5_data_list = []
    b2_data_list = []
    w2_data_list = []

    for i in range(2, -1, -1):
        b5_data_list.append(to_fxp(b5_data.values[0][i]).bin())

    for i in range(173, -1, -1):
        w5_data_list.append(to_fxp(w5_data.values[0][i]).bin())

    for i in range(57, -1, -1):
        b2_data_list.append(to_fxp(b2_data.values[0][i]).bin())

    for i in range(927, -1, -1):
        w2_data_list.append(to_fxp(w2_data.values[0][i]).bin())

    b5_bin_list = [int(bin_string) for data in b5_data_list for bin_string in data]
    w5_bin_list = [int(bin_string) for data in w5_data_list for bin_string in data]
    b2_bin_list = [int(bin_string) for data in b2_data_list for bin_string in data]
    w2_bin_list = [int(bin_string) for data in w2_data_list for bin_string in data]
    pixel_list = [0 for _ in range(512)]
    b5_w5_b2_w2_pixel_list = b5_bin_list + w5_bin_list + b2_bin_list + w2_bin_list + pixel_list

    csv_file = os.path.join(path, 'b5_w5_b2_w2_pixel_bin.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(b5_w5_b2_w2_pixel_list)
    
    return csv_file

def invert_weights(path):
    n_bits = 4
    lengths = {
        'b5': 3,
        'w5': 174,
        'b2': 58,
        'w2': 928,
        'pixel': 512
    }
    
    input_file = os.path.join(path, 'b5_w5_b2_w2_pixel_bin.csv')
    n_bits = 4
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        row = next(reader)
        bin_list = [int(x) for x in row]
    indices = np.cumsum([lengths[x]*n_bits for x in ['b5','w5','b2','w2','pixel']])
    b5_bits = bin_list[0:indices[0]]
    w5_bits = bin_list[indices[0]:indices[1]]
    b2_bits = bin_list[indices[1]:indices[2]]
    w2_bits = bin_list[indices[2]:indices[3]]
    def decode_block(block, n):
        return [bits_to_fxp_value(block[i*4:(i+1)*4]) for i in range(n)]
    b5_data = decode_block(b5_bits, lengths['b5'])
    w5_data = decode_block(w5_bits, lengths['w5'])
    b2_data = decode_block(b2_bits, lengths['b2'])
    w2_data = decode_block(w2_bits, lengths['w2'])
    pd.DataFrame([b5_data]).to_csv(f'{path}/b5_reco.txt', header=False, index=False)
    pd.DataFrame([w5_data]).to_csv(f'{path}/w5_reco.txt', header=False, index=False)
    pd.DataFrame([b2_data]).to_csv(f'{path}/b2_reco.txt', header=False, index=False)
    pd.DataFrame([w2_data]).to_csv(f'{path}/w2_reco.txt', header=False, index=False)


for i in range(1):
    # model=f'tmp_NoiseXe-/noiseTrainedModels/model0/qmodel_0_catapult_prj'
    task = 1 # 1: prepare weights, 2: invert weights
    if task == 1:
        # model=f'tmp_16x16_centeredIncidence/trainedModels/model4/qmodel_0_catapult_prj'
        model=f'tmp_16x16_noiseRetrained/noiseTrainedModels/model1/qmodel_0_catapult_prj'
        dnn_csv = prepareWeights(f'{model}/firmware/weights/')
    elif task == 2:
        model=f'/asic/projects/C/CMS_PIX_28/dshekar/CMSPIX28_DAQnewboardtestroomcarboard/spacely/PySpacely/spacely-asic-config/CMSPIX28Spacely/csv_new/'
        invert_weights(f'{model}')
