try:
    import os
    import sys
    import glob
    import pandas as pd
    import numpy as np
    import hls4ml
    from fxpmath import Fxp
    import csv
    from tqdm import tqdm

    # custom imports
    sys.path.insert(1, '../pretrain-data-prep/')
    # sys.path.append(os.path.join(os.path.dirname(__file__), 'pretrain-data-prep')) # https://github.com/smart-pix/pretrain-data-prep/tree/main
    from dataset_utils import quantize_manual, add_noise

except:
    print("Import error:", f"{__file__}: {str(e)}")
    sys.exit(1)  # Exit script immediately
    
# load example inputs and outputs
def loadExampleTestVectors():
    x_test = pd.read_csv("/asic/projects/C/CMS_PIX_28/benjamin/verilog/workarea/cms28_smartpix_verification/PnR_cms28_smartpix_verification_D/tb/dnn/csv/l6/input_1.csv", header=None)
    x_test = np.array(x_test.values.tolist())
    y_test = pd.read_csv("/asic/projects/C/CMS_PIX_28/benjamin/verilog/workarea/cms28_smartpix_verification/PnR_cms28_smartpix_verification_D/tb/dnn/csv/l6/layer7_out_ref_int.csv", header=None)
    y_test = np.array(y_test.values.tolist()).flatten()
    return x_test, y_test

def signalAboveThreshold(X, noise_threshold):
    X = np.where(X < noise_threshold, 0, X)
    return X

# Vectorized sumRow function for all rows
def sumRow_vectorized(X, size_x):
    # X = np.where(X < noise_threshold, 0, X) # moved to separate function called earlier
    # X shape: (num_samples, 273)
    X_reshaped = X.reshape(-1, 16, size_x) # Changed from (13, X size) to (16, X size) after padding (14Sep25)
    X_reshaped = np.sum(X_reshaped, axis=2)
    return X_reshaped  # shape: (num_samples, 16)

# load data from parquet files
def loadParquetData(
    inFilePath="data/", # in path where the labels, recon2D files sit
    noise_threshold = 0, # thresholds on the number of electrons
    zero_pad = True, # zero pad the 13x21 to 16x21
    threshold = 0.2, # pt threshold in GeV for high pT particle
    qm_charge_levels = [400, 1600, 2400], # quantize manual charge levels
    qm_quant_values = [0, 1, 2, 3], # quantize manual quant values
    outDir = None, # output directory for the csv files
    useYProfilePulsing=True, # Create pulsing pattern using Yprofile. If False, use the recon2D data directly
):
    if zero_pad == True:
        size_y, size_x = 13, 21
    else:
        size_y, size_x = 16, 16
    y_local_bins = np.linspace(-8.1, 8.1, 13)
    bin_number = 6
    y_local_min, y_local_max = y_local_bins[bin_number], y_local_bins[bin_number + 1]
    # load the labels and data
    inFilePaths = list(sorted(glob.glob(os.path.join(inFilePath, "labels*")))) 
    clslabels, pts, ylocals, trainrecons = [], [], [], []
    for inFile in tqdm(inFilePaths):

        # load the labels
        label = pd.read_parquet(inFile)
        # filter on y-local bin
        filtered_indices = label[(label['y-local'] >= y_local_min) & (label['y-local'] < y_local_max)].index
        label = label.loc[filtered_indices].reset_index(drop=True)
        
        pt = label['pt'].values
        ylocal = label['y-local'].values
        clslabel = np.full_like(pt, fill_value=-999, dtype=int)
        clslabel[np.abs(pt) > threshold] = 0
        clslabel[(pt < 0) & (pt >= -1 * threshold)] = 1 # -1*threshold<=row2['pt']<0
        clslabel[(pt >= 0) & (pt <= threshold)] = 2
        # save
        clslabels.append(clslabel)
        pts.append(pt)
        ylocals.append(ylocal)

        # Load the data
        temp = pd.read_parquet(inFile.replace("labels", "recon2D"))
        # Filter recon2D data using the same indices
        temp = temp.loc[filtered_indices].reset_index(drop=True)
        temp_array = temp.to_numpy().reshape(-1, size_y, size_x)  # Reshape
        # Zero-pad the data to form 16 x X_size
        if zero_pad:
            padded_array = np.pad(temp_array, ((0, 0), (0, 3), (0, 0)), mode='constant', constant_values=0)  
            temp_array = padded_array
        # # Add noise at last timestamp
        # if noise_sigma > 0:
        #     for j in range(1):  # Since the filtering algorithm uses only the last timestamp, add noise there
        #         noisy_array = add_noise(padded_array, mu=0, sig=noise_sigma, shuffled=False)  # Apply noise to the entire array
        #     padded_array = noisy_array
        temp_array = temp_array.reshape(temp_array.shape[0], -1)
        temp = pd.DataFrame(temp_array, columns=[f"{i}" for i in range(temp_array.shape[1])])
        # Quantize data
        temp = quantize_manual(temp, charge_levels=qm_charge_levels, quant_values=qm_quant_values, shuffled=True)
        trainrecons.append(temp)

    # concatenate the labels
    clslabels = np.concatenate(clslabels, axis=0)
    pts = np.concatenate(pts, axis=0)
    ylocals = np.concatenate(ylocals, axis=0)
    print(clslabels.shape, pts.shape, ylocals.shape)

    # convert to csv
    trainrecons_csv = pd.concat(trainrecons, ignore_index=True)
    print(len(trainrecons_csv))

    print("Creating yprofiles")
    
    # Convert DataFrames to numpy arrays for vectorized operations
    trainrecons_np = trainrecons_csv.values  # shape: (num_samples, 273) -> (num_samples, 16*21) after DS introduced noise-injection and padding (14Sep25)
    trainrecons_np = signalAboveThreshold(trainrecons_np, noise_threshold)
    # Compute yprofiles in a vectorized way
    yprofiles = sumRow_vectorized(trainrecons_np, size_x)
    # convert to numpy
    yprofiles = np.array(yprofiles)
    # pad the yprofiles to get to 16 dimension
    # yprofiles = np.pad(yprofiles, ((0, 0), (0, 3)), mode='constant', constant_values=0) # Padding moved to earlier part of code, DS (14Sep25)

    # save 
    if outDir is not None:
        outDir = os.path.join(outDir, "_".join(map(str, qm_charge_levels)))
        os.makedirs(outDir, exist_ok=True)
        # save with np.save and include the quantization parameters in the name
        np.save(os.path.join(outDir, 'yprofiles.npy'), yprofiles)
        np.save(os.path.join(outDir, 'ylocals.npy'), ylocals)
        np.save(os.path.join(outDir, 'clslabels.npy'), clslabels)
        np.save(os.path.join(outDir, 'pts.npy'), pts)
        if not useYProfilePulsing:
            if size_x != 16:
                print("Dataset passed should be 16x16. Reverting back to using yprofile pulsing.")
                useYProfilePulsing = True
            else:
                trainrecons_save = trainrecons_np.reshape(-1, 16, size_x)
                np.save(os.path.join(outDir, 'recon2D.npy'), trainrecons_save)

    # output dictionary
    outDict = {
        "yprofiles": yprofiles,
        "ylocals": ylocals,
        "clslabels": clslabels,
        "pts": pts,
        "outDir": outDir,
        "recon2D": trainrecons_save if not useYProfilePulsing else None,
    }
    return outDict


# convert yprofiles to the pixel programming for asic
def yprofileToCompoutWrite(yprofiles, csv_file_name, flip=True):
    # create compout of y-local subset
    print("Making compout of y-local subset")
    print("   writing to file:", csv_file_name,)
    filtered_pixelout = input_to_pixelout(yprofiles, flip)
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(filtered_pixelout)
    print("   done!")

# convert yprofiles to the pixel programming for asic
def recon2DToCompoutWrite(recon2D, csv_file_name, flip=True):
    # create compout of y-local subset
    print("Making compout of y-local subset")
    print("   writing to file:", csv_file_name,)
    filtered_pixelout = reconInput_to_pixelout(recon2D)
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(filtered_pixelout)
    print("   done!")

# get the y-local bin from the yprofiles, clslabels and ylocals
# Giuseppe seems to have chosen the 0th bin (corresponding to bin_number=6) to produce the test-vectors from dataset 8 (https://github.com/GiuseppeDiGuglielmo/directional-pixel-detectors/blob/asic-flow/multiclassifier/train.ipynb)
def getYLocalBin(yprofiles, ylocals, clslabels, outDir="./", bins = np.linspace(-8.1, 8.1, 13), bin_number=6, flip=True):
    
    # pick up the ylocal min, max and interested range
    ylocal_min = bins[bin_number]
    ylocal_max = bins[bin_number + 1]
    mask = (ylocals >= ylocal_min) & (ylocals < ylocal_max)
    # pick up just those values
    filtered_yprofiles = yprofiles[mask]
    filtered_clslabels = clslabels[mask]
    filtered_ylocals = ylocals[mask]
    print(f"Number of samples chosen after filter: {mask.sum()}/{mask.shape[0]} ({mask.sum()/mask.shape[0]*100:.2f}%)")

    # create output dictionary
    outDict = {
        "yprofiles": filtered_yprofiles,
        "clslabels": filtered_clslabels,
        "ylocals": filtered_ylocals,
    }

    # create compout of y-local subset
    if outDir is not None:
        compout_file_name = os.path.join(outDir, f'compouts_ylocal_{ylocal_min:.2f}_{ylocal_max:.2f}.csv')
        yprofileToCompoutWrite(filtered_yprofiles, compout_file_name, flip)
        outDict["compout_file_name"] = compout_file_name

    return outDict

# Generate a simple configuration from keras model
def gen_hls_model(model, output_dir="newModelWeights"):
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    # Convert to an hls model
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir)
    hls_model.write()

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

# convert yprofiles to the pixel programming for asic
def input_to_pixelout(x, flip):
    N_INFERENCES = x.shape[0]

    # first create compout
    encoder_values_Ninferences = []
    for i in range(N_INFERENCES):
        encoder_values = []
        for j in range(16):
            a = x[i][j]
            encoder_sum = []
            if(a==0):
                encoder_sum = [0 for _ in range(16)]
                if flip:
                    encoder_sum.reverse()
                
            elif(a==1):
                encoder_sum = [1]+[0 for _ in range(15)]
                if flip:
                    encoder_sum.reverse()

            elif(a==2):
                encoder_sum = [2]+[0 for _ in range(15)]
                if flip:
                    encoder_sum.reverse()

            elif(a==3):
                encoder_sum = [3]+[0 for _ in range(15)]
                if flip:
                    encoder_sum.reverse()

            else:
                l3=[]
                if a%3 == 0:
                    result = int(a/3)
                    l3 = [3 for _ in range(result)]
                    l_diff = 16-len(l3)
                    encoder_sum = l3 + [0 for _ in range(l_diff)]
                    if flip:
                        encoder_sum.reverse()
                else:
                    result = int(a//3)
                    l3 = [3 for _ in range(result)]
                    diff = a-sum(l3)
                    l3.append(diff)
                    l_diff = 16-len(l3)
                    encoder_sum = l3 + [0 for _ in range(l_diff)]
                    if flip:
                        encoder_sum.reverse()

            encoder_values.append(encoder_sum)
        encoder_values = [j for i in encoder_values for j in i]
        encoder_values_Ninferences.append(encoder_values)
        
    compout_values_Ninferences = []
    for i in range(N_INFERENCES):
        compout_values = []
        for j in range(256):
            a = encoder_values_Ninferences[i][j]
            if(a==3):
                compout_values.append(7)
            elif(a==2):
                compout_values.append(3)
            elif(a==1):
                compout_values.append(1)
            else:
                compout_values.append(0)
        compout_values_Ninferences.append(compout_values)

    return compout_values_Ninferences

def reconInput_to_pixelout(x):
    N_INFERENCES = x.shape[0]
    assert x[0].shape == (16,16), "Input recon2D data must have shape (16x16)"
    # first create compout
    encoder_values_Ninferences = []
    for i in range(N_INFERENCES):
        encoder_values = x[i]
        encoder_values = [j for i in encoder_values for j in i]
        encoder_values_Ninferences.append(encoder_values)
        
    compout_values_Ninferences = []
    for i in range(N_INFERENCES):
        compout_values = []
        for j in range(256):
            a = encoder_values_Ninferences[i][j]
            if(a==3):
                compout_values.append(7)
            elif(a==2):
                compout_values.append(3)
            elif(a==1):
                compout_values.append(1)
            else:
                compout_values.append(0)
        compout_values_Ninferences.append(compout_values)

    return compout_values_Ninferences