import sys
import numpy as np
np.set_printoptions(threshold=np.inf)
import random
import pandas as pd
import math
import os
import matplotlib.pyplot as plt

# TO DO: generalize for different sensor geometries
#      - generalize offset values
#      - Change offset1/2 var names to offsetX/Y for better readability

def read_blocks_from_file(filename):
    with open(filename, 'r') as file:
        blocks = []
        block = []
        for line in file:
            if line.startswith('<time slice'):
                if block:
                    blocks.append(block)
                    block = []
            else:
                block.append([float(x) for x in line.split()])
        if block:
            blocks.append(block)
    print(blocks)
    return blocks

def apply_offset(block, offset, rows=13, cols=21):
    new_block = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if block[i][j] != 0:
                new_i = i + offset[0]
                new_j = j + offset[1]
                if 0 <= new_i < rows and 0 <= new_j < cols:
                    new_block[new_i][new_j] = block[i][j]
    
    return new_block

def check_boundary(matrix, threshold=1):
    """
    Checks if there are non-zero elements at the boundary of the given matrix.
    """
    # Check the boundary elements
    top_row = matrix[0, :]  # Top row
    bottom_row = matrix[-1, :]  # Bottom row
    left_column = matrix[:, 0]  # Left column
    right_column = matrix[:, -1]  # Right column

    # If any boundary element is non-zero, set the flag to True
    if (np.any(np.abs(top_row) > threshold) or np.any(np.abs(bottom_row) > threshold) or np.any(np.abs(left_column) > threshold) or np.any(np.abs(right_column) > threshold)    ):
        return True
    else:
        return False

def split(index,df1,df2,df2_uncentered,df3,df3_uncentered):
        print("Columns of labels file: ", df1.columns)
        df1.columns = df1.columns.astype(str)
        df2.columns = df2.columns.astype(str)
        df2_uncentered.columns = df2_uncentered.columns.astype(str)
        df3.columns = df3.columns.astype(str)
        df3_uncentered.columns = df3_uncentered.columns.astype(str)
        # unflipped, all charge              
        if("unflipped" not in os.listdir()):
            os.mkdir("unflipped")                                               
        df1[df1['z-entry']==100].to_parquet("unflipped/labels_d"+str(index)+".parquet")
        df2[df1['z-entry']==100].to_parquet("unflipped/recon2D_d"+str(index)+".parquet")
        df2_uncentered[df1['z-entry']==100].to_parquet("unflipped/recon2D_uncentered_d"+str(index)+".parquet")
        df3[df1['z-entry']==100].to_parquet("unflipped/recon3D_d"+str(index)+".parquet")
        df3_uncentered[df1['z-entry']==100].to_parquet("unflipped/recon3D_uncentered_d"+str(index)+".parquet")

def parseFile(filein,tag,nevents=-1,row_size=13,col_size=21):
        with open(filein) as f:
                lines = f.readlines()
        header = lines[0].strip()
        #header = lines.pop(0).strip()
        pixelstats = lines[1].strip()
        #pixelstats = lines.pop(0).strip()
        print("Header: ", header)
        print("Pixelstats: ", pixelstats)
        readyToGetTruth = False
        readyToGetTimeSlice = False
        clusterctr = 0
        cluster_truth =[]
        timeslice = 0
        cur_slice = []
        cur_cluster = []
        events = []
        for line in lines:
                ## Start of the cluster
                if "<cluster>" in line:
                        readyToGetTruth = True
                        readyToGetTimeSlice = False
                        clusterctr += 1
                        # Create an empty cluster
                        cur_cluster = []
                        timeslice = 0
                        # move to next line
                        continue
                # the line after cluster is the truth
                if readyToGetTruth:
                        cluster_truth.append(line.strip().split())
                        readyToGetTruth = False
                        # move to next line
                        continue
                ## Put cluster information into np array
                if "time slice" in line:
                        readyToGetTimeSlice = True
                        cur_slice = []
                        timeslice += 1
                        # move to next line
                        continue
                if readyToGetTimeSlice:
                        cur_row = line.strip().split()
                        cur_slice += [float(item) for item in cur_row]
                        # When you have all elements of the 2D image:
                        if len(cur_slice) == row_size*col_size:
                                cur_cluster.append(cur_slice)
                        # When you have all time slices:
                        if len(cur_cluster) == 20:
                                events.append(cur_cluster)
                                readyToGetTimeSlice = False
        print("Number of clusters = ", len(cluster_truth))
        print("Number of events = ",len(events))
        print("Number of time slices in cluster = ", len(events[0]))
        arr_truth = np.array(cluster_truth)
        arr_events = np.array( events )
        return arr_events, arr_truth

def main():
        sensor_pitch_X = 50 # in um
        sensor_pitch_Y = 12.5 # in um
        sensor_thickness = 100 #um      
        row_size, col_size = 16, 16
        # row_size, col_size = 13, 21
        maxXshift = 9
        maxYshift = 5
        boundary_charge_threshold = 1
        print("==========================")                   
        print(f'NOTE - sensor geometry is hard-coded as {sensor_pitch_X}x{sensor_pitch_Y}x{sensor_thickness} um3. \nAssuming pixel array size = {col_size} X {row_size}/')
        print("==========================")                   
        index = int(sys.argv[1])
        tag = "d"+str(index)
        inputdir = "./"
        arr_events, arr_truth = parseFile(filein=inputdir+"pixel_clusters_d"+str(index)+".out",tag=tag, row_size=row_size, col_size=col_size)
        #truth quantities - all are dumped to DF                        
        df = pd.DataFrame(arr_truth, columns = ['x-entry', 'y-entry','z-entry', 'n_x', 'n_y', 'n_z', 'number_eh_pairs', 'y-local', 'pt'])
        cols = df.columns
        for col in cols:
                df[col] = df[col].astype(float)

        df['offset1'] = np.random.randint(-maxYshift, maxYshift+1, size=len(df))
        df['offset2'] = np.random.randint(-maxXshift, maxXshift+1, size=len(df))
        df['original_atEdge'] = False
        df['uncentered_atEdge'] = False
        # # For shifts (-9, 9) along X and (-5, 5) along Y
        # df['offset1'] = np.random.randint(-5, 6, size=len(df))
        # df['offset2'] = np.random.randint(-9, 10, size=len(df))
        # # For shifts (-7, 7) along X and (-3, 3) along Y
        # df['offset1'] = np.random.randint(-3, 4, size=len(df))
        # df['offset2'] = np.random.randint(-7, 8, size=len(df))
        df['y-entry'] = df['y-entry'] + df['offset1']*sensor_pitch_Y
        df['x-entry'] = df['x-entry'] + df['offset2']*sensor_pitch_X
        df['cotAlpha'] = df['n_x']/df['n_z']
        df['cotBeta'] = df['n_y']/df['n_z']
        df['y-midplane'] = df['y-entry'] + df['cotBeta']*(sensor_thickness/2 - df['z-entry'])
        df['x-midplane'] = df['x-entry'] + df['cotAlpha']*(sensor_thickness/2 - df['z-entry'])
        print("The shape of the event array: ", arr_events.shape)
        print("The ndim of the event array: ", arr_events.ndim)
        print("The dtype of the event array: ", arr_events.dtype)
        print("The size of the event array: ", arr_events.size)
#        print("The max value in the array is: ", np.amax(arr_events))
        # print("The shape of the truth array: ", arr_truth.shape)
        df2 = {}
        df2list = []
        df2_uncentered = {}
        df2list_uncentered = []
        df3 = {}
        df3list = []
        df3_uncentered = {}
        df3list_uncentered = []
        offset_values = []
        for i, e in enumerate(arr_events):
                # Only last time slice
                df2list.append(np.array(e[-1]).flatten())
                matrix = np.array(e[-1])
                assert matrix.shape == (row_size, col_size)
                df.loc[i, 'original_atEdge'] = check_boundary(matrix, boundary_charge_threshold)
                df3list.append(np.array(e).flatten())
                # print("old block:")
                # for idx, block in enumerate(e.reshape(20, row_size, col_size)):
                #     print(f'<time slice {idx} ps>')
                #     for row in block:
                #         print(' '.join(map(str, row)))
                # All time slices
                random_integer = df['offset1'].iloc[i]
                random_integer2 = df['offset2'].iloc[i]
                offset = (random_integer, random_integer2) #(4, 5) #4 is up/down, 5 is left/right
                offset_values.append(offset)
                new_blocks = [apply_offset(block, offset, row_size, col_size) for block in e.reshape(20, row_size, col_size)]
                # print("\nNew block:")
                # for idx, new_block in enumerate(new_blocks):
                #     print(f'<time slice {idx} ps>')
                #     for row in new_block:
                #         print(' '.join(map(str, row)))
                df2list_uncentered.append(np.array(new_blocks[-1]).flatten())
                matrix_uncentered = np.array(new_blocks[-1])
                assert matrix_uncentered.shape == (row_size, col_size)
                df.loc[i, 'uncentered_atEdge'] = check_boundary(matrix_uncentered, boundary_charge_threshold)
                df3list_uncentered.append(np.array(new_blocks).flatten())
        df2 = pd.DataFrame(df2list)
        df2_uncentered = pd.DataFrame(df2list_uncentered)
        df3 = pd.DataFrame(df3list)
        df3_uncentered = pd.DataFrame(df3list_uncentered)  

        # split into flipped/unflipped, pos/neg charge
        split(index,df,df2,df2_uncentered,df3,df3_uncentered)
        offsets_x = [offset[0] for offset in offset_values]
        offsets_y = [offset[1] for offset in offset_values]

        plt.hist2d(offsets_x, offsets_y, bins=[20, 20], cmap='Blues')
        plt.colorbar(label='Count')
        plt.xlabel('Offset X')
        plt.ylabel('Offset Y')
        plt.title('2D Histogram of Offset Values')
        plt.savefig("unflipped/offset_histogram_d"+str(index)+".png")
        # # Read block from temp.out file
        # blocks = read_blocks_from_file('temp2.out')
        # # Define the offset
        # offset = (4, 5)
        # # Apply the offset
        # new_blocks = [apply_offset(block, offset) for block in blocks]
        # print("\nNew block:")
        # # Print the new blocks
        # for idx, new_block in enumerate(new_blocks):
        #     print(f'<time slice {idx} ps>')
        #     for row in new_block:
        #         print(' '.join(map(str, row)))

if __name__ == "__main__":
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
