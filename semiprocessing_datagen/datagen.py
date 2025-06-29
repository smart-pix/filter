# DS 29June25 - datagen.py modified to include new labels after we've arrived at a definition of contained clusters.
import sys
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def split(sensor_thickness,index,df1,df2,df3):

        df1.columns = df1.columns.astype(str)
        df2.columns = df2.columns.astype(str)
        df3.columns = df3.columns.astype(str)

        # unflipped, all charge                                                                         
        df1[df1['z-entry']==sensor_thickness].to_parquet("unflipped/labels_d"+str(index)+".parquet")
        df2[df1['z-entry']==sensor_thickness].to_parquet("unflipped/recon2D_d"+str(index)+".parquet")
        df3[df1['z-entry']==sensor_thickness].to_parquet("unflipped/recon3D_d"+str(index)+".parquet")

def check_1pix_at_boundary(matrix, threshold=1):
        # Define the boundary
        boundaries = np.concatenate([matrix[0, :], matrix[-1, :], matrix[:, 0], matrix[:, -1]])
        # Check for at least one pixel above the threshold
        has_pixel_above_threshold = np.any(np.abs(boundaries) > threshold)
        # Count and sum pixels above the threshold
        count_above_threshold = np.sum(np.abs(boundaries) > threshold)
        total_sum = np.sum(boundaries)

        return has_pixel_above_threshold, count_above_threshold, total_sum

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
        row_size, col_size = 16, 16
        sensor_thickness = 100 #um         
        boundary_charge_threshold = 1 # threshold for charge at the boundary to be considered as a boundary charge
        print("==========================")                   
        print(f'NOTE - sensor thickness is hard-coded as {sensor_thickness} um. \nAssuming pixel array size = {col_size} X {row_size}/')
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

        df['cotAlpha'] = df['n_x']/df['n_z']
        df['cotBeta'] = df['n_y']/df['n_z']

        df['y-midplane'] = df['y-entry'] + df['cotBeta']*(sensor_thickness/2 - df['z-entry'])
        df['x-midplane'] = df['x-entry'] + df['cotAlpha']*(sensor_thickness/2 - df['z-entry'])
        df['original_atEdge'] = False
        df['charge_atEdge'] = 0
        df['nPix_atEdgeAbove1e'] = 0
        print("The shape of the event array: ", arr_events.shape)
        print("The ndim of the event array: ", arr_events.ndim)
        print("The dtype of the event array: ", arr_events.dtype)
        print("The size of the event array: ", arr_events.size)
#        print("The max value in the array is: ", np.amax(arr_events))
        # print("The shape of the truth array: ", arr_truth.shape)

        df2 = {}
        df2list = []

        df3 = {}
        df3list = []

        for i, e in enumerate(arr_events):
                
                # Only last time slice
                df2list.append(np.array(e[-1]).flatten())
                matrix = np.array(e[-1])
                assert matrix.shape[0] == row_size * col_size

                has_pixel_above_threshold_1pix, count_above_threshold_1pix, total_sum_1pix = check_1pix_at_boundary(matrix.reshape(row_size, col_size), boundary_charge_threshold)
                if has_pixel_above_threshold_1pix:
                        df.loc[i, 'original_atEdge'] = True
                        # Following columns are added to the df if has_pixel_above_threshold_1pix is True. Could add them if has_pixel_above_threshold_1pix is False, but would be too small of a charge at the edge to treat as relevant.
                        df.loc[i, 'charge_atEdge'] = total_sum_1pix
                        df.loc[i, 'nPix_atEdgeAbove1e'] = count_above_threshold_1pix
                # All time slices
                df3list.append(np.array(e).flatten())

        df2 = pd.DataFrame(df2list)
        df3 = pd.DataFrame(df3list)

        # split into flipped/unflipped, pos/neg charge
        split(sensor_thickness, index,df,df2,df3)

if __name__ == "__main__":
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
