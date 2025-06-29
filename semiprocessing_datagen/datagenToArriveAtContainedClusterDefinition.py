# DS 29June25 - datagen.py modified to understand data and produce plots in order to arrive at a definition of contained clusters.
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

def check_2pix_at_boundary(matrix, threshold=1):
        # Define the boundary
        top_row = matrix[0, :]  # Top row
        bottom_row = matrix[-1, :]  # Bottom row
        left_column = matrix[:, 0]  # Left column
        right_column = matrix[:, -1]  # Right column

        # Initialize variables
        total_sum = 0
        count_above_threshold = 0
        has_adjacent_above_threshold = False

        # Check for two or more adjacent pixels above the threshold in each boundary
        for boundary in [top_row, bottom_row, left_column, right_column]:
                # Update total sum and count of elements above the threshold
                total_sum += np.sum(boundary)
                count_above_threshold += np.sum(np.abs(boundary) > threshold)
                # Check for adjacent elements above the threshold
                if np.any((np.abs(boundary[:-1]) > threshold) & (np.abs(boundary[1:]) > threshold)):
                        has_adjacent_above_threshold = True

        return has_adjacent_above_threshold, count_above_threshold, total_sum

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
        boundary_data_2pix = {
        "boundary_pixel_counts": [],  # List to store count_above_threshold values
        "total_sums": [],  # List to store total_sum values
        "alpha": [],
        "beta": [],
        "pT": [],
        "cluster_size_X": [],
        "cluster_size_Y": [],
        "cluster_size_X_normalAlpha": [],
        "cluster_size_Y_normalAlpha": [],
        "pT_normalAlpha": []
        }

        boundary_data_1pix = {
        "boundary_pixel_counts": [],  # List to store count_above_threshold values
        "total_sums": [],  # List to store total_sum values
        "alpha": [],
        "beta": [],
        "pT": [],
        "cluster_size_X": [],
        "cluster_size_Y": [],
        "cluster_size_X_normalAlpha": [],
        "cluster_size_Y_normalAlpha": [],
        "pT_normalAlpha": []
        }




        df = pd.DataFrame(arr_truth, columns = ['x-entry', 'y-entry','z-entry', 'n_x', 'n_y', 'n_z', 'number_eh_pairs', 'y-local', 'pt'])
        cols = df.columns
        for col in cols:
                df[col] = df[col].astype(float)

        df['cotAlpha'] = df['n_x']/df['n_z']
        df['cotBeta'] = df['n_y']/df['n_z']

        df['y-midplane'] = df['y-entry'] + df['cotBeta']*(sensor_thickness/2 - df['z-entry'])
        df['x-midplane'] = df['x-entry'] + df['cotAlpha']*(sensor_thickness/2 - df['z-entry'])
        df['original_atEdge'] = False
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

                positive_indices = np.argwhere(matrix.reshape(row_size, col_size) > boundary_charge_threshold)
                if positive_indices.size > 0:
                        bottom_left_index = positive_indices.min(axis=0)
                        top_right_index = positive_indices.max(axis=0)
                        cluster_size_X = top_right_index[1] - bottom_left_index[1] + 1
                        cluster_size_Y = top_right_index[0] - bottom_left_index[0] + 1

                has_adjacent_above_threshold_2pix, count_above_threshold_2pix, total_sum_2pix = check_2pix_at_boundary(matrix.reshape(row_size, col_size), boundary_charge_threshold)
                has_pixel_above_threshold_1pix, count_above_threshold_1pix, total_sum_1pix = check_1pix_at_boundary(matrix.reshape(row_size, col_size), boundary_charge_threshold)
                if( has_adjacent_above_threshold_2pix):
                        boundary_data_2pix["boundary_pixel_counts"].append(count_above_threshold_2pix)
                        boundary_data_2pix["total_sums"].append(total_sum_2pix)
                        boundary_data_2pix["alpha"].append(df.loc[i, 'cotAlpha'])
                        boundary_data_2pix["beta"].append(df.loc[i, 'cotBeta'])
                        boundary_data_2pix["pT"].append(df.loc[i, 'pt'])
                        boundary_data_2pix["cluster_size_X"].append(cluster_size_X)
                        boundary_data_2pix["cluster_size_Y"].append(cluster_size_Y)
                        if abs(df.loc[i, 'cotAlpha']) < 2.5:
                                boundary_data_2pix["cluster_size_X_normalAlpha"].append(cluster_size_X)
                                boundary_data_2pix["cluster_size_Y_normalAlpha"].append(cluster_size_Y)
                                boundary_data_2pix["pT_normalAlpha"].append(df.loc[i, 'pt'])  # Assuming pt is the same for normal alpha
                if( has_pixel_above_threshold_1pix):
                        boundary_data_1pix["boundary_pixel_counts"].append(count_above_threshold_1pix)
                        boundary_data_1pix["total_sums"].append(total_sum_1pix)
                        boundary_data_1pix["alpha"].append(df.loc[i, 'cotAlpha'])
                        boundary_data_1pix["beta"].append(df.loc[i, 'cotBeta'])
                        boundary_data_1pix["pT"].append(df.loc[i, 'pt'])
                        boundary_data_1pix["cluster_size_X"].append(cluster_size_X)
                        boundary_data_1pix["cluster_size_Y"].append(cluster_size_Y)
                        if abs(df.loc[i, 'cotAlpha']) < 2.5:
                                boundary_data_1pix["cluster_size_X_normalAlpha"].append(cluster_size_X)
                                boundary_data_1pix["cluster_size_Y_normalAlpha"].append(cluster_size_Y)
                                boundary_data_1pix["pT_normalAlpha"].append(df.loc[i, 'pt'])  # Assuming pt is the same for normal alpha

                df.loc[i, 'original_atEdge'] = has_adjacent_above_threshold_2pix
                # All time slices
                df3list.append(np.array(e).flatten())

                max_val = np.amax(e)

        df2 = pd.DataFrame(df2list)
        df3 = pd.DataFrame(df3list)

        x_bins = np.linspace(1, col_size-1, col_size-1)  # No. of boundary pixels (x-axis)
        y_bins = np.linspace(0, 1000, 501)  # Total sum (y-axis)
        alpha_bins = np.linspace(-10, 10, 101)  # Range from -5 to 5 with 50 bins
        beta_bins = np.linspace(-1, 1, 41)   # Range from -5 to 5 with 50 bins
        pt_bins = np.linspace(-5, 5, 101)    # Range from 0 to 100 with 50 bins


        hist_2pix, x_edges_2pix, y_edges_2pix = np.histogram2d(boundary_data_2pix["boundary_pixel_counts"], boundary_data_2pix["total_sums"], bins=[x_bins, y_bins])
        hist_1pix, x_edges_1pix, y_edges_1pix = np.histogram2d(boundary_data_1pix["boundary_pixel_counts"], boundary_data_1pix["total_sums"], bins=[x_bins, y_bins])

        x_bins_clust = np.linspace(1, col_size-1, col_size-1)  # No. of boundary pixels (x-axis)
        y_bins_clust = np.linspace(1, row_size-1, row_size-1)  # No. of boundary pixels (y-axis)
        clustHist_2pix, x_edges_clust_2pix, y_edges_clust_2pix = np.histogram2d(boundary_data_2pix["cluster_size_X"], boundary_data_2pix["cluster_size_Y"], bins=[x_bins_clust, y_bins_clust])
        clustHist_1pix, x_edges_clust_1pix, y_edges_clust_1pix = np.histogram2d(boundary_data_1pix["cluster_size_X"], boundary_data_1pix["cluster_size_Y"], bins=[x_bins_clust, y_bins_clust])
        clustHist_normalAlpha_2pix, x_edges_clust_2pix, y_edges_clust_2pix = np.histogram2d(boundary_data_2pix["cluster_size_X_normalAlpha"], boundary_data_2pix["cluster_size_Y_normalAlpha"], bins=[x_bins_clust, y_bins_clust])
        clustHist_normalAlpha_1pix, x_edges_clust_1pix, y_edges_clust_1pix = np.histogram2d(boundary_data_1pix["cluster_size_X_normalAlpha"], boundary_data_1pix["cluster_size_Y_normalAlpha"], bins=[x_bins_clust, y_bins_clust])


        alpha_hist_2pix, alpha_edges_2pix = np.histogram(boundary_data_2pix["alpha"], bins=alpha_bins)
        beta_hist_2pix, beta_edges_2pix = np.histogram(boundary_data_2pix["beta"], bins=beta_bins)
        pt_hist_2pix, pt_edges_2pix = np.histogram(boundary_data_2pix["pT"], bins=pt_bins)
        pT_normalAlpha_hist_2pix, pt_edges_2pix = np.histogram(boundary_data_2pix["pT_normalAlpha"], bins=pt_bins)
        alpha_hist_1pix, alpha_edges_1pix = np.histogram(boundary_data_1pix["alpha"], bins=alpha_bins)
        beta_hist_1pix, beta_edges_1pix = np.histogram(boundary_data_1pix["beta"], bins=beta_bins)
        pt_hist_1pix, pt_edges_1pix = np.histogram(boundary_data_1pix["pT"], bins=pt_bins)
        pT_normalAlpha_hist_1pix, pt_edges_1pix = np.histogram(boundary_data_1pix["pT_normalAlpha"], bins=pt_bins)

        # np.savez("unflipped/2Dhist_ContainedClusters_chargeThreshStudy_" + str(index) + ".npz", hist_2pix=hist_2pix, x_edges_2pix=x_edges_2pix, y_edges_2pix=y_edges_2pix)
        # Save all histograms for 2pix
        np.savez("unflipped/2Dhist_ContainedClusters_chargeThreshStudy_2pix_" + str(index) + ".npz", 
                hist_2pix=hist_2pix, x_edges_2pix=x_edges_2pix, y_edges_2pix=y_edges_2pix,
                alpha_hist_2pix=alpha_hist_2pix, alpha_edges_2pix=alpha_edges_2pix,
                beta_hist_2pix=beta_hist_2pix, beta_edges_2pix=beta_edges_2pix,
                pt_hist_2pix=pt_hist_2pix, pt_edges_2pix=pt_edges_2pix,
                clustHist_2pix=clustHist_2pix, x_edges_clust_2pix=x_edges_clust_2pix, y_edges_clust_2pix=y_edges_clust_2pix,
                clustHist_normalAlpha_2pix=clustHist_normalAlpha_2pix, pT_normalAlpha_hist_2pix=pT_normalAlpha_hist_2pix)



        # Save all histograms for 1pix
        np.savez("unflipped/2Dhist_ContainedClusters_chargeThreshStudy_1pix_" + str(index) + ".npz", 
                hist_1pix=hist_1pix, x_edges_1pix=x_edges_1pix, y_edges_1pix=y_edges_1pix,
                alpha_hist_1pix=alpha_hist_1pix, alpha_edges_1pix=alpha_edges_1pix,
                beta_hist_1pix=beta_hist_1pix, beta_edges_1pix=beta_edges_1pix,
                pt_hist_1pix=pt_hist_1pix, pt_edges_1pix=pt_edges_1pix,
                clustHist_1pix=clustHist_1pix, x_edges_clust_1pix=x_edges_clust_1pix, y_edges_clust_1pix=y_edges_clust_1pix,
                clustHist_normalAlpha_1pix=clustHist_normalAlpha_1pix, pT_normalAlpha_hist_1pix=pT_normalAlpha_hist_1pix)

        # split into flipped/unflipped, pos/neg charge
        split(sensor_thickness, index,df,df2,df3)

if __name__ == "__main__":
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
