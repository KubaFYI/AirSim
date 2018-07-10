import setup_path 
import airsim

# requires Python 3.5.3 :: Anaconda 4.4.0
# pip install opencv-python
import cv2
import time
import sys
import os
import threading
import csv
import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Freeform collect data stream from AirSim')
parser.add_argument('-o', '--output', default='D:\\AirSimCollectedData', help='Directory to store the collected information.')
parser.add_argument('-f', '--freq', default=30, help='Target frequency of data collection.')
# parser.add_argument('--no_position', action='store_false', help='Do not record position.')
# parser.add_argument('--no_orientation', action='store_false', help='Do not record orientation.')
parser.add_argument('--cameraPosition', default='0', help='Camera position to be used for recording (\'front_center\', \'front_right\', \'front_left\', \'bottom_center\' and \'back_center\')')
parser.add_argument('--imageType', default='collisiondist', help='Extra image type to be collected along the RGB-D (default -> CollisionDistance)')
parser.add_argument('--imageWidth', default=320, help='Width of the images to collect.')

cameraTypeMap = {
 'depth': airsim.ImageType.DepthVis,
 'segmentation': airsim.ImageType.Segmentation,
 'seg': airsim.ImageType.Segmentation,
 'scene': airsim.ImageType.Scene,
 'disparity': airsim.ImageType.DisparityNormalized,
 'normals': airsim.ImageType.SurfaceNormals,
 'collisiondist': airsim.ImageType.CollisionDistance
}

# csv_fieldnames = ['Timestamp', 'LV_x', 'LV_y', 'LV_z', 'AV_x', 'AV_y', 'AV_z']

# How often frames should the written into a file
frame_buffer_size = 100

def save_pfm_to_file(filename, pfm_data_array):
    np.save(filename, pfm_data_array)

def main(args):
    # Setup directories to store collected data
    dirname = time.strftime('%y-%m-%d_%H-%M-%S')
    data_dir = os.path.join(args.output, dirname)
    images_dir = os.path.join(data_dir, 'images')
    rec_filename = os.path.join(args.output, dirname, 'airsim_rec.csv')

    os.makedirs(images_dir)

    with open(rec_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # csvwriter = csv.writer(csvfile, fieldnames=csv_fieldnames)
        # csvwriter.writeheader()

        # Open connection to AirSim
        client = airsim.MultirotorClient()
        client.confirmConnection()

        img_response = client.simGetImages([
                airsim.ImageRequest(int(args.cameraPosition), cameraTypeMap['depth'], True)])
        pfm = airsim.get_pfm_array(img_response[0])
        pfm_data = np.empty((frame_buffer_size, 2, pfm.shape[0], pfm.shape[1]), dtype=float)

        # Main loop collecting data
        frames_processed = 0
        last_pfm_timestamp = None
        while True:
            loopstart = time.time()

            # Get timestamp and state estimation info
            airsim_state = client.getMultirotorState()
            timestamp = airsim_state.timestamp
            if last_pfm_timestamp is None:
                last_pfm_timestamp = timestamp

            gtk = client.simGetGroundTruthKinematics()
            csvwriter.writerow([timestamp,
                             gtk['linear_velocity']['x_val'],
                             gtk['linear_velocity']['y_val'],
                             gtk['linear_velocity']['z_val'],
                             gtk['angular_velocity']['x_val'],
                             gtk['angular_velocity']['y_val'],
                             gtk['angular_velocity']['z_val']])

            # Get the images
            img_responses = client.simGetImages([
                # RGB
                airsim.ImageRequest(int(args.cameraPosition), airsim.ImageType.Scene),
                # depth
                airsim.ImageRequest(int(args.cameraPosition), cameraTypeMap['depth'], True),
                # misc
                airsim.ImageRequest(int(args.cameraPosition), cameraTypeMap[args.imageType], True)])

            # Save the images
            png_filename = os.path.join(images_dir, 'rgb_{}.png'.format(timestamp))
            airsim.write_file(png_filename, img_responses[0].image_data_uint8)
            pfm_data[frames_processed%frame_buffer_size, 0, ...] = (airsim.get_pfm_array(img_responses[1]))
            pfm_data[frames_processed%frame_buffer_size, 1, ...] = (airsim.get_pfm_array(img_responses[2]))
            frames_processed += 1

            if frames_processed%frame_buffer_size == 0:
                # Dump the recorded frames into a file
                # Spin in a separate thread to not have a hiccup when recording
                pfm_data_filename = os.path.join(images_dir, 'pfm_data_{}.npy'.format(last_pfm_timestamp))
                thread_args = (pfm_data_filename, np.copy(pfm_data))
                threading.Thread(target=save_pfm_to_file, args=thread_args).start()
                last_pfm_timestamp = None
                # np.save(os.path.join(images_dir, 'pfm_data_{}.npy'.format(timestamp)), pfm_data)

            if frames_processed % 1000 == 0:
                print('Processed {} frames.'.format(frames_processed))

            if time.time() - loopstart > 1/args.freq:
                print('Running too slow to record at {}Hz ({}ms per frame))'.format(args.freq, time.time() - loopstart))
            while time.time() - loopstart < 1/args.freq:
                pass

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)