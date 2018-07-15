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
import signal

parser = argparse.ArgumentParser(description='Freeform collect data stream from AirSim')
parser.add_argument('-o', '--output', default='D:\\AirSimCollectedData', help='Directory to store the collected information.')
parser.add_argument('-f', '--freq', default=15, help='Target frequency of data collection.')
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

# csv_fieldnames = ['Timestamp', 'LV_x', 'LV_y', 'LV_z', 'AV_x', 'AV_y', 'AV_z', 'OQ_x', 'OQ_y', 'OQ_z']

def save_images_to_files(images_dir, timestamp, image_data, depth_data, misc_data):
    rgb_filename = os.path.join(images_dir, 'rgb_'+str(timestamp)+'.png')
    depth_filename = os.path.join(images_dir, 'depth_'+str(timestamp))
    misc_filename = os.path.join(images_dir, 'misc_'+str(timestamp))

    airsim.write_file(rgb_filename, image_data)
    np.savez(depth_filename, depth_data)
    np.savez(misc_filename, misc_data)

def main(args):
    # Setup directories to store collected data
    dirname = time.strftime('%y-%m-%d_%H-%M-%S')
    data_dir = os.path.join(args.output, dirname)
    images_dir = os.path.join(data_dir, 'images')
    rec_filename = os.path.join(args.output, dirname, 'airsim_rec.csv')

    os.makedirs(images_dir)
    
    time_to_terminate = False

    def signal_handler(sig, frame):
        ''' Performs cleanup and makes sure all the data is saved properly '''
        nonlocal time_to_terminate
        time_to_terminate = True

    signal.signal(signal.SIGINT, signal_handler)
    print("Terminate recording with CTRL+C")

    with open(rec_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # csvwriter = csv.writer(csvfile, fieldnames=csv_fieldnames)
        # csvwriter.writeheader()

        # Open connection to AirSim
        client = airsim.MultirotorClient()
        client.confirmConnection()

        img_response = client.simGetImages([
                airsim.ImageRequest(int(args.cameraPosition), cameraTypeMap['depth'], True)])
        img_size = airsim.get_pfm_array(img_response[0]).shape

        # Main loop collecting data
        frames_processed = 0
        while True:
            loopstart = time.time()

            # Get timestamp and state estimation info
            airsim_state = client.getMultirotorState()
            timestamp = airsim_state.timestamp

            gtk = client.simGetGroundTruthKinematics()
            csvwriter.writerow([timestamp,
                             gtk['linear_velocity']['x_val'],
                             gtk['linear_velocity']['y_val'],
                             gtk['linear_velocity']['z_val'],
                             gtk['angular_velocity']['x_val'],
                             gtk['angular_velocity']['y_val'],
                             gtk['angular_velocity']['z_val'],
                             gtk['orientation']['w_val'],
                             gtk['orientation']['x_val'],
                             gtk['orientation']['y_val'],
                             gtk['orientation']['z_val']])

            # Get the images
            img_responses = client.simGetImages([
                # RGB
                airsim.ImageRequest(int(args.cameraPosition), airsim.ImageType.Scene),
                # depth
                airsim.ImageRequest(int(args.cameraPosition), cameraTypeMap['depth'], True),
                # misc
                airsim.ImageRequest(int(args.cameraPosition), cameraTypeMap[args.imageType], True)])

            # Save the data
            frames_processed += 1
            thread_args = (images_dir, timestamp,
                           img_responses[0].image_data_uint8,   # rgb image data
                           airsim.list_to_2d_float_array(img_responses[1].image_data_float,
                                                         img_size[1],
                                                         img_size[0]),   # depth float array
                           airsim.list_to_2d_float_array(img_responses[2].image_data_float,
                                                         img_size[1],
                                                         img_size[0]))   # misc info float array
            threading.Thread(target=save_images_to_files, args=thread_args).start()

            if frames_processed % 1000 == 0:
                print('Processed {} frames.'.format(frames_processed))

            if time.time() - loopstart > 1/args.freq:
                print('Running too slow to record at {}Hz ({}ms per frame))'.format(args.freq, time.time() - loopstart))
            
            while time.time() - loopstart < 1/args.freq:
                pass

            sys.stdout.write('\rProcessed {} frames\tRecording at {:.1f}Hz'.format(frames_processed, 1./(time.time() - loopstart)))

            if time_to_terminate:
                break

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)