import numpy as np
import cv2 as cv
import argparse, os
from pathlib import Path
import cProfile
import pstats
from pstats import SortKey
import time

def timestamp_ms2str(ms):
    hours = int(ms // 3.6e+6)  
    mins = int((ms % 3.6e+6) // 60000)
    secs = int((ms % 60000) // 1000)
    timestamp_str = f'{hours:0>2}:{mins:0>2}:{secs:0>2}'
    return timestamp_str

def resize_frame_cpu(frame, w, h):
    return cv.resize(frame, (w, h), interpolation=cv.INTER_AREA)

frame_device = cv.cuda_GpuMat()
def resize_frame_gpu(frame, w, h):
    frame_device.upload(frame)
    frame_device_resize = cv.cuda.resize(frame_device, (w, h))
    frame_host = frame_device_resize.download()
    return np.copy(frame_host)

def create_video_sheet(path, gridsize=(3, 3), outputsize=(1920, 1080), preview=False, save=False, savedir=None):
    NROWS, NCOLS = gridsize
    NSTREAMS = NROWS * NCOLS + 1
    PARENT_WIDTH, PARENT_HEIGHT = outputsize
    TILE_WIDTH, TILE_HEIGHT = PARENT_WIDTH//NCOLS, PARENT_HEIGHT//NROWS

    # Initialize video streams
    streams = [cv.VideoCapture(path) for _ in range(NSTREAMS)]

    # Get video properties
    fps = int(streams[0].get(cv.CAP_PROP_FPS))
    n_frames = streams[0].get(cv.CAP_PROP_FRAME_COUNT)
    duration = n_frames // fps * 1000 # full duration of the video in ms

    # Set stream offsets
    ms_per_tile = duration // NSTREAMS
    offset = 0
    for s in streams:
        s.set(cv.CAP_PROP_POS_MSEC, offset)
        offset += ms_per_tile

    # Calculate scroll speed
    ms_per_px = ms_per_tile // TILE_WIDTH
    ms_per_frame = 1000 // fps
    # How many pixels to scroll per frame so that we move `FRAME_WIDTH` pixels in `ms_per_tile`
    px_scroll_per_frame = ms_per_frame / ms_per_px

    if save:
        codec = cv.VideoWriter_fourcc('D', 'I', 'V', 'X')
        output_video = cv.VideoWriter(f'{savedir}/{Path(path).stem}_vish{NROWS}x{NCOLS}.mp4', codec, fps, (PARENT_WIDTH, PARENT_HEIGHT))

    scroll_offset = 0 
    while scroll_offset < TILE_WIDTH: # The first tile starts off left of the screen
        print(f'scroll_offset: {scroll_offset:.2f} px')
        print(f'TILE_WIDTH: {TILE_WIDTH}')
        print(f'{int(scroll_offset/TILE_WIDTH*100)}%')
        # Read in the frames
        frame_count = 0
        resize_start_time = time.time()
        frames = []
        for s in streams: 
            retval, frame = s.read()
            if not retval:
                continue
            frame_count += 1

            # Resize the frame to fit in a grid of given dimension, specified by `outputsize`
            # frame = resize_frame_cpu(frame=frame, w=TILE_WIDTH, h=TILE_HEIGHT)
            frame = resize_frame_gpu(frame=frame, w=TILE_WIDTH, h=TILE_HEIGHT)

            # Add time stamp to frame
            current_ms = s.get(cv.CAP_PROP_POS_MSEC)
            timestamp = timestamp_ms2str(ms=current_ms)
            cv.putText(frame, timestamp, (20, TILE_HEIGHT - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            frames.append(frame)
        resize_end_time = time.time()
        resize_time = (resize_end_time - resize_start_time)*1000/frame_count
        print(f'Resize: {frame_count} frames, {resize_time:.2f} ms/frame')

        # Concatenate the frames required for a single frame of the output video into one long frame
        full_frame = np.hstack(frames)

        # Construct grid rows using sliding windows
        rows = []
        row = None
        for i in range(1, NSTREAMS+1): 
            start = int((TILE_WIDTH * i) - scroll_offset)
            end = int(start + TILE_WIDTH)
            window = full_frame[:, start:end]

            # Append the current window to our row
            if row is None:
                row = window
            else:
                row = np.hstack((row, window))

            # Wrap the frame to a new row
            row_width = row.shape[1]
            if row_width > PARENT_WIDTH:
                rows.append(row[:,:PARENT_WIDTH])
                row = row[:,PARENT_WIDTH:]

        # Construct final frame
        final_frame = np.vstack(rows)
        if save: 
            output_video.write(final_frame)

        # Show live preview
        if preview: 
            cv.imshow('frame', final_frame)
            if cv.waitKey(ms_per_frame) & 0xFF == ord('q'):
                cv.destroyAllWindows()

        scroll_offset += px_scroll_per_frame

    for s in streams:
        s.release()

GRID_WIDTH  = 3
GRID_HEIGHT = 3
PREVIEW = True
SAVE_DIR = 'C:/Users/enovid/github/scrolling-video-sheets/output'
VIDEO_PATH = 'C:\Users\enovid\github\scrolling-video-sheets\sample_data\sample.avi'


if __name__ == '__main__':
    cProfile.run('create_video_sheet(path=VIDEO_PATH, gridsize=(GRID_HEIGHT, GRID_WIDTH), outputsize=(1920, 1080), preview=False, save=True, savedir=SAVE_DIR)', 'profilerstats')
    p = pstats.Stats('profilerstats')
    p.sort_stats(SortKey.TIME).print_stats(10)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('paths', metavar='filepath', type=str, nargs='+', help='path to input video')
    # args = parser.parse_args()
    # print(args)
    # print(args.paths)
    # for path in args.paths:
        # create_video_sheet(path, gridsize=(GRID_HEIGHT, GRID_WIDTH), outputsize=(1920, 1080), preview=PREVIEW, save=True)
    # print('Done.')
