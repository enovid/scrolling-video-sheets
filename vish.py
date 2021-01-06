import numpy as np
import cv2 as cv
import argparse, os
from pathlib import Path
import cProfile
import pstats
from pstats import SortKey
import time
from alive_progress import alive_bar
from multiprocessing import Pool, Process, Queue
from functools import partial

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

    def proc_next_frame(stream):
        retval, frame = s.read()
        if not retval:
            return

        # Resize the frame to fit in a grid of given dimension, specified by `outputsize`
        frame = resize_frame_cpu(frame=frame, w=TILE_WIDTH, h=TILE_HEIGHT)

        # Add time stamp to frame
        current_ms = s.get(cv.CAP_PROP_POS_MSEC)
        timestamp = timestamp_ms2str(ms=current_ms)
        cv.putText(frame, timestamp, (20, TILE_HEIGHT - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        frames.append(frame)
        return frame


    pool = Pool()
    taskqueue = Queue()

    frames = []
    scroll_offset = 0 
    with alive_bar(manual=True) as bar:
        while scroll_offset < TILE_WIDTH: # The top left tile starts off left of the screen
            # Progress bar
            completion_pct = '%.2f'%(scroll_offset/TILE_WIDTH)
            bar(completion_pct)

            # Read in the frames
            frames = []
            pool.map(proc_next_frame, streams)
            # for s in streams: 
                # frame = proc_next_frame(s)
                # frames.append(frame)

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

GRID_WIDTH  = 10
GRID_HEIGHT = 10
PREVIEW = True
SAVE_DIR = '/home/enovid/repos/scrolling-video-sheets/output'
VIDEO_PATH = '/home/enovid/repos/scrolling-video-sheets/sample_data/BigBuckBunny1080p.mp4'


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
