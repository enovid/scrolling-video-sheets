import copy
from copy import deepcopy
import numpy as np
import cv2 as cv
VIDEO_PATH = 'sample_data/sample.avi'


def video_frames(path, gridsize=(3,3), outputsize=(1920, 1080), save=False):
    NROWS, NCOLS = gridsize
    NSTREAMS = NROWS * NCOLS + 1
    PARENT_WIDTH, PARENT_HEIGHT = outputsize
    TILE_WIDTH, TILE_HEIGHT = PARENT_WIDTH//NCOLS, PARENT_HEIGHT//NROWS

    # Initialize video streams
    streams = [cv.VideoCapture(path) for _ in range(NSTREAMS)]

    # Get video properties
    fps = int(streams[0].get(cv.CAP_PROP_FPS))
    frame_count = streams[0].get(cv.CAP_PROP_FRAME_COUNT)
    duration = frame_count//fps * 1000 # full duration of the video in ms
    ms_per_tile = duration // NSTREAMS
    # Resize video to fit in a grid of size `outputsize`
    retval, frame = streams[0].read()
    frame = cv.resize(frame, (TILE_WIDTH, TILE_HEIGHT), interpolation=cv.INTER_AREA)
    height, width = len(frame), len(frame[0])

    # Set stream offsets
    offset = 0
    for s in streams:
        s.set(cv.CAP_PROP_POS_MSEC, offset)
        offset += ms_per_tile

    ms_per_px = ms_per_tile // width
    ms_per_frame = 1000 // fps
    # How many pixels to scroll per frame so that we move `FRAME_WIDTH` pixels in ms_per_tile
    px_scroll_per_frame = ms_per_frame // ms_per_px

    if save:
        codec = cv.VideoWriter_fourcc('D', 'I', 'V', 'X')
        output_video = cv.VideoWriter(f'output{NROWS}x{NCOLS}.mp4', codec, fps, (PARENT_WIDTH, PARENT_HEIGHT))

    scroll_offset = 0 
    while scroll_offset < width: # The first tile starts off left of the screen
        # Read in the frames
        frames = []
        for s in streams: 
            retval, frame = s.read()
            if not retval:
                continue
            frame = cv.resize(frame, (TILE_WIDTH, TILE_HEIGHT), interpolation=cv.INTER_AREA)
            # Add time stamp to frame
            current_ms = s.get(cv.CAP_PROP_POS_MSEC)
            hours = int(current_ms // 3.6e+6)  
            mins = int((current_ms % 3.6e+6) // 60000)
            secs = int((current_ms % 60000) // 1000)
            cv.putText(frame, f'{hours:0>2}:{mins:0>2}:{secs:0>2}', (20, TILE_HEIGHT - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            frames.append(frame)
        full_frame = np.hstack(frames)

        # Construct row using sliding windows
        rows = []
        row = None
        for i in range(1, NSTREAMS+1): 
            start = int((width * i) - scroll_offset)
            end = int(start + width)
            window = full_frame[:,start:end]
            if row is None:
                row = window
            else:
                row = np.hstack((row, window))

            row_width = row.shape[1]
            if row_width > PARENT_WIDTH:
                rows.append(row[:,:PARENT_WIDTH])
                row = row[:,PARENT_WIDTH:]

        # Construct final frame
        if len(rows) == NROWS:
            final_frame = np.vstack(rows)
            cv.imshow('frame', final_frame)
            if save: output_video.write(final_frame)
        if cv.waitKey(ms_per_frame) & 0xFF == ord('q'):
            cv.destroyAllWindows()

        scroll_offset += px_scroll_per_frame

    for s in streams:
        s.release()

video_frames(VIDEO_PATH, gridsize=(3, 3), outputsize=(1920, 1080), save=True)
