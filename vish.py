import copy
from copy import deepcopy
import numpy as np
import cv2 as cv
VIDEO_PATH = 'sample_data/sample.avi'


def video_frames(path, save=False):
    NROWS, NCOLS = 3, 3
    NSTREAMS = NROWS * NCOLS + 1
    PARENT_WIDTH, PARENT_HEIGHT = 1920, 1080
    FW, FH = PARENT_WIDTH//NCOLS, PARENT_HEIGHT//NROWS

    # Initialize video streams
    streams = [cv.VideoCapture(path) for _ in range(NSTREAMS)]

    # Get video properties
    fps = int(streams[0].get(cv.CAP_PROP_FPS))
    frame_count = streams[0].get(cv.CAP_PROP_FRAME_COUNT)
    duration = frame_count//fps * 1000 # full duration of the video in ms
    ms_per_tile = duration // NSTREAMS
    retval, frame = streams[0].read()
    frame = cv.resize(frame, (FW, FH), interpolation=cv.INTER_AREA)
    height, width = len(frame), len(frame[0])
    print(f'height, width: {height, width}')

    # Set stream properties
    offset = 0
    for s in streams:
        print(offset)
        s.set(cv.CAP_PROP_POS_MSEC, offset)
        offset += ms_per_tile

    # Calculate time offset
    # how many pixels to scroll per frame so that we've moved 480 pixels in exactly ms_per_tile
    frames_per_tile = ms_per_tile//1000 * fps
    ms_per_px = ms_per_tile//width
    ms_per_frame = 1000 // fps
    px_scroll_per_frame = ms_per_frame // ms_per_px

    if save:
        codec = cv.VideoWriter_fourcc('D', 'I', 'V', 'X')
        output_video = cv.VideoWriter(f'output{NROWS}x{NCOLS}.mp4', codec, fps, (PARENT_WIDTH, PARENT_HEIGHT))
    scroll_offset = 0 # first tile is off left of the screen
    while scroll_offset < width:
        # Read in the frames
        frames = []
        for s in streams: 
            retval, frame = s.read()
            if not retval:
                continue
            frame = cv.resize(frame, (FW, FH), interpolation=cv.INTER_AREA)
            current_ms = s.get(cv.CAP_PROP_POS_MSEC)
            hours = int(current_ms // 3.6e+6)  
            mins = int((current_ms % 3.6e+6) // 60000)
            secs = int((current_ms % 60000) // 1000)
            cv.putText(frame, f'{hours:0>2}:{mins:0>2}:{secs:0>2}', (20, FH - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            frames.append(frame)
        full_frame = np.hstack(frames)

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

        if len(rows) == NROWS:
            final_frame = np.vstack(rows)
            cv.imshow('frame', final_frame)
            if save: output_video.write(final_frame)
        if cv.waitKey(ms_per_frame) & 0xFF == ord('q'):
            cv.destroyAllWindows()

        scroll_offset += px_scroll_per_frame

    for s in streams:
        s.release()

video_frames(VIDEO_PATH, save=True)
