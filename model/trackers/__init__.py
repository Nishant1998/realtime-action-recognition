from model.trackers.bytetrack.byte_tracker import BYTETracker


def get_tracker(track_thresh=0.6,match_thresh=30,track_buffer=0.8,frame_rate=30):
    tracker = BYTETracker(
        track_thresh=track_thresh, match_thresh=match_thresh,
        track_buffer=track_buffer, frame_rate=frame_rate
    )
    return tracker
