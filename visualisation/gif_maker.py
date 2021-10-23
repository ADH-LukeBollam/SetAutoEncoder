import numpy as np
import cv2 as cv
import tensorflow as tf
import base64
from PIL import Image

def encode_gif(frames, fps):
    from subprocess import Popen, PIPE
    h, w, c = frames[0].shape
    pxfmt = {1: 'gray', 3: 'rgb24'}[c]
    cmd = ' '.join([
        'ffmpeg -y -f rawvideo -vcodec rawvideo',
        f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
        '[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
        f'-r {fps:.02f} -f gif -'])
    proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.astype(np.uint8).tostring())
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
    del proc
    return out


def create_xy_gif(sets, set_size):
    img_res = 256

    frames = []
    for s in sets:
        set_x = s[:, 0]  # scale our points to pixel locations
        set_y = s[:, 1]

        frame = np.zeros((img_res, img_res, 1))

        for u in range(set_size):
            point_x = int(set_y[u] * img_res)
            point_y = int(set_x[u] * img_res)
            cv.circle(frame, (point_x, point_y), 6, (255.0,), -1, lineType=cv.LINE_AA)
        frames.append(frame)

    frames = np.stack(frames, axis=0)

    # image = tf.compat.v1.Summary.Image(height=img_res, width=img_res, colorspace=1)

    # _writer = tf.summary.create_file_writer('', max_queue=1000)
    # summary = tf.compat.v1.Summary()
    # image = tf.compat.v1.Summary.Image(height=img_res, width=img_res, colorspace=1)
    img = encode_gif(frames, 10)
    # summary.value.add(tag='image', image=image)
    # tf.summary.experimental.write_raw_pb(summary.SerializeToString(), 0)
    # _writer.flush()

    with open("interpolation.gif", "wb") as fh:
        fh.write(img)

    return frames

