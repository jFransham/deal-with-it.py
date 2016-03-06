import cv2
import math
from itertools import chain
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import numpy as np


def get_eyes((x, y, w, h)):
    roi_gray = gray[y:y+h, x:x+w]
    eyes = find_eyes.detectMultiScale(roi_gray)
    return sorted(
        map(
            lambda (ex, ey, ew, eh): (ex+x, ey+y, ew, eh),
            eyes
        ),
        key=lambda ((x, y, w, h)): y
    )[:2]


def reversedim(M, k=0):
    idx = tuple(
        (
            slice(None, None, -1) if ii == k else slice(None)
            for ii in xrange(M.ndim)
        )
    )
    return M[idx]


def remove_alpha(img):
    return cv2.merge(
        (
            img[:, :, 0],
            img[:, :, 1],
            img[:, :, 2]
        )
    )


def midpoint((x, y, w, h)):
    return (x + w/2, y + h/2)


def t_midpoint((x0, y0), (x1, y1)):
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def alpha_composite(src, dst):
    '''
    Return the alpha composite of src and dst.

    Parameters:
    src -- PIL RGBA Image object
    dst -- PIL RGBA Image object

    The algorithm comes from http://en.wikipedia.org/wiki/Alpha_compositing
    '''
    # http://stackoverflow.com/a/3375291/190597
    # http://stackoverflow.com/a/9166671/190597
    out = np.empty(src.shape, dtype='float')
    alpha = np.index_exp[:, :, 3:]
    rgb = np.index_exp[:, :, :3]
    src_a = src[alpha]/255.0
    dst_a = dst[alpha]/255.0
    out[alpha] = src_a+dst_a*(1-src_a)
    old_setting = np.seterr(invalid='ignore')
    out[rgb] = (src[rgb]*src_a + dst[rgb]*dst_a*(1-src_a))/out[alpha]
    np.seterr(**old_setting)
    out[alpha] *= 255
    np.clip(out, 0, 255)
    # astype('uint8') maps np.nan (and np.inf) to 0
    out = out.astype('uint8')
    return out


def tuple_minus((x0, y0), (x1, y1)):
    return (x0 - x1, y0 - y1)


def tuple_atan2((x0, y0)):
    return math.atan2(y0, x0)


def rotate((x, y), r):
    cos = math.cos(r)
    sin = math.sin(r)
    return (
        x * cos - y * sin,
        y * cos + x * sin
    )


# a0, a1 = target
# b0, b1 = current
def get_rot_difference((a0, a1), (b0, b1)):
    x0, y0 = tuple_minus(a0, a1)
    x1, y1 = tuple_minus(b0, b1)
    t0 = math.atan2(y0, x0)
    t1 = math.atan2(y1, x1)
    return t1 - t0


def flatten(seq):
    return chain.from_iterable(seq)


def get_r_l((x0, y0), (x1, y1)):
    if x0 < x1:
        return ((x0, y0), (x1, y1))
    else:
        return ((x1, y1), (x0, y0))


def transform_img((outW, outH), (x, y), scale, r, img):
    h, w, _ = img.shape

    trans0 = np.array(
        [[1, 0, (outW - w)/2],
         [0, 1, (outH - h)/2]],
        np.float64
    )
    rotmat = cv2.getRotationMatrix2D(
        (outW/2, outH/2),
        r,
        scale
    )
    trans1 = np.array(
        [[1, 0, x - outW/2],
         [0, 1, y - outH/2]],
        np.float64
    )

    centered = cv2.warpAffine(
        img,
        trans0,
        (outW, outH),
        cv2.INTER_CUBIC
    )
    rotated = cv2.warpAffine(
        centered,
        rotmat,
        (outW, outH),
        cv2.INTER_CUBIC
    )
    translated = cv2.warpAffine(
        rotated,
        trans1,
        (outW, outH),
        cv2.INTER_CUBIC
    )

    return translated


def distance((x0, y0), (x1, y1)):
    x = x0 - x1
    y = y0 - y1
    return math.sqrt(x*x + y*y)


def get_scale((t0, t1), (c0, c1)):
    return distance(t0, t1) / distance(c0, c1)


def map_frame(input_img, glasses_img, glasses_eyes, eyes):
    return lambda (perc): frame(
        np.copy(input_img),
        np.copy(glasses_img),
        glasses_eyes,
        eyes,
        perc
    )


def frame(input_img, glasses_img, (g_eye0, g_eye1), (eye0, eye1), perc):
    eye0, eye1 = get_r_l(eye0, eye1)

    h, w, _ = input_img.shape
    x, y = t_midpoint(eye0, eye1)
    g_x, g_y = t_midpoint(g_eye0, g_eye1)
    g_h, g_w, _ = glasses_img.shape

    drop_start = 0

    scale = get_scale((eye0, eye1), (g_eye0, g_eye1))
    glasses_rot = get_rot_difference((eye0, eye1), (g_eye0, g_eye1))
    offset_x, offset_y = rotate(
        tuple_minus((g_w/2, g_h/2), (g_x, g_y)),
        -glasses_rot
    )
    target_x, target_y = (
        x + offset_x * scale,
        y + offset_y * scale
    )

    current_height = drop_start + perc * target_y

    glasses_rot_deg = glasses_rot * (180 / math.pi)
    rotated_glasses = transform_img(
        (w, h),
        (target_x, current_height),
        scale,
        glasses_rot_deg,
        glasses_img
    )

    return alpha_composite(rotated_glasses, input_img)


def scale_img(img, (w, h)):
    ih, iw, _ = img.shape
    scale = min(float(w)/iw, float(h)/ih)
    ow, oh = (int(iw*scale), int(ih*scale))
    mat = cv2.getRotationMatrix2D(
        (0, 0),
        0,
        scale
    )
    return cv2.warpAffine(
        img,
        mat,
        (ow, oh),
        cv2.INTER_CUBIC
    )


def map_deal_with_it(img):
    return lambda (t): deal_with_it(np.copy(img), t)


def normalize(l):
    size = len(l)
    tot = sum(l)
    return map(
        lambda a: a*size/tot,
        l
    )


def deal_with_it(img, t):
    r = (math.sin(t) + 1) / 2
    g = (math.sin(t - math.pi / 3) + 1) / 2
    b = (math.sin(t - 2 * math.pi / 3) + 1) / 2
    r, g, b = normalize([r, g, b])
    img = cv2.merge(
        (
            img[:, :, 0] * r,
            img[:, :, 1] * g,
            img[:, :, 2] * b
        )
    )

    h, w, _ = img.shape
    x, y = 30, h - 30
    iwidthscale = 0.0038
    ithickscale = 0.03
    innerthickscale = 0.7
    scale = w * iwidthscale
    thickness = int(w * ithickscale)
    inner_thickness = int(innerthickscale * thickness)
    text = 'DEAL WITH IT'

    cv2.putText(
        img,
        text,
        (x, y),
        0,
        scale,
        (255, 255, 255),
        thickness
    )
    cv2.putText(
        img,
        text,
        (x, y),
        0,
        scale,
        (0, 0, 0),
        inner_thickness
    )

    return img

img = cv2.imread('data/joker-face-66.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

out_img = img if img.shape[2] == 4 else cv2.merge(
    (
        img[:, :, 0],
        img[:, :, 1],
        img[:, :, 2],
        np.ones(img.shape[:2], img.dtype)
    )
)

glasses = cv2.imread('data/glasses.png', -1)

gscale = 0.25
glasses_left_eye, glasses_right_eye = (
    (gscale*260, gscale*40),
    (gscale*460, gscale*50)
)

find_face = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
find_eyes = cv2.CascadeClassifier('data/haarcascade_eye.xml')

# faces = find_face.detectMultiScale(gray, 1.3, 5)
faces = [(0, 0, img.shape[1], img.shape[0])]

eyes = map(
    lambda ((a, b)): (midpoint(a), midpoint(b)),
    map(get_eyes, faces)
)

num_frames = 10
deal_frames = 20
deal_t_range = math.pi * 2

out_size = (600, 400)

frames = map(
    map_frame(out_img, glasses, (glasses_left_eye, glasses_right_eye), eyes[0]),
    map(
        lambda (x): float(x+1) / num_frames,
        range(num_frames)
    )
)

final_frame = scale_img(np.copy(frames[-1]), out_size)

rgb_frames = chain(
    map(
        lambda f: scale_img(reversedim(f, 2), out_size),
        map(
            remove_alpha,
            frames
        )
    ),
    map(
        map_deal_with_it(final_frame),
        map(
            lambda (x): float(x * deal_t_range) / deal_frames,
            range(deal_frames)
        )
    )
)

ImageSequenceClip(list(rgb_frames), fps=10).write_gif('data/test.gif')
