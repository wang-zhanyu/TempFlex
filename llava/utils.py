import datetime
import logging
import logging.handlers
import os
import sys
import re
import torch
import numpy as np
import zipfile
import requests
from io import BytesIO
from llava.constants import LOGDIR
from PIL import Image

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "I am sorry. Your input may violate our content moderation guidelines. Please avoid using harmful or offensive content."

handler = None

import torch.distributed as dist

try:
    import av
    from decord import VideoReader, cpu
except ImportError:
    print("Please install pyav to use video processing functions.")

def process_video_with_decord(video_file, data_args):
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    avg_fps = round(vr.get_avg_fps() / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
    frame_time = [i/avg_fps for i in frame_idx]

    
    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound or data_args.force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    
    video = vr.get_batch(frame_idx).asnumpy()
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    num_frames_to_sample = num_frames = len(frame_idx)
    # https://github.com/dmlc/decord/issues/208
    vr.seek(0)
    return video, video_time, frame_time, num_frames_to_sample


def extract_frame_index(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else -1

# read frames from zip file
def process_video_from_zip(zip_file, data_args):
    with zipfile.ZipFile(zip_file, 'r') as z:
        image_files = [name for name in z.namelist() if name.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_files = sorted(image_files, key=extract_frame_index)
        total_frame_num = len(image_files)
        avg_fps = round(data_args.video_fps)
        frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
        frame_time = [i / avg_fps for i in frame_idx]

        if data_args.frames_upbound > 0:
            if len(frame_idx) > data_args.frames_upbound or data_args.force_sample:
                uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()
                frame_time = [i / avg_fps for i in frame_idx]

        frames = []
        for index in frame_idx:
            image_name = image_files[index]
            with z.open(image_name) as image_file:
                image_data = image_file.read()
                image = Image.open(BytesIO(image_data)).convert('RGB')
                frames.append(np.array(image))

        # video = np.stack(frames, axis=0)
        frame_time_str = ",".join([f"{t:.2f}s" for t in frame_time])
        video_time = total_frame_num / data_args.video_fps
        num_frames_to_sample = len(frame_idx)

    return frames, video_time, frame_time_str, num_frames_to_sample

def process_video_with_pyav(video_file, data_args):
    container = av.open(video_file)
    # !!! This is the only difference. Using auto threading
    container.streams.video[0].thread_type = "AUTO"

    video_frames = []
    for packet in container.demux():
        if packet.stream.type == 'video':
            for frame in packet.decode():
                video_frames.append(frame)
    total_frame_num = len(video_frames)
    video_time = video_frames[-1].time
    avg_fps = round(total_frame_num / video_time / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]

    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()


    frames = [video_frames[i] for i in frame_idx]
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank_print(*args):
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)

def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(filename, when="D", utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False
    except KeyError as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


def extract_question_ids(input_ids: torch.Tensor) -> torch.Tensor:
    input_ids = input_ids.squeeze()

    # Only for Gemma
    START_OF_TURN = 105
    NEWLINE = 107
    END_OF_TURN = 106
    IMAGE = -200

    # 1. 找到第一个 <start_of_turn>
    start_indices = (input_ids == START_OF_TURN).nonzero(as_tuple=True)[0]
    if len(start_indices) == 0:
        raise ValueError("No <start_of_turn> token found.")
    start_idx = start_indices[0].item()

    # 2. 找到 start_idx 之后的第一个 \n
    newline_indices = (input_ids[start_idx + 1:] == NEWLINE).nonzero(as_tuple=True)[0]
    if len(newline_indices) == 0:
        raise ValueError("No newline after <start_of_turn> found.")
    user_newline_idx = newline_indices[0].item() + start_idx + 1

    # 3. 判断是否存在 <image> 占位符
    possible_image_idx = user_newline_idx + 1
    if possible_image_idx < len(input_ids) and input_ids[possible_image_idx].item() == IMAGE:
        # 跳过 <image> 和后面的 \n
        question_start = possible_image_idx + 2
    else:
        question_start = user_newline_idx + 1

    # 4. 找到第一个 <end_of_turn>
    end_of_turn_indices = (input_ids[question_start:] == END_OF_TURN).nonzero(as_tuple=True)[0]
    if len(end_of_turn_indices) == 0:
        raise ValueError("No <end_of_turn> token found after question.")
    end_of_turn_idx = end_of_turn_indices[0].item() + question_start

    # 5. 提取 question 的 input_ids
    question_ids = input_ids[question_start:end_of_turn_idx+1]

    return question_ids