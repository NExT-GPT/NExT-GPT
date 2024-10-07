# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15
LOGDIR = "."
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"
IMAGE_SIGNAL_TOKEN = "<image_{:05d}>"
MAX_IMAGE_LENGTH = 16

VIDEO_TOKEN_INDEX = -300
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"
VIDEO_PLACEHOLDER = "<video-placeholder>"
VIDEO_SIGNAL_TOKEN = "<video_{:05d}>"
MAX_VIDEO_LENGTH = 16


AUDIO_TOKEN_INDEX = -400
DEFAULT_AUDIO_TOKEN = "<audio>"
DEFAULT_AUDIO_PATCH_TOKEN = "<aud_patch>"
DEFAULT_AUD_START_TOKEN = "<aud_start>"
DEFAULT_AUD_END_TOKEN = "<aud_end>"
AUDIO_PLACEHOLDER = "<audio-placeholder>"
AUDIO_SIGNAL_TOKEN = "<audio_{:05d}>"
MAX_AUDIO_LENGTH = 16


