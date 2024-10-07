import os


class DatasetCatalog:
    def __init__(self):
        # the following dataset utilized for encoding-side alignment learning
        self.audiocap_enc = {
            "target": "nextgpt.dataset.base_dataset.LazySupervisedDataset",
            "params": dict(
                data_path="./data/T_X_pair_data/audiocap/audiocap_comprehension.json",
                audio_folder="./data/T_X_pair_data/audiocap/audios",
                image_folder="./data/T_X_pair_data/cc3m/images",
                video_folder="./data/T_X_pair_data/webvid/videos",
            ),
        }

        self.webvid_enc = {
            "target": "nextgpt.dataset.base_dataset.LazySupervisedDataset",
            "params": dict(
                data_path="./data/T_X_pair_data/webvid/webvid_comprehension.json",
                video_folder="./data/T_X_pair_data/webvid/videos",
                audio_folder="./data/T_X_pair_data/audiocap/audios",
                image_folder="./data/T_X_pair_data/cc3m/images",
            ),
        }

        self.cc3m_enc = {
            "target": "nextgpt.dataset.base_dataset.LazySupervisedDataset",
            "params": dict(
                data_path="./data/T_X_pair_data/cc3m/cc3m_comprehension.json",
                image_folder="./data/T_X_pair_data/cc3m/images",
                video_folder="./data/T_X_pair_data/webvid/videos",
                audio_folder="./data/T_X_pair_data/audiocap/audios",
            ),
        }

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

        # the following dataset utilized for decoding-side alignment learning.

        self.audiocap_dec = {
            "target": "nextgpt.dataset.base_dataset.LazySupervisedDataset",
            "params": dict(
                data_path="./data/T_X_pair_data/audiocap/audiocap_generation.json",
                audio_folder="./data/T_X_pair_data/audiocap/audios",
                audio_caption_emb_folder="./data/embed/",
                image_folder="./data/T_X_pair_data/cc3m/images",
                image_caption_emb_folder="./data/embed/",
                video_folder="./data/T_X_pair_data/webvid/videos",
                video_caption_emb_folder="./data/embed/",
            ),
        }

        self.webvid_dec = {
            "target": "nextgpt.dataset.base_dataset.LazySupervisedDataset",
            "params": dict(
                data_path="./data/T_X_pair_data/webvid/webvid_generation.json",
                audio_folder="./data/T_X_pair_data/audiocap/audios",
                audio_caption_emb_folder="./data/embed/",
                image_folder="./data/T_X_pair_data/cc3m/images",
                image_caption_emb_folder="./data/embed/",
                video_folder="./data/T_X_pair_data/webvid/videos",
                video_caption_emb_folder="./data/embed/",
            ),
        }

        self.cc3m_dec = {
            "target": "nextgpt.dataset.base_dataset.LazySupervisedDataset",
            "params": dict(
                data_path="./data/T_X_pair_data/cc3m/cc3m_generation.json",
                audio_folder="./data/T_X_pair_data/audiocap/audios",
                audio_caption_emb_folder="./data/embed/",
                image_folder="./data/T_X_pair_data/cc3m/images",
                image_caption_emb_folder="./data/embed/",
                video_folder="./data/T_X_pair_data/webvid/videos",
                video_caption_emb_folder="./data/embed/",
                
            ),
        }

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

        # the following dataset utilized for instruction tuning, so they are instruction dataset.

        self.mosit_instruction = {
            "target": "nextgpt.dataset.base_dataset.LazySupervisedDataset",
            "params": dict(
                data_path="./data/IT_data/MosIT_data/mosit_instruction_tuning.json",
                audio_caption_emb_folder="./data/it/embed/",
                video_caption_emb_folder="./data/it/embed/",
                image_caption_emb_folder="./data/it/embed/",
                video_folder=None,
                audio_folder=None,
                image_folder="./data/IT_data/MosIT_data/images",
            ),
        }

        self.audio_instruction = {
            "target": "nextgpt.dataset.base_dataset.LazySupervisedDataset",
            "params": dict(
                data_path="./data/IT_data/T-T+X_data/audio_instruction_tuning.json",
                audio_caption_emb_folder="./data/it/embed/",
                video_caption_emb_folder="./data/it/embed/",
                image_caption_emb_folder="./data/it/embed/",
                video_folder='./data/T_X_pair_data/webvid/videos',
                audio_folder='./data/T_X_pair_data/audiocap/audios',
                image_folder="./data/T_X_pair_data/cc3m/images",
            ),
        }

        self.video_instruction = {
            "target": "nextgpt.dataset.base_dataset.LazySupervisedDataset",
            "params": dict(
                data_path="./data/IT_data/T-T+X_data/video_instruction_tuning.json",
                audio_caption_emb_folder="./data/it/embed/",
                video_caption_emb_folder="./data/it/embed/",
                image_caption_emb_folder="./data/it/embed/",
                video_folder='./data/T_X_pair_data/webvid/videos',
                audio_folder='./data/T_X_pair_data/audiocap/audios',
                image_folder="./data/T_X_pair_data/cc3m/images",
            ),
        }

        self.image_instruction = {
            "target": "nextgpt.dataset.base_dataset.LazySupervisedDataset",
            "params": dict(
                data_path="./data/IT_data/T-T+X_data/image_instruction_tuning.json",
                audio_caption_emb_folder="./data/it/embed/",
                video_caption_emb_folder="./data/it/embed/",
                image_caption_emb_folder="./data/it/embed/",
                video_folder='./data/T_X_pair_data/webvid/videos',
                audio_folder='./data/T_X_pair_data/audiocap/audios',
                image_folder="./data/T_X_pair_data/cc3m/images",
            ),
        }

        self.llava_instruction = {
            "target": "nextgpt.dataset.base_dataset.LazySupervisedDataset",
            "params": dict(
                data_path="./data/IT_data/T+X-T_data/llava/preprocessed_llava.json",
                image_folder="./data/IT_data/T+X-T_data/llava/images",
                audio_caption_emb_folder="./data/it/embed/",
                video_caption_emb_folder="./data/it/embed/",
                image_caption_emb_folder="./data/it/embed/",
                video_folder=None,
                audio_folder=None,
            ),
        }

        self.alpaca_instruction = {
            "target": "nextgpt.dataset.base_dataset.LazySupervisedDataset",
            "params": dict(
                data_path="./data/IT_data/T+X-T_data/alpaca/preprocessed_alpaca.json",
                audio_caption_emb_folder="./data/it/embed/",
                video_caption_emb_folder="./data/it/embed/",
                image_caption_emb_folder="./data/it/embed/",
                video_folder=None,
                audio_folder=None,
                image_folder="./data/llava-150k/images",
            ),
        }

        self.videochat_instruction = {
            "target": "nextgpt.dataset.base_dataset.LazySupervisedDataset",
            "params": dict(
                data_path="./data/IT_data/T+X-T_data/videochat/videochat.json",
                video_folder="../data/IT_data/T+X-T_data/videochat/video",
                audio_caption_emb_folder="./data/it/embed/",
                video_caption_emb_folder="./data/it/embed/",
                image_caption_emb_folder="./data/it/embed/",
                image_folder=None,
                audio_folder=None,
            ),
        }