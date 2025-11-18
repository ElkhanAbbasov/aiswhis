import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

# --------------------
# Logging / output
# --------------------
RUN_NAME = "XTTSv2_Turkish_FT"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_tr_tr", "training")

# --------------------
# Training parameters
# --------------------
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True
START_WITH_EVAL = True
BATCH_SIZE = 4           # adjust if you OOM
GRAD_ACUMM_STEPS = 64    # ensure BATCH_SIZE * GRAD_ACUMM_STEPS >= ~256

# --------------------
# Dataset config (Turkish, LJSpeech-style formatter)
# --------------------
DATASET_ROOT = os.environ.get("XTTS_TR_DATASET_ROOT", "/content/drive/MyDrive/xtts_tr_dataset")
METADATA_FILE = os.environ.get("XTTS_TR_METADATA_FILE", os.path.join(DATASET_ROOT, "metadata.txt"))

config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="turkish_single_speaker",
    path=DATASET_ROOT,
    meta_file_train=METADATA_FILE,
    language="tr",
)

DATASETS_CONFIG_LIST = [config_dataset]

# --------------------
# Checkpoint paths (XTTS v2 + DVAE)
# --------------------
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2_tr_base")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK       = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE   = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

TOKENIZER_FILE_LINK   = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK  = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))

if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v2.0 files!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK],
        CHECKPOINTS_OUT_PATH,
        progress_bar=True,
    )

# --------------------
# Some short Turkish eval sentences for monitoring
# --------------------
SPEAKER_REFERENCE = [
    # any 3–6 sec clear samples from your dataset:
    # You can leave this empty; XTTS will still train, these are just for periodic audio eval.
]

LANGUAGE = config_dataset.language

# --------------------
# Early stopping (simple)
# --------------------
BEST_LOSS = None
MAX_PATIENCE = 1
CURRENT_PATIENCE = 0

def early_stopping_fn(eval_results):
    global BEST_LOSS, CURRENT_PATIENCE
    print(" > Early stopping hook")
    current_best_loss = eval_results.best_loss["eval_loss"]

    if BEST_LOSS is None:
        BEST_LOSS = current_best_loss
        return False

    if current_best_loss < BEST_LOSS:
        BEST_LOSS = current_best_loss
        CURRENT_PATIENCE = 0
        return False

    CURRENT_PATIENCE += 1
    if CURRENT_PATIENCE >= MAX_PATIENCE:
        print(" > Early stopping triggered")
        return True

    return False

# --------------------
# Main
# --------------------
def main():
    # model args (from XTTS v2 recipe, unchanged)
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6s
        min_conditioning_length=66150,   # 3s
        debug_loading_failures=False,
        max_wav_length=255995,           # ~11.6s
        max_text_length=300,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000,
    )

    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="GPT XTTS v2 Turkish fine-tuning",
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=4,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=10000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        print_eval=True,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-6,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={
            "milestones": [50000 * 18, 150000 * 18, 300000 * 18],
            "gamma": 0.5,
            "last_epoch": -1,
        },
        test_sentences=[
            {
                "text": "Merhaba, bu model Türkçe konuşmak için ince ayarlanmıştır.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "Uzun bir paragraf yerine daha doğal kısa cümleler kullanmak daha iyidir.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
        ],
        #change to 0.7 or 0.8 for more natural results when you have more samples
        eval_split_size=0.09, 
    )

    # Init XTTS GPT trainer model
    model = GPTTrainer.init_from_config(config)

    # Load dataset
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=False,
        eval_split_max_size=0,
        eval_split_size=0,
    )

    # hard-disable eval
    eval_samples = None

    # Trainer
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
            run_eval=False,
            # early_stopping_fn=early_stopping_fn,  # optional hook
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=None,   # no eval samples
    )

    trainer.run_eval = False
    trainer._eval_loader = None
    trainer.eval_loader = None

    trainer.fit()


if __name__ == "__main__":
    main()
