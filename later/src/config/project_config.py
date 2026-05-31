import base64
from datetime import datetime
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


class project_config:
    # load env
    project_root = Path(__file__).parent.parent.parent.parent
    ENV_FILE_NAME = os.getenv("ENV_FILE_NAME", ".env")
    config_path = os.path.join(project_root.absolute(), ENV_FILE_NAME)
    load_dotenv(dotenv_path=config_path, override=False)

    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3_32b")
    API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "")
    BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "Qwen3-14B")

    # python config
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "-1"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))
    TOP_P = float(os.getenv("TOP_P", "0.95"))
    GENERATE_BS = int(os.getenv("GENERATE_BS", "1"))
    TEXT_MAS_CONTEXT_LENGTH = int(os.getenv("TEXT_MAS_CONTEXT_LENGTH", "-1"))
    LATENT_STEPS = int(os.getenv("LATENT_STEPS", "50"))

    METHOD = os.getenv("METHOD", "latent_switch")
    TASK = os.getenv("TASK", "aime2025")
    EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "baseline_test")

    SPLIT = os.getenv("SPLIT", "test")

    MAX_STEPS = int(os.getenv("MAX_STEPS", "4096"))
    CHECK_N_TOKENS = int(os.getenv("CHECK_N_TOKENS", "5"))
    ENTROPY_THRESHOLD = float(os.getenv("ENTROPY_THRESHOLD", "1.2"))
    LATENT_ENTROPY_THRESHOLD = float(os.getenv("LATENT_ENTROPY_THRESHOLD", "5"))
    COT_SWITCH_ENTROPY_THRESHOLD = float(os.getenv("COT_SWITCH_ENTROPY_THRESHOLD", "3"))
    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "38912"))
    MAX_LATENT_REASONING_STEPS = int(os.getenv("MAX_LATENT_REASONING_STEPS", "1"))
    LATENT_TOKENS_LIMIT = int(os.getenv("LATENT_TOKENS_LIMIT", "50"))
    EXPLICIT_TOKENS_LIMIT = int(os.getenv("EXPLICIT_TOKENS_LIMIT", "128"))
    STEP_DELIMITER = os.getenv("STEP_DELIMITER", "\n\n")
    STOP_LATENT_BY_TOKEN = os.getenv("STOP_LATENT_BY_TOKEN", "true").lower() == "true"

    IF_BOS = os.getenv("IF_BOS", "false").lower() == "true"
    IF_SEQUENCIAL = os.getenv("IF_SEQUENCIAL", "false").lower() == "true"
    IF_SEQUENCIAL_NOTHINK = os.getenv("IF_SEQUENCIAL_NOTHINK", "false").lower() == "true"
    IF_EXPLICIT_MODEL = os.getenv("IF_EXPLICIT_MODEL", "false").lower() == "true"
    SKIP_SPECIAL_TOKENS = os.getenv("SKIP_SPECIAL_TOKENS", "true").lower() == "true"

    IF_EVALUATE_WITH_INSIGHT = os.getenv("IF_EVALUATE_WITH_INSIGHT", "false").lower() == "true"

    ### draw entropy
    WITH_ENTROPY = os.getenv("WITH_ENTROPY", "true").lower() == "true"
    DRAW_ENTROPY = os.getenv("DRAW_ENTROPY", "false").lower() == "true"
    ENTROPY_VIZ_DIR = os.getenv("ENTROPY_VIZ_DIR", str(project_root / "results"))

    DRAW_ATTENTION = os.getenv("DRAW_ATTENTION", "false").lower() == "true"
    SAVE_STATES = os.getenv("SAVE_STATES", "false").lower() == "true"

    # logger
    PRINT_TERMINAL = os.getenv("PRINT_TERMINAL", "true").lower() == "true"
    PRINT_TERMINAL_LEVEL = os.getenv("PRINT_TERMINAL_LEVEL", "DEBUG")
    PRINT_FILE = os.getenv("PRINT_FILE", "false").lower() == "true"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    EXPERIMENT_LOG_DIR = os.path.join(
        project_root, "logs", EXPERIMENT_NAME, TASK, METHOD, BASE_MODEL_NAME.split("/")[-1], timestamp
    )
    EXPERIMENT_RESULT_DIR = os.path.join(
        ENTROPY_VIZ_DIR, EXPERIMENT_NAME, TASK, METHOD, BASE_MODEL_NAME.split("/")[-1], timestamp
    )

    DEFAULT_DISTILLED_REASONING_PATH = os.getenv("DEFAULT_DISTILLED_REASONING_PATH", str(project_root / "data" / "latent_reasoning_distill" / "distilled_latent_reasoning.jsonl"))

    logger.remove()  # remove handel
    # print to terminal
    if PRINT_TERMINAL:
        logger.add(sys.stdout, level=PRINT_TERMINAL_LEVEL)  # stderr
    # print to file
    if PRINT_FILE:
        if DEBUG:
            # logger.add(
            #     os.path.join(EXPERIMENT_LOG_DIR, "debug.log"), level="DEBUG", rotation="1 MB", retention="7 days"
            # )
            logger.add(
                os.path.join(EXPERIMENT_LOG_DIR, "debug_{time:YYYY-MM-DD_HH-mm-ss}.log"),
                level="DEBUG",
                rotation="2 MB",
                retention="90 days",
            )
        logger.add(
            os.path.join(EXPERIMENT_LOG_DIR, "info_warning.log"),
            level="INFO",
            filter=lambda record: record["level"].no < logger.level("ERROR").no,
            rotation="2 MB",
        )
        logger.add(os.path.join(EXPERIMENT_LOG_DIR, "error.log"), level="ERROR", rotation="2 MB")


def print_config():
    try:
        config_details = ["\n" + "=" * 20 + " Project Configuration " + "=" * 20]
        for key, value in vars(project_config).items():
            if key.isupper() and not key.startswith("__"):
                if "KEY" in key or "SECRET" in key or "AUTH" in key:
                    value = f"***{str(value)[-4:]}"
                config_details.append(f"    {key:<30}: {value}")
        config_details.append("=" * 63 + "\n")
        logger.debug("\n".join(config_details))
    except Exception as e:
        logger.error(f"Failed to log project configuration: {e}")
