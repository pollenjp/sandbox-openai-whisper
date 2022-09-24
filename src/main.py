# Standard Library
from logging import NullHandler
from logging import getLogger
from pathlib import Path

# Third Party Library
import pandas as pd
import whisper

# First Party Library
from whisper_demo.utils import get_available_device

logger = getLogger(__name__)
logger.addHandler(NullHandler())


def init_config() -> None:
    pd.options.display.max_rows = 100
    pd.options.display.max_colwidth = 1000


def get_language(locale: str) -> str:
    languages = {
        "af_za": "Afrikaans",
        "am_et": "Amharic",
        "ar_eg": "Arabic",
        "as_in": "Assamese",
        "az_az": "Azerbaijani",
        "be_by": "Belarusian",
        "bg_bg": "Bulgarian",
        "bn_in": "Bengali",
        "bs_ba": "Bosnian",
        "ca_es": "Catalan",
        "cmn_hans_cn": "Chinese",
        "cs_cz": "Czech",
        "cy_gb": "Welsh",
        "da_dk": "Danish",
        "de_de": "German",
        "el_gr": "Greek",
        "en_us": "English",
        "es_419": "Spanish",
        "et_ee": "Estonian",
        "fa_ir": "Persian",
        "fi_fi": "Finnish",
        "fil_ph": "Tagalog",
        "fr_fr": "French",
        "gl_es": "Galician",
        "gu_in": "Gujarati",
        "ha_ng": "Hausa",
        "he_il": "Hebrew",
        "hi_in": "Hindi",
        "hr_hr": "Croatian",
        "hu_hu": "Hungarian",
        "hy_am": "Armenian",
        "id_id": "Indonesian",
        "is_is": "Icelandic",
        "it_it": "Italian",
        "ja_jp": "Japanese",
        "jv_id": "Javanese",
        "ka_ge": "Georgian",
        "kk_kz": "Kazakh",
        "km_kh": "Khmer",
        "kn_in": "Kannada",
        "ko_kr": "Korean",
        "lb_lu": "Luxembourgish",
        "ln_cd": "Lingala",
        "lo_la": "Lao",
        "lt_lt": "Lithuanian",
        "lv_lv": "Latvian",
        "mi_nz": "Maori",
        "mk_mk": "Macedonian",
        "ml_in": "Malayalam",
        "mn_mn": "Mongolian",
        "mr_in": "Marathi",
        "ms_my": "Malay",
        "mt_mt": "Maltese",
        "my_mm": "Myanmar",
        "nb_no": "Norwegian",
        "ne_np": "Nepali",
        "nl_nl": "Dutch",
        "oc_fr": "Occitan",
        "pa_in": "Punjabi",
        "pl_pl": "Polish",
        "ps_af": "Pashto",
        "pt_br": "Portuguese",
        "ro_ro": "Romanian",
        "ru_ru": "Russian",
        "sd_in": "Sindhi",
        "sk_sk": "Slovak",
        "sl_si": "Slovenian",
        "sn_zw": "Shona",
        "so_so": "Somali",
        "sr_rs": "Serbian",
        "sv_se": "Swedish",
        "sw_ke": "Swahili",
        "ta_in": "Tamil",
        "te_in": "Telugu",
        "tg_tj": "Tajik",
        "th_th": "Thai",
        "tr_tr": "Turkish",
        "uk_ua": "Ukrainian",
        "ur_pk": "Urdu",
        "uz_uz": "Uzbek",
        "vi_vn": "Vietnamese",
        "yo_ng": "Yoruba",
    }
    return languages[locale]


def setup_config() -> None:
    # Standard Library
    from logging.config import dictConfig

    dictConfig(
        {
            "version": 1,
            "formatters": {
                "console_formatter": {
                    "format": "".join(
                        [
                            "[%(asctime)s]",
                            "[%(name)20s]",
                            "[%(levelname)10s]",
                            "[%(threadName)10s]",
                            "[%(processName)10s]",
                            "[%(filename)20s:%(lineno)4d]",
                            " - %(message)s",
                        ]
                    ),
                },
                # "file_formatter": {
                #     "format": "".join(["%(message)s"]),
                # },
            },
            "handlers": {
                "console_handler": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "console_formatter",
                },
                # "file_handler": {
                #     "class": "logging.FileHandler",
                #     "level": "INFO",
                #     "formatter": "file_formatter",
                #     "filename": "output.log",
                # },
            },
            "loggers": {
                "__main__": {
                    "level": "INFO",
                    "handlers": [
                        "console_handler",
                        # "file_handler",
                    ],
                },
            },
        }
    )


def main() -> None:

    setup_config()

    init_config()

    locale: str = "ja_jp"
    language: str = get_language(locale)

    # Load audio
    fname = Path("My-Audio.wav")
    audio = whisper.load_audio(fname)
    audio = whisper.pad_or_trim(audio)

    logger.info("loading model...")
    model = whisper.load_model("medium", device=get_available_device())
    logger.info("model loaded")

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Output the recognized text
    options = whisper.DecodingOptions(
        language=language,
        # without_timestamps=True,
    )
    result = whisper.decode(model, mel, options)
    logger.info(result.text)


if __name__ == "__main__":
    main()
