"""
languages.py
------------
The languages Whisper can transcribe, shared by the CLI and the GUI.

The CLI has always accepted any Whisper language code, while the GUI offered a
short hand-picked list — so a Turkish recording could be transcribed from the
terminal but not from the browser. Both now read from here.
"""

# Whisper's own language table (code -> English name).
LANGUAGE_NAMES = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese", "az": "Azerbaijani",
    "ba": "Bashkir", "be": "Belarusian", "bg": "Bulgarian", "bn": "Bengali", "bo": "Tibetan",
    "br": "Breton", "bs": "Bosnian", "ca": "Catalan", "cs": "Czech", "cy": "Welsh",
    "da": "Danish", "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish", "fo": "Faroese",
    "fr": "French", "gl": "Galician", "gu": "Gujarati", "ha": "Hausa", "haw": "Hawaiian",
    "he": "Hebrew", "hi": "Hindi", "hr": "Croatian", "ht": "Haitian Creole", "hu": "Hungarian",
    "hy": "Armenian", "id": "Indonesian", "is": "Icelandic", "it": "Italian", "ja": "Japanese",
    "jw": "Javanese", "ka": "Georgian", "kk": "Kazakh", "km": "Khmer", "kn": "Kannada",
    "ko": "Korean", "la": "Latin", "lb": "Luxembourgish", "ln": "Lingala", "lo": "Lao",
    "lt": "Lithuanian", "lv": "Latvian", "mg": "Malagasy", "mi": "Maori", "mk": "Macedonian",
    "ml": "Malayalam", "mn": "Mongolian", "mr": "Marathi", "ms": "Malay", "mt": "Maltese",
    "my": "Myanmar", "ne": "Nepali", "nl": "Dutch", "nn": "Nynorsk", "no": "Norwegian",
    "oc": "Occitan", "pa": "Punjabi", "pl": "Polish", "ps": "Pashto", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sa": "Sanskrit", "sd": "Sindhi", "si": "Sinhala",
    "sk": "Slovak", "sl": "Slovenian", "sn": "Shona", "so": "Somali", "sq": "Albanian",
    "sr": "Serbian", "su": "Sundanese", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil",
    "te": "Telugu", "tg": "Tajik", "th": "Thai", "tk": "Turkmen", "tl": "Tagalog",
    "tr": "Turkish", "tt": "Tatar", "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek",
    "vi": "Vietnamese", "yi": "Yiddish", "yo": "Yoruba", "yue": "Cantonese", "zh": "Chinese",
}

# Shown first in the dropdown; the rest follow alphabetically by name.
COMMON = ["it", "en", "fr", "de", "es", "pt"]

AUTO = "auto"


def is_supported(code: str) -> bool:
    """True for 'auto' or any language Whisper knows."""
    value = (code or "").strip().lower()
    return value == AUTO or value in LANGUAGE_NAMES


def name_for(code: str) -> str:
    """English name of a language code, or the code itself if unknown."""
    return LANGUAGE_NAMES.get((code or "").strip().lower(), code)


def audio_options() -> list[tuple[str, str]]:
    """(value, label) pairs for the 'spoken language' dropdown."""
    rest = sorted(
        ((code, name) for code, name in LANGUAGE_NAMES.items() if code not in COMMON),
        key=lambda pair: pair[1],
    )
    return (
        [(AUTO, "Auto-detect")]
        + [(code, LANGUAGE_NAMES[code]) for code in COMMON]
        + rest
    )


def summary_options() -> list[tuple[str, str]]:
    """(value, label) pairs for the 'summary language' dropdown."""
    return [("same", "Same as the audio")] + audio_options()[1:]
