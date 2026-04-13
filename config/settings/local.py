from config.settings.settings_base import *  # noqa: F401


STAGE = "local"
DEBUG = True

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "../db.sqlite3",
    },
}
