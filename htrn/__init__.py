"""Shared code for the HTRN occlusion / MEG analysis pipeline."""
import logging


def setup_logging(level=logging.INFO):
    """Print log messages without prefixes; call once at the top of a script's ``main``."""
    logging.basicConfig(level=level, format='%(message)s')
