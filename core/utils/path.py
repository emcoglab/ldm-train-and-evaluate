"""
===========================
Path manipulation.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2017
---------------------------
"""
import os


def parent_directory(path: str) -> str:
    return os.path.join(path, os.pardir)
