"""
===========================
Separate speech and text documents from BNC.
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
import sys
import glob
import logging

from enum import Enum, auto
from shutil import copyfile

from lxml import etree

from ..preferences.preferences import Preferences
from ..core.utils.logging import log_message, date_format

logger = logging.getLogger(__name__)


def main():
    docs_parent_dir = Preferences.bnc_processing_metas["raw"].path
    out_text_dir    = Preferences.bnc_text_processing_metas["raw"].path
    out_speech_dir  = Preferences.bnc_speech_processing_metas["raw"].path

    logger.info("Detagging and sorting corpus documents")

    for doc_i, source_doc_path in enumerate(glob.glob(os.path.join(docs_parent_dir, "**/*.xml"), recursive=True)):

        xml_doc = etree.parse(source_doc_path)
        xml_root = xml_doc.getroot()

        # Written documents contain the wtext node
        wtext = xml_root.find(".//wtext")
        # Spoken documents contain the stext node
        stext = xml_root.find(".//stext")
        # Written-for-speech documents contain xtext, but also a specific type code in the catRef node
        catref = xml_root.find(".//catRef")

        # Figure out what kind of document this is
        if wtext is not None:
            # Written-for-speech documents are classified as "ALLTYP4"
            if catref is not None and "ALLTYP4" in catref.get("targets").split():
                this_doc_type = DocType.text_for_speech
            else:
                this_doc_type = DocType.text
        elif stext is not None:
            this_doc_type = DocType.speech
        else:
            raise ImportError(f"No type found for {source_doc_path}")

        # Set the appropriate destination for this document
        if this_doc_type == DocType.speech:
            destination_dir = out_speech_dir
        elif this_doc_type == DocType.text:
            destination_dir = out_text_dir
        elif this_doc_type == DocType.text_for_speech:
            destination_dir = out_text_dir
        else:
            # This error logically cannot be raised, it's just here to help PyCharm's static analysis.
            raise ImportError()

        target_doc_path = os.path.join(destination_dir, os.path.basename(source_doc_path))

        copyfile(source_doc_path, target_doc_path)

        # Occasional feedback
        if doc_i % 100 == 0 and doc_i > 0:
            logger.info(f"\t{doc_i} files")


class DocType(Enum):
    """
    The type of a document within the corpus
    """
    text            = auto()
    speech          = auto()
    text_for_speech = auto()


if __name__ == '__main__':
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
