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

import sys
import glob
import logging
from enum import Enum, auto
from os import path, mkdir
from shutil import copyfile

from lxml import etree

from ..preferences.preferences import Preferences
from ..core.utils.logging import log_message, date_format

logger = logging.getLogger(__name__)


def main():
    docs_parent_dir = Preferences.bnc_processing_metas["raw"].path
    out_text_dir    = Preferences.bnc_text_processing_metas["raw"].path
    out_speech_dir  = Preferences.bnc_speech_processing_metas["raw"].path

    logger.info("Sorting corpus documents")

    for doc_i, source_doc_path in enumerate(glob.glob(path.join(docs_parent_dir, "**/*.xml"), recursive=True)):

        xml_doc = etree.parse(source_doc_path)
        xml_root = xml_doc.getroot()

        # Written documents contain the wtext node
        wtext = xml_root.find(".//wtext")
        # Spoken documents contain the stext node
        stext = xml_root.find(".//stext")
        # Written-for-speech documents contain xtext, but also a specific type code in the catRef node
        catref = xml_root.find(".//catRef")

        # Figure out what kind of document this is
        # Document classifications from http://www.natcorp.ox.ac.uk/docs/catRef.xml
        if catref is not None:
            if "ALLTYP1" in catref.get("targets").split():
                document_type = BncDocumentType.alltyp1_spoken_demographically_sampled
            elif "ALLTYP2" in catref.get("targets").split():
                document_type = BncDocumentType.alltyp2_spoken_context_governed
            elif "ALLTYP3" in catref.get("targets").split():
                document_type = BncDocumentType.alltyp3_written_books_and_periodicals
            elif "ALLTYP4" in catref.get("targets").split():
                document_type = BncDocumentType.alltyp4_written_to_be_spoken
            elif "ALLTYP5" in catref.get("targets").split():
                document_type = BncDocumentType.alltyp5_written_miscellaneous
            else:
                raise ImportError(f"No type found for {source_doc_path}")
        else:
            raise ImportError(f"No type found for {source_doc_path}")

        # Set the appropriate destination for this document
        if document_type == BncDocumentType.alltyp1_spoken_demographically_sampled:
            destination_dir = out_speech_dir
        elif document_type == BncDocumentType.alltyp2_spoken_context_governed:
            destination_dir = out_speech_dir
        elif document_type == BncDocumentType.alltyp3_written_books_and_periodicals:
            destination_dir = out_text_dir
        elif document_type == BncDocumentType.alltyp4_written_to_be_spoken:
            # Text for speech is still text
            destination_dir = out_text_dir
        elif document_type == BncDocumentType.alltyp5_written_miscellaneous:
            destination_dir = out_text_dir
        else:
            # This error logically cannot be raised, it's just here to help PyCharm's static analysis.
            raise ImportError()

        if not path.isdir(destination_dir):
            mkdir(destination_dir)
        target_doc_path = path.join(destination_dir, path.basename(source_doc_path))

        copyfile(source_doc_path, target_doc_path)

        # Occasional feedback
        if doc_i % 100 == 0 and doc_i > 0:
            logger.info(f"\t{doc_i} files")


class BncDocumentType(Enum):
    """
    The type of a document within the corpus
    """
    alltyp1_spoken_demographically_sampled = auto()
    alltyp2_spoken_context_governed        = auto()
    alltyp3_written_books_and_periodicals  = auto()
    alltyp4_written_to_be_spoken           = auto()
    alltyp5_written_miscellaneous          = auto()


if __name__ == '__main__':
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
