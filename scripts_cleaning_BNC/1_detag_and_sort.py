import os
import sys
import glob
import logging

from enum import Enum

from lxml import etree

logger = logging.getLogger()


def main():
    docs_parent_dir         = "/Users/caiwingfield/corpora/BNC/0 XML version/Texts"
    out_text_dir            = "/Users/caiwingfield/corpora/BNC/1 Text"
    out_speech_dir          = "/Users/caiwingfield/corpora/BNC/1 Speech"
    out_text_for_speech_dir = "/Users/caiwingfield/corpora/BNC/1 Text for speech"
    # docs_parent_dir         = "/Users/caiwingfield/corpora/BNC-mini/0 Raw"
    # out_text_dir            = "/Users/caiwingfield/corpora/BNC-mini/1 Text"
    # out_speech_dir          = "/Users/caiwingfield/corpora/BNC-mini/1 Speech"
    # out_text_for_speech_dir = "/Users/caiwingfield/corpora/BNC-mini/1 Text for speech"

    xsl_filename = os.path.join(os.path.dirname(__file__), "justTheWords.xsl")
    xslt = etree.parse(xsl_filename)
    xslt_transform = etree.XSLT(xslt)

    logger.info("Detagging and sorting corpus documents")

    for i, doc_path in enumerate(glob.glob(os.path.join(docs_parent_dir, "**/*.xml"), recursive=True)):

        xml_doc = etree.parse(doc_path)
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
            logging.error(f"No type found for {doc_path}")

        # Move the document to the appropriate location
        if this_doc_type == DocType.speech:
            destination_dir = out_speech_dir
        elif this_doc_type == DocType.text:
            destination_dir = out_text_dir
        elif this_doc_type == DocType.text_for_speech:
            destination_dir = out_text_for_speech_dir

        newdom = xslt_transform(xml_doc)

        target_filename = os.path.join(destination_dir, os.path.splitext(os.path.basename(doc_path))[0] + ".txt")

        with open(target_filename, mode="w", encoding="utf-8") as target_file:
            target_file.write(str(newdom))

        if i % 100 == 0:
            logger.info(f"\t{i} files")


class DocType(Enum):
    """
    The type of a document within the corpus
    """
    text = 1
    speech = 2
    text_for_speech = 3


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
