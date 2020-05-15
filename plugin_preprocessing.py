#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import stanza

from deepnlpf.core.iplugin import IPlugin


class Plugin(IPlugin):
    """This plugin contains features for pre-processing data."""

    def __init__(self, document:str, pipeline:dict):
        """Loads all settings.

        Arguments:
            document {str} -- document containing the text to be processed.
            pipeline {dict} -- pipeline customized by the user.
        """

        self._document = document
        self._pipeline = pipeline
        
        self._toolnlp, self._processors = self.get_toolnlp_base()
        
        self._lang = pipeline["lang"]

    def run(self):
        doc_annotation = self.wrapper()
        # annotation = self.out_format(doc_annotation)
        return doc_annotation

    def wrapper(self):

        if self._toolnlp == "stanza":
            nlp = stanza.Pipeline(
                lang=self._lang, processors=", ".join(self._processors), use_gpu=False,
            )

            doc = nlp(self._document)

            return doc
        elif self._toolnlp == "stanfordcorenlp":
            pass

    def out_format(self, doc):
        doc_json = json.loads(str(doc))
        return doc_json

    def get_toolnlp_base(self):
        """Select which NLP tool will be used as the basis for all other tools.

        Returns:
            [type] -- [description]
        """
        if self._pipeline["tools"]["stanza"]["processors"]:
            return "stanza", self._pipeline["tools"]["stanza"]["processors"]
        elif self._pipeline["tools"]["stanfordcorenlp"]["processors"]:
            return "stanfordcorenlp", self._pipeline["tools"]["stanfordcorenlp"]["processors"]
        else:
            print("This tool nlp is not defined as a base tool, try: [stanza|stanfordecorenlp]")
            sys.exit(0)
