"""
Script for entity-wise calculation of precision, recall and F1 score according to the IOB2 scheme using the seqeval package

Prerequisites:
    - installed NTEE
    - a prediction file in the format generated by NTEE for a prediction

Customize settings:
    - TEI Reader Config
        in dictionary: current_tei_reader_config
    - Entity classes and entity class encoding
        in dictionary: current_tei_writer_mapping (= “entity_dict” part of a writer definition)
    - Language setting for correct tokenization (if the gold standard file is delivered in XML format)
        in string: current_language_for_sentence_seperation

Usage:
    - activate the virtual environment in which NTEE is installed
    - place this script in a folder together with two files: a gold standard file and a prediction file
    -- the gold standard file must be in XML or JSON format
    -- the prediction file should be "data_to_predict.pred.json", which is automatically generated by NTEE in the folder in which the prediction result is saved for each prediction
    - if all three files (script, gold standard, prediction) are in one folder, change to this folder in the terminal
    - input example: "python calc_seqeval_scores.py --true=filename_true.xml --pred=data_to_predict.pred.json --output=result.txt"
    - output: an overview with all scores is displayed in the terminal and (if a file name was passed to the optional parameter "output") saved in the output file in the same folder (if this file does not yet exist)

Parameters:
    --true: file path to the gold standard file (XML or JSON)
    --pred: file path to the prediction file (JSON)
    --output: file path to the file to be created in which the determined metrics are saved (if the file already exists, the results are only displayed in the terminal)
    --help: overview of the parameters
"""

# import modules
from typing import Union, List
from collections.abc import Iterable
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from tei_entity_enricher.util.spacy_lm import get_spacy_lm
import tei_entity_enricher.util.tei_parser as tp
import argparse
import logging
import os
import sys
import json

if __name__ == "__main__":
    # parse inline arguments
    parser = argparse.ArgumentParser(description="seqeval scores calculator")
    parser.add_argument(
        "--true",
        type=str,
        required=True,
        default=os.path.join(".", "true.json"),
        help="Path of the gold standard file",
    )
    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        default=os.path.join(".", "pred.json"),
        help="Path of the prediction file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path to creates result file to",
    )
    args = parser.parse_args()

    # check if files exist
    if not os.path.exists(args.true) or not os.path.exists(args.pred):
        logging.error(
            "One or both files refered to in --true and --pred parameters not found."
        )
        sys.exit()

    do_output = False

    if args.output is not None:
        if os.path.exists(args.output):
            logging.warning(
                "File refered to in --output parameter already exists. The result will only be printed in the terminal."
            )
        else:
            do_output = True

    # load content from input files and parse it into dicts
    # optional: transform xml of true file into json (IOB2 standard)
    def get_iob2_version_of_xml_file(
        filepath: str,
        tei_reader_config: dict,
        tei_writer_mapping: dict,
        language_for_sentence_seperation: str,
    ) -> list:
        teifile = tp.TEIFile(
            filename=filepath,
            tr_config=tei_reader_config,
            entity_dict=tei_writer_mapping,
            nlp=get_spacy_lm(lang=language_for_sentence_seperation),
            with_position_tags=True,
        )
        iob2_data = tp.split_into_sentences(teifile.build_tagged_text_line_list())
        return iob2_data

    # file in /TR_Configs folder of an ntee project folder
    current_tei_reader_config = {
        "exclude_tags": [], 
        "note_tags": [], 
        "name": "CANSpiN_Reader-Config", 
        "use_notes": False, 
        "template": False
    }
    # "entity_dict" part of a writer definition in /TNW folder of an ntee project folder
    current_tei_writer_mapping = {
        "Ort-Container": ["Ort-Container", {}], 
        "Ort-Container-BK": ["Ort-Container-BK", {}], 
        "Ort-Objekt": ["Ort-Objekt", {}], 
        "Ort-Objekt-BK": ["Ort-Objekt-BK", {}], 
        "Ort-Abstrakt": ["Ort-Abstrakt", {}], 
        "Ort-Abstrakt-BK": ["Ort-Abstrakt-BK", {}],
        "Ort-ALT": ["Ort-ALT", {}],
        "Bewegung-Subjekt": ["Bewegung-Subjekt", {}], 
        "Bewegung-Objekt": ["Bewegung-Objekt", {}], 
        "Bewegung-Licht": ["Bewegung-Licht", {}], 
        "Bewegung-Schall": ["Bewegung-Schall", {}], 
        "Bewegung-Geruch": ["Bewegung-Geruch", {}], 
        "Bewegung-ALT": ["Bewegung-ALT", {}],
        "Dimensionierung-Menge": ["Dimensionierung-Menge", {}], 
        "Dimensionierung-Abstand": ["Dimensionierung-Abstand", {}], 
        "Dimensionierung-Groesse": ["Dimensionierung-Groesse", {}], 
        "Dimensionierung-ALT": ["Dimensionierung-ALT", {}],
        "Richtung": ["Richtung", {}], 
        "Richtung-ALT": ["Richtung-ALT", {}], 
        "Positionierung": ["Positionierung", {}],
        "Positionierung-ALT": ["Positionierung-ALT", {}]
    }

    # available language for tokenization: German, English, Multilingual, French, Spanish
    current_language_for_sentence_seperation = "German"

    _, true_file_extension = os.path.splitext(args.true)

    retrieved_iob2_data = (
        get_iob2_version_of_xml_file(
            filepath=args.true,
            tei_reader_config=current_tei_reader_config,
            tei_writer_mapping=current_tei_writer_mapping,
            language_for_sentence_seperation=current_language_for_sentence_seperation,
        )
        if true_file_extension == ".xml"
        else None
    )

    # debug: transformation xml > json (uncomment the following 3 lines if your input is xml and you want to save the transformation result)
    # if retrieved_iob2_data:
        # with open("true.json", mode="w", encoding="utf8") as fw:
            # fw.write(json.dumps(retrieved_iob2_data))

    def get_true_file_content(
        filepath: str, retrieved_iob2_data: Union[dict, None]
    ) -> dict:
        if retrieved_iob2_data is None:
            with open(filepath, mode="r", encoding="utf8") as fw:
                result = json.load(fw)
                return result
        else:
            return retrieved_iob2_data

    y_true_file_content = get_true_file_content(args.true, retrieved_iob2_data)
    with open(args.pred, mode="r", encoding="utf8") as fw:
        y_pred_file_content = json.load(fw)

    # transform dicts into correct format, which seqeval can manage
    def delete_idx_0_and_2_of_string_list_with_3_members(input: list) -> None:
        if type(input) == list:
            if len(input) == 3 and type(input[0]) == str:
                del input[0]
                del input[1]
                if input[0] == "UNK":
                    input[0] = "O"
            else:
                for item in input:
                    delete_idx_0_and_2_of_string_list_with_3_members(item)

    def unpack_strings_out_of_lists(input: list) -> list:
        new_text = []
        for sentence in input:
            new_sentence = []
            for item in sentence:
                new_sentence.append(item[0])
            new_text.append(new_sentence)
        return new_text

    delete_idx_0_and_2_of_string_list_with_3_members(y_true_file_content)
    delete_idx_0_and_2_of_string_list_with_3_members(y_pred_file_content)
    y_true = unpack_strings_out_of_lists(y_true_file_content)
    y_pred = unpack_strings_out_of_lists(y_pred_file_content)

    # unify shape of y_true and y_pred lists by flattening and creating nested list structure 
    def flatten(list):
     for item in list:
         if isinstance(item, Iterable) and not isinstance(item, str):
             for x in flatten(item):
                 yield x
         else:        
             yield item
    y_true = list(flatten(y_true))
    y_pred = list(flatten(y_pred))

    words_per_sentence_amount = 20
    y_true = [[word for word in y_true[i:i + words_per_sentence_amount]] for i in range(0, len(y_true), words_per_sentence_amount)]
    y_pred = [[word for word in y_pred[i:i + words_per_sentence_amount]] for i in range(0, len(y_pred), words_per_sentence_amount)]

    # debug: print und save transformation and unification results (uncomment the following 6 lines if you want to debug)
    # print(f"y_true = {y_true}")
    # print(f"y_pred = {y_pred}")
    # with open("y_true.json", mode="w", encoding="utf8") as fw:
        # fw.write(json.dumps(y_true))
    # with open("y_pred.json", mode="w", encoding="utf8") as fw:
        # fw.write(json.dumps(y_pred))

    # get and print result
    def performance_measure(y_true: List[list[str]], y_pred: List[list[str]]) -> dict:
        """
        Compute the performance metrics: TP, FP, FN, TN for all categories
        Args:
            y_true : 2d array. Ground truth (correct) target values.
            y_pred : 2d array. Estimated targets as returned by a tagger.
        Returns:
            performance_dict : dict
        """
        performance_dict = dict()
        if any(isinstance(s, list) for s in y_true):
            y_true = [item for sublist in y_true for item in sublist]
            y_pred = [item for sublist in y_pred for item in sublist]
        performance_dict["TP_all"] = sum(
            y_t == y_p
            for y_t, y_p in zip(y_true, y_pred)
            if ((y_t != "O") or (y_p != "O"))
        )
        for entity in list(current_tei_writer_mapping.keys()):
            performance_dict[f"TP_{entity}"] = sum(
                y_t == y_p
                for y_t, y_p in zip(y_true, y_pred)
                if ((y_t == f"B-{entity}") or (y_t == f"I-{entity}"))
            )
        performance_dict["FP_all"] = sum(
            ((y_t != y_p) and (y_p != "O")) for y_t, y_p in zip(y_true, y_pred)
        )
        for entity in list(current_tei_writer_mapping.keys()):
            performance_dict[f"FP_{entity}"] = sum(
                ((y_t != y_p) and (y_p != "O"))
                for y_t, y_p in zip(y_true, y_pred)
                if (y_p == f"B-{entity}" or y_p == f"I-{entity}")
            )
        performance_dict["FN_all"] = sum(
            ((y_t != "O") and (y_p == "O")) for y_t, y_p in zip(y_true, y_pred)
        )
        for entity in list(current_tei_writer_mapping.keys()):
            performance_dict[f"FN_{entity}"] = sum(
                ((y_t != "O") and (y_p == "O"))
                for y_t, y_p in zip(y_true, y_pred)
                if (y_t == f"B-{entity}" or y_t == f"I-{entity}")
            )
        performance_dict["TN"] = sum(
            (y_t == y_p == "O") for y_t, y_p in zip(y_true, y_pred)
        )
        return performance_dict

    result = f"""
PERFORMANCE:
############
    
{json.dumps(performance_measure(y_true=y_true, y_pred=y_pred), sort_keys=True, indent=4)}

SCORES:
#######

{classification_report(y_true=y_true, y_pred=y_pred, mode="strict", scheme=IOB2)}
"""
    print(result)

    # export result
    if do_output:
        with open(args.output, mode="w", encoding="utf8") as fw:
            fw.write(result)
            logging.info(f"The results were saved to {os.path.abspath(args.output)}.")
