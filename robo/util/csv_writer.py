#!/usr/bin/env python
import csv
import os
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def csv_reader(filepath):
    with open(filepath, 'rb') as csvfile:  # opens the csv file
        result = {}
        # reads from the DICT Reader with that openend file
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            for column, value in row.iteritems():
                result.setdefault(column, []).append(value)
    return result  # returns the CSV file as a Dict


def result_dict(filepath):
    dictionary_list = []
    final_dictionary = {}

    for k in filepath:
        dictionary = {}  # The main dictionary that will contain all the data
        sub_dictionary = {}  # temporary helper dictionary
        directory_tree = [x for x in os.walk(k)]

        for i in directory_tree[0][
                1]:  # add the first descendandts of the main file path in the dictionary first
            dictionary[i] = " "

        for i in dictionary:
            sub_dictionary = {}
            a = os.path.join(k, i)  # Go down one level in the main file path (a have the new path)
            sub_directory_tree = [x for x in os.walk(a)]
            for j in sub_directory_tree[0][1]:
                b = os.path.join(a, j)
                sub_dictionary[j] = b
                csv_file_name = [y[2] for y in os.walk(b)]
                sub_dictionary[j] = os.path.join(b, csv_file_name[0][0])
            dictionary[i] = sub_dictionary
        # Now save the CSV file as a dict in our dictionary
        for i in dictionary:
            for j in dictionary[i]:
                csv_file_path = dictionary[i][j]
                dictionary[i][j] = csv_reader(csv_file_path)
        dictionary_list.append(dictionary)
    # Merge all the dictionaries in the list dictionary_list
    for i in dictionary_list:
        final_dictionary = dict(final_dictionary.items() + i.items())
    # print dictionary['optimizer_1']['1_run_1']['incumbent'][0]
    return final_dictionary


def get_all_columns(dictionary, optimizer_name, column_name):
    # IT should return to you a list which contains the enteries from all the
    # csv files of that column.
    for i in dictionary[optimizer_name]:
        logger.info(dictionary[optimizer_name][i][column_name])
