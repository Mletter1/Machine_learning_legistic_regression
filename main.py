#!/usr/bin/env python
# coding=utf-8
__author__ = 'Matthew Letter'
import sys
import os
import copy
import traceback
import optparse
import time
import cPickle as pickle
import scipy
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
import numpy

debug = False


def get_doc():
    doc = """
    SYNOPSIS

        main [-h,--help] [-v,--verbose] [--version]

    DESCRIPTION

        This is the main entry point for the logistical regression algorithm

    EXAMPLES
        print out help message:
            python main.py -h
        Run logistic regression algorithm with verbose output
            python main.py -v
        Run process data then logistic regression algorithm with verbose output assumes starting directory = "."
            python main.py -p
        Run process data then logistic regression algorithm with verbose output assumes starting directory = arg[0]
            python main.py -p <path to dir with opihi.cs.uvic.ca>
        Run logistic regression algorithm with pre-processed data assumes starting directory = "." for pickle data
            python main.py

    EXIT STATUS

        0 no issues
        1 unknown error
        2 improper params

    AUTHOR

        Name Matthew Letter mletter1@unm.edu

    LICENSE

        This script is in the public domain, free from copyrights or restrictions.

    VERSION

        v0.1
    """
    return doc


def normalize_min_max(input_data):
    """
    Normalization by Scaling Between 0 and 1
    Assume that there are n rows with seven variables, A, B, C, D, E, F and G, in the data.
    We use variable E as an example in the calculations below.
    The remaining variables in the rows are normalized in the same way.

    The normalized value of ei for variable E in the ith row is calculated as:

    normalized(ei)= ei-Emin/(Emax-Emin)

    Emin = the minimum value for variable E

    Emax = the maximum value for variable E

    If Emax is equal to Emin then Normalized (ei) is set to 0.5.

    Returns:
            normalized_data: the normalized data.
    """
    normalized_data = copy.deepcopy(input_data)
    for i in range(len(normalized_data[0][0])):  # point on vector
        maximum = float('-inf')
        minimum = float('inf')
        for j in range(len(normalized_data)):  # class of map
            for k in normalized_data[j]:  # vector from class map
                temp_value = k[i]
                if temp_value >= maximum:
                    maximum = temp_value
                elif temp_value <= minimum:
                    minimum = temp_value
        for j in range(len(normalized_data)):  # class of map
            for k in normalized_data[j]:  # vector from class map
                k[i] = (k[i] - minimum) / (maximum - minimum)
    return normalized_data


def get_dir_and_label_map(directory_name):
    """
    iterate the directory and gets its wav files.
    http://www.tutorialspoint.com/python/os_walk.htm]
    ................................................

    :parameter: directory_name: name of the directory you wish to start from

    :return: label_map: {} where key[number] = genre
            music_dictionary: where {} # key[genre] = [list paths to wave files]
    """
    label_map = {}  # key[number] = genre
    reverse_label_map = {}
    music_dictionary = {}  # key[number] = [list paths to wave files]
    label = 0
    for subdir, dirs, files in os.walk(directory_name, topdown=True, onerror=None, followlinks=False):
        for f in files:
            if os.path.splitext(f)[1] == ".wav":  # look for .wavs
                genre = subdir.split("/")[4]  # get genre name
                if genre in reverse_label_map:
                    music_dictionary[reverse_label_map[genre]].append(subdir + "/" + f)
                else:
                    if debug:
                        print(genre)
                    label_map[label] = genre
                    reverse_label_map[genre] = label
                    label += 1
                    music_dictionary[reverse_label_map[genre]] = [subdir + "/" + f]
    if debug:
        print
        print "music dic: ", music_dictionary
        print
        print "label map: ", label_map
        print
    return label_map, music_dictionary


def process_fft_data(directory_name="."):
    """

    goes through ,wav files and extracts the first 1000 point fft data
    and puts it in a dictionary where key[class] = [list of fft vector's of that class]
    ....................................................................

    the directory needs to be run from ./ where this ./opihi.cs.uvic.ca
    directory is in ./  or else it wont find genre names properly

    main goal is to create a map{} of dimensions

                        feature vectors = [] for each class
                --------------------------------
                |
                |
      classes   |
                |
                |
                |


     ....................................................................
     saves out a picle file of a list where [label_map, music_dictionary,class_to_fft_map]
            label_map: {} where key[number] = genre
            music_dictionary: where {} # key[genre] = [list paths to wave files]
            class_to_fft_map:
                                            feature vectors = [] for each class
                        --------------------------------
                        |
                        |
              classes   |
                        |
                        |
                        |

    this file is name data-fft.p

    :param: directory_name:
    """
    label_map, music_dictionary = get_dir_and_label_map(directory_name)
    class_to_fft_map = {genre: [] for genre in label_map}
    class_to_gfcc_map = {genre: [] for genre in label_map}
    if debug:
        print "initialize fft class map: ", class_to_fft_map
        print

    # load out data
    for genre_class_value in music_dictionary:
        for wav_file_path in music_dictionary[genre_class_value]:
            print wav_file_path
            # get its fft information.
            sample_rate, x = scipy.io.wavfile.read(wav_file_path)
            #get mfcc
            ceps, mspec, spec = mfcc(x)
            ceps_vector = ceps.shape[0]
            mfcc_feature = numpy.mean(ceps[int(ceps_vector * 1 / 10):int(ceps_vector * 9 / 10)], axis=0)
            class_to_gfcc_map[genre_class_value].append(mfcc_feature)

            # Use the 1000 first FFT components as features
            fft_feature = abs(scipy.fft(x)[:1000])
            class_to_fft_map[genre_class_value].append(fft_feature)
    if debug:
        print
        print class_to_fft_map.keys()
        print
        print "pickle dump"
        print
    print
    pickle_data = [label_map, music_dictionary, class_to_fft_map, class_to_gfcc_map]
    pickle.dump(pickle_data, open("data.p", "wb"))


def load_pickle(file_path="data.p"):
    """
    load pickle data.p file
    ................................................

    :parameter: directory_name: name of the directory you wish to start from

    :return: label_map: {} where key[number] = genre
             music_dictionary: where {} # key[genre] = [list paths to wave files]
             class_to_fft_map:
                                            feature vectors = [] for each class
                        --------------------------------
                        |
                        |
              classes   |
                        |
                        |
                        |
             class_to_gfcc_map:
                                      gfcc feature vectors = [] for each class
                        --------------------------------
                        |
                        |
              classes   |
                        |
                        |
                        |

    """
    if debug:
        print "pickle load"
    # load fft pickle data
    data_load = pickle.load(open(file_path, "rb"))
    if debug:
        for element in data_load:
            print
            print "length of each pickle load: ", len(element)
    # get data values
    return data_load[0], data_load[1], data_load[2], data_load[3]


def get_testing_training_samples(class_to_vectors_map, cross_validation_const):
    """
    creates a dictionary of training and validation data from a set of data
    :parameter

                class_to_vectors_map: classes mapped to all its vectors
                cross_validation_const: number of cross validation data sets you need
    :returns
                data_dictionary: {cross_validation_round: [] for cross_validation_round in
                range(cross_validation_const)}
                where data_dictionary[0] = {{traning_data[class]=[vectors]}, {testing_data[class]=[vectors]}}

    """
    data_dictionary = {cross_validation_round: [] for cross_validation_round in range(cross_validation_const)}
    for k in range(cross_validation_const):
        training_dict = {i: [] for i in range(len(class_to_vectors_map))}
        testing_dict = {i: [] for i in range(len(class_to_vectors_map))}
        for genre in class_to_vectors_map:
            for vector_index in range(len(class_to_vectors_map[genre])):
                if (vector_index - k) % 10 != 0:
                    training_dict[genre].append(class_to_vectors_map[genre][vector_index])
                else:
                    testing_dict[genre].append(class_to_vectors_map[genre][vector_index])
        data_dictionary[k] = [copy.deepcopy(training_dict), copy.deepcopy(testing_dict)]
    if debug:
        length = 0
        for vector_index in range(len(data_dictionary[0][0])):
            length += len(data_dictionary[0][0][vector_index]) + len(
                data_dictionary[0][1][vector_index])  # trs len plus tst len
        print "training plus testing list len: ", length
        print "# of sets: ", len(data_dictionary)
    return data_dictionary


def get_delta_x_weight_matrix(data, n_plus_1=1001, number_of_classes=6):
    """
    runs through the data and create three matrices used in logistic regression matrix calculation
    :parameter data: {class: [vectors of data]}
    :parameter n_plus_1: vector +1
    :parameter number_of_classes: number of classes
    :return:
               delta_matrix: Δ, a t×m matrix where Δji=δ(Yi=yj)
                    (using the delta equation as found in equation (29) in the Mitchell chapter)
               X_matrix: X, an m×(n+1) matrix of examples, where ∀i,Xi0=1,
                    and Xi1 through Xin are the attributes for example i
               weight_matrix: W, a t×(n+1) matrix of weights
    """
    number_of_samples = sum([len(data[v]) for v in data])  # number of cols

    weight_matrix = numpy.zeros((number_of_classes, n_plus_1))

    delta_matrix = numpy.zeros((number_of_classes, number_of_samples))
    x_matrix = numpy.zeros((number_of_samples, n_plus_1))

    sample_number = 0
    for genre in data:
        for vector in data[genre]:
            vector_point_index = 1  # start by +1 as index Xi0 = 1
            x_matrix[sample_number][0] = 1.0
            delta_matrix[genre][sample_number] = 1.0
            for point in vector:
                x_matrix[sample_number][vector_point_index] = point
                vector_point_index += 1
            sample_number += 1
    if debug:
        print "Delta matrix dimensions row length:", len(delta_matrix), " col length:", len(
            delta_matrix[0]), "\n", delta_matrix, "\n"
        print "X matrix dimensions row length:", len(x_matrix), " col length:", len(x_matrix[0]), "\n"
        print x_matrix
    return delta_matrix, x_matrix, weight_matrix


def get_class(weight_matrix, sample, n_plus_1):
    """
    return the class value based off the inputs
    :param weight_matrix: trained weight matrix
    :param sample: vector we are testing
    :param n_plus_1: vector + 1 size
    :return:
        class of sample
    """
    sample_plus_1 = numpy.zeros(n_plus_1)
    sample_plus_1[0] = 1
    index = 1
    for point in sample:
        sample_plus_1[index] = point
        index += 1
    result_table = numpy.exp(numpy.dot(weight_matrix, sample_plus_1.T)).tolist()
    return result_table.index(max(result_table))


def test(weight_matrix, number_of_classes, testing_data, n_plus_1, confusion_matrix):
    """
    make a confusion matrix and obtain the accuracy of a run of logistic regression
    :parameter
        weight_matrix: matrix of weights from training
        number_of_classes: number of classes we have
        testing_data: data used for testing
        n_plus_1: vecotr length of data plus 1
        confusion_matrix:
    :returns
        accuracy: accuracy of run
        confusion_matrix: the confusion matrix
    """
    number_of_samples = sum([len(testing_data[v]) for v in testing_data])  # number of cols
    if confusion_matrix is None:
        confusion_matrix = numpy.zeros((number_of_classes, number_of_classes))
    correct = 0.0
    if debug:
        print "\n number of test samples:", number_of_samples, "initial confusion matrix: \n", confusion_matrix
    for genre in testing_data:
        for sample in testing_data[genre]:
            predicted_class = get_class(weight_matrix, sample, n_plus_1)
            if predicted_class == genre:
                correct += 1
            confusion_matrix[genre][predicted_class] += 1

    print "\n correct: ", ((correct / number_of_samples) * 100), "%"
    print confusion_matrix
    return (correct / number_of_samples) * 100, confusion_matrix


def train_test_network(label_map, class_to_fft_map, smart_weights=False, cross_x=10, iterations=2000,
                       eta_not=0.02, lam=0.0001):
    """
    from piazza post:

    Given: m, the number of examples
           t, the number of unique classifications an example can have
           n, the number of attributes each example has
           η, a learning rate
           λ, a penalty term
           Δ, a t×m matrix where Δji=δ(Yi=yj) (using the delta equation as found in equation (29) in the Mitchell chapter)
           X, an m×(n+1) matrix of examples, where ∀i,Xi0=1, and Xi1 through Xin are the attributes for example i
           Y, an m×1 vector of true classifications for each example
           W, a t×(n+1) matrix of weights

           P(Y|W,X)∼exp(WXT), a t×m matrix of probability values.
           To follow the format of equations (27) and (28) in the text,
           fill the last row with all 1's, and then normalize each column to sum
           to one by dividing the each value in the column by the sum of the column.

    Then the update step for the logistic regression is

    Wt+1=Wt+η((Δ−P(Y|W,X))X−λWt)
    """
    vector_size = len(class_to_fft_map[0][0])
    n_plus_1 = vector_size + 1
    number_of_classes = len(label_map)
    test_train_data = get_testing_training_samples(class_to_fft_map, cross_x)
    cross_x_accumulator = list()
    if debug:
        print "\n", "vector + 1 =", n_plus_1, " number of classes =", number_of_classes
    weight_matrix = None
    if smart_weights:
        training_data = normalize_min_max(test_train_data[0][0])
        testing_data = normalize_min_max(test_train_data[0][1])
        delta_matrix, x_matrix, weight_matrix = get_delta_x_weight_matrix(training_data, n_plus_1, number_of_classes)
    # for each cross validation
    print "\n strating traing with ", cross_x, " validation runs \n"
    for run_number in range(len(test_train_data)):
        training_data = normalize_min_max(test_train_data[run_number][0])
        testing_data = normalize_min_max(test_train_data[run_number][1])
        if smart_weights:
            delta_matrix, x_matrix, weight_matrix_o = get_delta_x_weight_matrix(training_data, n_plus_1,
                                                                                number_of_classes)
        else:
            delta_matrix, x_matrix, weight_matrix = get_delta_x_weight_matrix(training_data, n_plus_1,
                                                                              number_of_classes)
        print
        print "run number", run_number
        for epoch in range(iterations):
            eta = eta_not / (10 + epoch / iterations)
            exponent_matrix = numpy.exp(numpy.dot(weight_matrix, x_matrix.T))
            exponent_matrix /= numpy.sum(exponent_matrix, axis=0)
            delta_minus_exp_matrix = delta_matrix - exponent_matrix
            delta_minus_exp_matrix_times_x = numpy.dot(delta_minus_exp_matrix, x_matrix)
            lambda_weight_matrix = lam * weight_matrix
            delta_minus_exp_times_x_minus_lam_weight_matrix = delta_minus_exp_matrix_times_x - lambda_weight_matrix
            delta_minus_exp_times_x_minus_lam_weight_matrix *= eta
            weight_matrix += delta_minus_exp_times_x_minus_lam_weight_matrix
        if run_number is 0:
            accuracy, confusion_matrix = test(weight_matrix, number_of_classes, testing_data, n_plus_1, None)
        else:
            accuracy, confusion_matrix = test(weight_matrix, number_of_classes, testing_data, n_plus_1, confusion_matrix)
        cross_x_accumulator.append(accuracy)
    print cross_x_accumulator
    print sum(cross_x_accumulator) / len(cross_x_accumulator)
    return weight_matrix


def class_to_fft_map_top_twenty(class_to_fft_map, label_map):
    """
    creates a map of top number of features which maximize the
    standard deviation

    :parameter
        class_to_fft_map: class  to fft map
        label_map: map of class to genre name
    :returns
        top_twenty_map: map of top 20 features
    """
    top_twenty_map = {genre: [] for genre in label_map}
    data = copy.deepcopy(class_to_fft_map)
    for genre in data:
        col_std_list = list()
        for vector_index in range(len(data[0][0])):
            col_point_list = list()
            for vector in data[genre]:
                col_point_list.append(vector[vector_index])
            col_std_list.append(numpy.std(col_point_list))
        sorted_std_list = sorted(col_std_list)
        for vector in range(len(data[genre])):
            new_vector = list()
            for num in range(20):
                new_vector.append(data[genre][vector][col_std_list.index(sorted_std_list[len(data[0][0])-1-num])])
            top_twenty_map[genre].append(new_vector)
    return top_twenty_map


def get_weight_top_twenty(weight_matrix, class_to_fft_map, label_map):
    """
    creates a map of top number of features which maximize the
    standard deviation

    :parameter
        class_to_fft_map: class  to fft map
        label_map: map of class to genre name
    :returns
        top_twenty_map: map of top 20 features
    """
    top_twenty_map = {genre: [] for genre in label_map}
    data = copy.deepcopy(class_to_fft_map)
    weight_matrix = numpy.abs(weight_matrix)
    for genre in data:
        unsorted_vec = weight_matrix[genre]
        sorted_vec = numpy.argsort(unsorted_vec)
        for vector in data[genre]:
            temp_vec = list()
            for i in range(20):
                #index = sorted_vec[len(sorted_vec)-1-i]
                index = sorted_vec[i]
                #print vector
                #print data[genre][vector][index-1]
                temp_vec.append(vector[index-1])
            top_twenty_map[genre].append(temp_vec)
    return top_twenty_map


def run():
    """
    this method loads pre-processed pickle data and feeds it
    into the multinominal network
    """
    print "\n" + "***********************************"
    print 'Runnning fft 10x cross validation with 2000 iteration per validation\n'
    # load pickle data
    label_map, music_dictionary, class_to_fft_map, class_to_gfcc_map = load_pickle("data.p")

    # fft run
    weight_matrix = train_test_network(label_map, class_to_fft_map)

    #weight based top 20
    print '\n Runnning fft top twenty with weight matrix, 10x cross validation with 2000 iteration per validation\n'
    top_twenty_weight = get_weight_top_twenty(weight_matrix, class_to_fft_map, label_map)
    train_test_network(label_map, top_twenty_weight)

    # top 20 run
    print '\n Runnning fft top twenty with standard deviation, 10x cross validation with 2000 iteration per validation\n'
    top_twenty_map = class_to_fft_map_top_twenty(class_to_fft_map, label_map)
    train_test_network(label_map, top_twenty_map)

    # gfcc run
    print '\n Runnning gfcc 10x cross validation with 2000 iteration per validation\n'
    train_test_network(label_map, class_to_gfcc_map)

    print label_map
    print "***********************************\n"


if __name__ == '__main__':
    """
    determine running params
    """
    global options, args
    try:
        start_time = time.time()
        parser = optparse.OptionParser(formatter=optparse.TitledHelpFormatter(), usage=get_doc(),
                                       version='%prog 0.1')
        parser.add_option('-v', '--verbose', action='store_true', default=False, help='verbose output')
        parser.add_option('-p', '--process', action='store_true', default=False, help='process wave files')
        # get the options and args
        (options, args) = parser.parse_args()
        # determine what to do with the options supplied by the user
        if options.verbose:
            debug = True
        print "options ", options
        print "args", args
        print "start time: " + time.asctime()
        if options.process:
            if args:
                starting_dir = args[0]
                process_fft_data(starting_dir)  # TOTAL TIME IN MINUTES: 15.9127927979
            else:
                process_fft_data()
        run()

        print "finish time: " + time.asctime()
        print 'TOTAL TIME IN MINUTES:',
        print (time.time() - start_time) / 60.0
        # smooth exit if no exceptions are thrown
        sys.exit(0)

    except KeyboardInterrupt, e:  # Ctrl-C
        raise e
    except SystemExit, e:  # sys.exit()
        raise e
    except Exception, e:  # unknown exception
        print 'ERROR, UNEXPECTED EXCEPTION'
        print str(e)
        traceback.print_exc()
        os._exit(1)