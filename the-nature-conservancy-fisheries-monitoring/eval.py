from os import path

from src.util import *


def evaluate_predictions(Predictor):
    test_list = list_files(labeled_train=False, labeled_test=True)
    predictor = Predictor()

    num_guesses = 0.0
    num_correct = 0.0

    for example in test_list:
        image_name, ground_truth = example.split(' ')
        image_path = path.join(DATA_PATH, image_name)
        prediction = predictor.predict(image_path)
        
        if prediction == ground_truth:
            num_correct += 1
        num_guesses += 1

        print_stats(prediction, ground_truth, num_guesses, num_correct)
    
    print "------\nACCURACY: {}\n------".format(num_correct / num_guesses)

def ouput_validation_likelihoods(Predictor):
    test_list = list_files(False, False, True)
    predictor = Predictor()

    for image_name in test_list:
        image_path = path.join(DATA_PATH, image_name)
        _ = predictor.predict(image_path, for_submission=True)

def print_stats(guess, ground_truth, num_guesses, num_correct):
    print "{}\tGuess: {}\t\tActual: {}".format(
        int(num_guesses),
        guess,
        ground_truth
    )
            
    if (num_guesses % 100 == 0):
        print '-----------------------------------------------------'
        print 'Accuracy through {} predictions: {}'.format(
            int(num_guesses),
            num_correct / num_guesses
        )
        print '-----------------------------------------------------'


if __name__ == "__main__":
    import importlib
    from argparse import ArgumentParser

    parser = ArgumentParser("Evaluate a model on the validation set")
    parser.add_argument(
        "model_dir",
        help="The directory the model is stored in (e.g. src)."
    )
    parser.add_argument(
        "--submission", 
        action="store_true",
        help="Specify if output should be formatted for competition submission"
    )
    args = parser.parse_args()

    try:
        run = importlib.import_module(args.model_dir+'.run')
    except:
        print "Failed to load 'run.py' in '{}'".format(args.model_dir)
        exit(1)

    if args.submission:
        ouput_validation_likelihoods(run.Predictor)
    else:
        evaluate_predictions(run.Predictor)
        
