import naivebayes
import argparse

naive = naivebayes.NaiveBayesClassifier()

def load(model):
  # loads the model from the training of the data
  # contains (reliable, unreliable, prior_reliable_probability, prior_unreliable_probability)
  if not naive.load_data(model):
    print("Train your dataset")
    print()
    argParser.print_help()
    exit()

def train(model):
  # trains and saves the model of the training.
  naive.train()
  naive.save(model)

  return

def classify(text, alpha):
  # classifies the text and gives a feedback of the result (reliable or unreliable)
  klass = naive.classify(text, alpha=alpha)
  if klass is 1:
    print("Text classified as unreliable")
  else:
    print("Text classified as reliable")

def test():
  # prints the accuracy of the classification of real and fake articles
  print("Test accuracy:", str(naive.test()*100) + "%")

if __name__ == '__main__':
  # Sets up the argument parser for use of arguments in the command line
  argParser = argparse.ArgumentParser(description="Process the fake news dataset")
  argParser.add_argument('--train', action='store_true', help="trains the dataset", required=False)
  argParser.add_argument('--model', type=str, help="path to classified model file", required=False, default="model.json")
  argParser.add_argument('--classify', type=str, help="classifies text", required=False)
  argParser.add_argument('--test', action='store_true', help="tests the model", required=False)
  argParser.add_argument('--alpha', type=float, help="set the alpha, value between 0.1 and 1.0", default=1.0)
  args = argParser.parse_args()

  if args.train:
    train(args.model)
  else:
    load(args.model)

    if args.classify:
      classify(args.classify, args.alpha)
    if args.test:
      test()


'''
To find a helpful overview of how the code can be used, write "python main.py --help" in the command line.
Here you will be presented by a list of optional arguments:
--------------------------------------------------------------------
--help                    show this help message and exit
--train                   trains the dataset
--model MODEL             path to classified model file
--classify CLASSIFY       classifies text
--test                    tests the model
--alpha ALPHA             set the alpha, value between 0.1 and 1.0
--------------------------------------------------------------------
You can start with "--train" argument to train the data, and then can test by using the "--test" argument.
If you want to test a news article from the command line, then use the "--classify" argument
followed by the article. 
'''