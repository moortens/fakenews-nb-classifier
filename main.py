import naivebayes
import argparse

naive = naivebayes.NaiveBayesClassifier()

def load():
  if not naive.load_data("model.json"):
    print("Train your dataset")
    print()
    argParser.print_help()
    exit()

def train():
  naive.train()
  naive.save(file="model.json")

  return

def classify(text, alpha):
  naive.classify(text, alpha=alpha)

if __name__ == '__main__':
  argParser = argparse.ArgumentParser(description="Process the fake news dataset")
  argParser.add_argument('--train', action='store_true', help="trains the dataset", required=False)
  argParser.add_argument('--classify', type=str, help="classifies text", required=False)
  argParser.add_argument('--alpha', type=float, help="set the alpha, value between 0.1 and 1.0", default=1.0)
  args = argParser.parse_args()

  if args.train:
    train()
  else:
    load()

    if args.classify:
      classify(args.classify, args.alpha)