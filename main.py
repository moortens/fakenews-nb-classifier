import naivebayes
import argparse

naive = naivebayes.NaiveBayesClassifier()

def load(model):
  if not naive.load_data(model):
    print("Train your dataset")
    print()
    argParser.print_help()
    exit()

def train(model):
  naive.train()
  naive.save(model)

  return

def classify(text, alpha):
  klass = naive.classify(text, alpha=alpha)
  if klass is 1:
    print("Text classified as unreliable")
  else:
    print("Text classified as reliable")

def test():
  print("Test accuracy:", str(naive.test()*100) + "%")

if __name__ == '__main__':
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