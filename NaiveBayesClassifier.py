import math
import json

# Formula: P(x|S_{i})P(S_{i})>P(x|S_{j})P(S_{j})}, ∀j ≠,𝑖

# comments to train
comments = [
    "Win money now",
    "Free offer just for you",
    "How are you doing today?",
    "This is a great blog post",
    "Congratulations, you won a lottery",
    "Check out my new blog post",
    "Buy cheap products now",
    "Get your free trial",
    "Hope you have a nice day",
    "Hot Meal Hot",
    "Hot Water"
]

# Determines which of those comments are spam or not
labels = [1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0]

# Tokenize the sentence
def tokenize(text):
    return text.lower().split()

# Parse each word
def build_vocabulary(docs):
    vocabulary = {}
    for doc in docs:
        for word in tokenize(doc):
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)
    return vocabulary

#Set frequency of the word
#Using BoW - Bag of Words algorithm
# TODO: Try others
#1. TF-IDF (Term Frequency - Inverse Document Frequency)
#2. Word Emveddings (Word2Vec, GloVe)
#3. Sentence Embeddings (Doc2Vec, Universal Sentence Enconder)
#4. Transformers Algorithms: 
    #4.1. Transformers
    #4.2. Bert
    #4.3 Gpt
def vectorize(doc, vocabulary):
    vector = [0] * len(vocabulary)
    for word in tokenize(doc):
        if word in vocabulary:
            vector[vocabulary[word]] += 1
    return vector

# Build vocabulary and vectorize the training data
vocabulary = build_vocabulary(comments)
X = [vectorize(comment, vocabulary) for comment in comments]
y = labels

class NaiveBayesClassifier:

    def __init__(self) -> None:
        self.classes ={} 
        self.mean = {}
        self.var = {}
        self.priors = {}


    # The Model traniner
    # Calculates, mean, variance and prior
    def fit(self, X, y):

        print("self.classes", y)
        self.classes = list(set(y))
        print("self.classes", self.classes)

        for c in self.classes:
            # Parsing the data
            X_c = []
            for i in range(len(X)):
                if y[i] == c:

                    # print("X[i]", X[i])
                    X_c.append(X[i])

            # calculate the mean 
            self.mean[c] = []
            for feature in zip(*X_c):
                mean_value = sum(feature) / len(feature)
                self.mean[c].append(mean_value)
            
            # calculate the variance 
            self.var[c] = []
            for feature, m in zip(zip(*X_c), self.mean[c]):
                var_value = sum((x - m) ** 2 for x in feature) / len(feature)
                self.var[c].append(var_value)
            
            # calculate the prior 
            self.priors[c] = len(X_c) / len(X)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred
    
    def _predict(self, x):
        posteriors = []
        
        for c in self.classes:
            prior = math.log(self.priors[c])
            class_conditional = sum(math.log(max(self._pdf(x[i], self.mean[c][i], self.var[c][i]), 1e-9)) for i in range(len(x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        
        return self.classes[posteriors.index(max(posteriors))]
    
    def _pdf(self, x, mean, var):
        if var == 0:
            var = 1e-9  
        numerator = math.exp(- (x - mean) ** 2 / (2 * var))
        denominator = math.sqrt(2 * math.pi * var)
        return numerator / denominator

    def save_model(self, filepath):
        model_data = {
            "mean": self.mean,
            "var": self.var,
            "priors": self.priors,
            "classes": self.classes
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f)

    def load_model(self, filepath):
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        self.mean = {int(k): v for k, v in model_data["mean"].items()}
        self.var = {int(k): v for k, v in model_data["var"].items()}
        self.priors = {int(k): float(v) for k, v in model_data["priors"].items()}
        self.classes = [int(c) for c in model_data["classes"]]

model = NaiveBayesClassifier()

model.fit(X, y)

model.save_model("naive_bayes_model.json")

testing_comments = [
    "Win a free ticket",
    "Shit",
    "This blog is very informative",
    "How are you?",
    "Hot Meal",
    "Who are you?",
    "Are you okay?",
    "Free Water"
]

X_new = [vectorize(comment, vocabulary) for comment in testing_comments]

predictions = model.predict(X_new)

# Display the predictions
for i, prediction in enumerate(predictions):
    label = "Spam" if prediction == 1 else "Not Spam"
    print(f"Comment: \"{testing_comments[i]}\" : {label}")