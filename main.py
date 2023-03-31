import allennlp
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import json
import joblib
import spacy
import numpy as np

models = 3
tests = 6

# Create performance metrics datastructures
sb_performance = []
sb_flagged = []
sb_flagged_performance = []
open_performance = []
open_flagged = []
open_flagged_performance = []
logclas_performance = []
logclas_flagged = []
logclas_flagged_performance = []

example_failures = []
example_flagged = []

# Load the saved classifier and vectorizer
loaded_model = joblib.load('models/logistic_classifier.joblib')
loaded_vec = joblib.load('models/vectorizer.joblib')

# Load spaCy model to extract features of sentences
nlp = spacy.load('en_core_web_sm')

# Necessary function for the logistic regression

def custom_predict(input_sentence):
    # Process the input_sentence using spaCy
    doc = nlp(input_sentence)

    # Extract features for each token
    tokens_features = []
    for token in doc:
        features = {
            'WORD': token.text,
            'LEMMA': token.lemma_,
            'BASIC DEP': token.dep_,
            'XPOS': token.tag_,
        }
        tokens_features.append(features)

    # Vectorize the features
    X_input = loaded_vec.transform(tokens_features)

    # Make predictions using the model
    preds = loaded_model.predict(X_input)

       # Create the output dictionary
    words = [token.text for token in doc]
    description = " ".join(words)
    verbs = {'verb': 'generic', 'description': description, 'tags': list(preds)}
    
    output = {
        'verbs': [verbs],
        'words': words,
    }

    # Add reference keys if available
    arg0_indices = np.where(preds == 'ARG0')
    if len(arg0_indices[0]) > 0:
        arg0 = words[arg0_indices[0][0]]
        output['arg0'] = arg0

    arg1_indices = np.where(preds == 'ARG1')
    if len(arg1_indices[0]) > 0:
        arg1 = words[arg1_indices[0][0]]
        output['arg1'] = arg1

    return output

for m in range(models):

    use_custom_model = False

    if m == 0:
        model = "BertSRL"
        model_path = "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
        predictor = Predictor.from_path(model_path)
    if m == 1:
        model = "BiLSTM"
        model_path = "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz"
        predictor = Predictor.from_path(model_path)
    if m == 2:
        model = "Logistic_classifier"
        use_custom_model = True
    for t in range(tests):

        if t == 0:
            dataset = "causative-alternations"
        if t == 1:
            dataset = "dative-alternations"    
        if t == 2:
            dataset = "clefts"
        if t == 3:
            dataset = "statement-questions"
        if t == 4:
            dataset = "active-passive"
        if t == 5:
            dataset = "adverb-detection"

        predictions = []

        # load dataset

        with open(f"datasets/{dataset}.json") as f:
            data = json.load(f)
            for sentence in data["data"]:
                # Join sentence tokens into a string to input into the predictor
                input_sentence = ' '.join(sentence["tokens"])
                # Remove the space between the last word and period
                input_sentence = input_sentence[:-2] + '.'
                # Predict the labels of the sentence
                if use_custom_model:
                    prediction = custom_predict(input_sentence)
                else:
                    prediction = predictor.predict(sentence=input_sentence)
                # Add the prediction to the list of predictions
                predictions.append(prediction)
             
                if t == 0:
                    arg0 = None
                    arg1 = None

                if data["data"].index(sentence) % 2 == 0 and t != 0:
                    # Grab the word that is the ARG0 from the data object if there is an ARG0
                    if "ARG0" in sentence["BIO"]:
                        arg0 = sentence["tokens"][sentence["BIO"].index("ARG0")]
                    # Grab the word that is the ARG1 from the data object if there is an ARG1
                    if "ARG1" in sentence["BIO"]:
                        arg1 = sentence["tokens"][sentence["BIO"].index("ARG1")]

                elif t == 0:
                    # Grab the word that is the ARG0 from the data object if there is an ARG0
                    if "ARG0" in sentence["BIO"]:
                        arg0 = sentence["tokens"][sentence["BIO"].index("ARG0")]
                    # Grab the word that is the ARG1 from the data object if there is an ARG1
                    if "ARG1" in sentence["BIO"]:
                        arg1 = sentence["tokens"][sentence["BIO"].index("ARG1")]

                # add arg0 and arg1 to the prediction as reference key
                if arg0:
                    prediction["arg0"] = arg0
                if arg1:
                    prediction["arg1"] = arg1
                if t == 0:
                    arg0 = None
                    arg1 = None


        cases = len(predictions) / 2
        passes = 0
        fails = 0
        flagged = []
        failures = []

        if t == 1 or t == 2 or t == 3 or t == 4:
            for i in range(0, len(predictions), 2):
                pair_correct = True
                for j in range(2):
                    if i + j < len(predictions):
                        prediction = predictions[i + j]
                        arg0_correct = False
                        arg1_correct = False
                        it_correct = True
                        who_correct = True
                        for verb in prediction["verbs"]:
                            # Exclude the was/were verbs
                            if verb["verb"] != "was" and verb["verb"] != "were":
                                len_sent = len(prediction["words"])

                                # Check which label the word "it" has
                                if "It" in prediction["words"] and t == 1:
                                    it_index = prediction["words"].index("It")
                                    it_label = verb["tags"][it_index]
                                else:
                                    it_label = None

                                # Check which label the word "who" has
                                if "who" in prediction["words"] and t == 1:
                                    who_index = prediction["words"].index("who")
                                    who_label = verb["tags"][who_index]
                                else:
                                    who_label = None

                                for k in range(len_sent):
                                    # Check if the word is the labeled ARG0
                                    # if prediction["arg0"] does not exist, stop this iteration
                                    if prediction["words"][k].lower() == prediction["arg0"].lower():
                                        # Check if the word is the predicted ARG0
                                        if "ARG0" in verb["tags"][k]:
                                            arg0_correct = True
                                            arg0_index = k
                                        else:
                                            continue
                                    # Do the same for ARG1
                                    elif prediction["words"][k].lower() == prediction["arg1"].lower():
                                        if "ARG1" in verb["tags"][k]:
                                            arg1_correct = True
                                            arg1_index = k
                                        else:
                                            continue

                                # Check if there are more ARG1s which are not "a" or "an" or "the" and which are not located at the arg1_index
                                if any("ARG1" in verb["tags"][idx] for idx in range(len(verb["tags"])) if idx != arg1_index and prediction["words"][idx].lower() not in ["a", "an", "the", "her", "his", "at"]):
                                    arg1_correct = False

                                # Do the same for ARG0s
                                if any("ARG0" in verb["tags"][idx] for idx in range(len(verb["tags"])) if idx != arg0_index and prediction["words"][idx].lower() not in ["a", "an", "the", "by", "who"]):
                                    arg0_correct = False

                                if (who_label != 'O' and who_label != 'B-R-ARG0') and who_label:
                                    who_correct = False
                                
                                if it_label != 'O' and it_label:
                                    it_correct = False

                        if arg0_correct == False or arg1_correct == False or who_correct == False or it_correct == False:
                            pair_correct = False
                            if j == 0:  # First sentence of the pair is incorrect
                                flagged.append(prediction)
                            else:  # Second sentence of the pair is incorrect
                                failures.append(prediction)
                                fails += 1
                            break

                if pair_correct:
                    passes += 1

        if t == 0 or t == 5:

            for i in range(0, len(predictions), 2):
                pair_correct = True
                for j in range(2):
                    if i + j < len(predictions):
                        prediction = predictions[i + j]
                        arg0_correct = False
                        arg1_correct = False
                        it_correct = True
                        who_correct = True
                        for verb in prediction["verbs"]:
                            if t == 0:
                                # Exclude the was/were verbs
                                if verb["verb"] != "helped" and verb["verb"] != "made" and verb["verb"] != "let":
                                    len_sent = len(prediction["words"])

                                    # Check which label the word "it" has
                                    if "It" in prediction["words"] and t == 1:
                                        it_index = prediction["words"].index("It")
                                        it_label = verb["tags"][it_index]
                                    else:
                                        it_label = None

                                    # Check which label the word "who" has
                                    if "who" in prediction["words"] and t == 1:
                                        who_index = prediction["words"].index("who")
                                        who_label = verb["tags"][who_index]
                                    else:
                                        who_label = None

                                    for k in range(len_sent):
                                        # Get the values for arg0 and arg1, or set them to None if they don't exist
                                        arg0_value = prediction.get("arg0", None)
                                        arg1_value = prediction.get("arg1", None)


                                        # If there's no value for arg0 and theres no ARG0 in verb[tags], set the corresponding correct flag to True
                                        if arg0_value is None and not any("ARG0" in tag for tag in verb["tags"]):
                                            arg0_correct = True
                                            arg0_index = len(verb["tags"])
                                        elif arg0_value is not None and "ARG0" in verb["tags"]:
                                            arg0_index = len(verb["tags"])

                                        # If there's no value for arg1 and theres no ARG1 in verb[tags], set the corresponding correct flag to True
                                        if arg1_value is None and not any("ARG1" in tag for tag in verb["tags"]):
                                            arg1_correct = True
                                            arg1_index = len(verb["tags"])
                                        elif arg1_value is not None and "ARG1" in verb["tags"]:
                                            arg1_index = len(verb["tags"])
                                        

                                        # Check if the word is the labeled ARG0
                                        # if prediction["arg0"] does not exist, stop this iteration
                                        # Check if the word is the labeled ARG0
                                        if arg0_value is not None and prediction["words"][k].lower() == arg0_value.lower():
                                            # Check if the word is the predicted ARG0
                                            if "ARG0" in verb["tags"][k]:
                                                arg0_correct = True
                                                arg0_index = k
                                            else:
                                                continue
                                        # Do the same for ARG1
                                        elif arg1_value is not None and prediction["words"][k].lower() == arg1_value.lower():
                                            if "ARG1" in verb["tags"][k]:
                                                arg1_correct = True
                                                arg1_index = k
                                            else:
                                                continue

                                    # Check if there are more ARG1s which are not "a" or "an" or "the" and which are not located at the arg1_index
                                    if any("ARG1" in verb["tags"][idx] for idx in range(len(verb["tags"])) if idx != arg1_index and prediction["words"][idx].lower() not in ["a", "an", "the", "her", "his", "at"]):
                                        arg1_correct = False

                                    # Do the same for ARG0s
                                    if any("ARG0" in verb["tags"][idx] for idx in range(len(verb["tags"])) if idx != arg0_index and prediction["words"][idx].lower() not in ["a", "an", "the", "by"]):
                                        arg0_correct = False

                                    if (who_label != 'O' and who_label != 'B-R-ARG0') and who_label:
                                        who_correct = False
                                    
                                    if it_label != 'O' and it_label:
                                        it_correct = False

                                if arg0_correct == False or arg1_correct == False or who_correct == False or it_correct == False:
                                    pair_correct = False
                                    if j == 0:  # First sentence of the pair is incorrect
                                        flagged.append(prediction)
                                    else:  # Second sentence of the pair is incorrect
                                        failures.append(prediction)
                                        fails += 1
                                    break
                            
                            elif t == 5 and j == 0:  # Only compare when it's the first prediction in the pair
                                if i + 1 < len(predictions):
                                    if m == 2:
                                        v_index_next = predictions[i + 1]["verbs"][0]["tags"].index("V")
                                    else:
                                        v_index_next = predictions[i + 1]["verbs"][0]["tags"].index("B-V")
                                    next_tags_filtered = [tag for idx, tag in enumerate(predictions[i + 1]["verbs"][0]["tags"]) if idx != (v_index_next - 1)]

                                    if verb["tags"] != next_tags_filtered:
                                        pair_correct = False
                                        failures.append(prediction)
                                        failures.append(predictions[i + 1])
                                        fails += 1
                                        break

                        if not pair_correct:
                            break

                if pair_correct:
                    passes += 1

        print("=====================================")
        print("performance of", model, "on", dataset, "tests")
        print(f"{passes} passes out of {cases}")
        # print the percentages of failures without flagged
        failure_rate = round((fails / cases) * 100, 2)
        print(f"{failure_rate}% failures")

        if m == 0:
            sb_performance.append(failure_rate)
            sb_flagged.append(len(flagged))
        if m == 1:
            open_performance.append(failure_rate)
            open_flagged.append(len(flagged))
        if m == 2:
            logclas_performance.append(failure_rate)
            logclas_flagged.append(len(flagged))

        # print the percentage of failures with flagged
        failure_rate_flagged = round((fails + len(flagged)) / (cases) * 100, 2)
        print(f"{failure_rate_flagged}% failures with flagged cases")

        if m == 0:
            sb_flagged_performance.append(failure_rate_flagged)
        if m == 1:
            open_flagged_performance.append(failure_rate_flagged)
        if m == 2:
            logclas_flagged_performance.append(failure_rate_flagged)

        # Write the predictions to a json file
        with open(f"output/{model}_{dataset}.json", "w") as f:
            # remove arg0 and arg1 keys from predictions
            for prediction in predictions:
                if "arg0" in prediction:
                    del prediction["arg0"]
                if "arg1" in prediction:
                    del prediction["arg1"]
            json.dump(predictions, f)

datasets = ["causative-alternations", "dative-alternations", "clefts", "statements-questions", "active-passive", "adverb detection"]

print("\n=====================================")
print("SRL BERT performance:")
print("=====================================\n")
for i in range(tests):
    print(f"{datasets[i]} failure rate: {sb_performance[i]}%")
    print(f"{datasets[i]} flagged failure rate: {sb_flagged_performance[i]}%")

print("\n=====================================")
print("SRL BiLSTM performance:")
print("=====================================\n")
for i in range(tests):
    print(f"{datasets[i]} failure rate: {open_performance[i]}%")
    print(f"{datasets[i]} flagged failure rate: {open_flagged_performance[i]}%")

print("\n=====================================")
print("LogClassifier performance:")
print("Failure Rate is skewed*")
print("=====================================\n")
for i in range(tests):
    print(f"{datasets[i]} failure rate: {logclas_performance[i]}%")
    print(f"{datasets[i]} flagged failure rate: {logclas_flagged_performance[i]}%")