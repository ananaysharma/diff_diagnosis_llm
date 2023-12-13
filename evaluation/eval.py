import json
from sklearn.metrics import accuracy_score, f1_score
import sys

# Check if the required arguments are provided
if len(sys.argv) != 3:
    print(
        "Usage: python evaluation_script.py <ground_truth_file_path> <predicted_answers_file_path>"
    )
    sys.exit(1)

ground_truth_path = sys.argv[1]
pred_path = sys.argv[2]

try:
    # Load ground truth and predicted answers from JSON files
    ground_truth = json.load(open(ground_truth_path))
    answers = json.load(open(pred_path))
except FileNotFoundError as e:
    print("File not found:", e.filename)
    sys.exit(1)
except json.JSONDecodeError as e:
    print("Error reading JSON:", e)
    sys.exit(1)

# Clean and standardize predicted answers
for k, v in answers.items():
    v = v.strip().lower()
    if v in ("1", "yes"):
        answers[k] = "yes"
    elif v in ("2", "no"):
        answers[k] = "no"
    else:
        answers[k] = "maybe"

# Ensure the predicted instances match the ground truth
assert set(ground_truth.keys()) == set(
    answers.keys()
), "Please predict all and only the instances in the test set."

# Retrieve PMIDs and corresponding truth and predictions
pmids = list(ground_truth)
truth = [ground_truth[pmid] for pmid in pmids]
preds = [answers[pmid] for pmid in pmids]

# Calculate accuracy and macro-F1 scores
acc = accuracy_score(truth, preds)
maf = f1_score(truth, preds, average="macro")

print("Accuracy: %f" % acc)
print("Macro-F1: %f" % maf)