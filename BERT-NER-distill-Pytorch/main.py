# data = []
#
# with open("/Users/fran.yang/Downloads/nlp-from-scratch-assignment-2022-main/data/anlp-sciner-test-empty.conll", "r") as f:
#     for line in f:
#         data.append(line)
#
#
# model_data = []
# with open("./test_prediction_for_bilstm.txt", "r") as f:
#     for line in f:
#         line = line.split(" ")
#         if len(line) == 1:
#             model_data.append([""])
#         else:
#             model_data.append([line[0], line[1].strip()])
#
#
# with open("./test_prediction_bilstm.conll", "w") as f:
#     for line, model_line in zip(data, model_data):
#         if len(model_line) != 1:
#             line = line.replace("O", model_line[1])
#         f.write(line)

def _read_text(input_file):
    lines = []
    with open(input_file, 'r') as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    lines.append({"words": words, "labels": labels})
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    label = splits[-1].replace("\n", "")
                    if label.startswith("E") or label.startswith("M"):
                        label = "I-" + label.split("-")[-1]
                    labels.append(label)
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            lines.append({"words": words, "labels": labels})
    return lines

file1 = "/Users/fran.yang/Documents/tmp/data/1.conll"
file2 = "/Users/fran.yang/Documents/tmp/data/2.conll"

data = _read_text(file1) + _read_text(file2)
train_data_num = int((0.8 * len(data)))
train_data = data[:train_data_num]
test_data = data[train_data_num:]

with open("/Users/fran.yang/Documents/tmp/data/train.conll", "w") as f:
    for lines in train_data:
        for word, label in zip(lines["words"], lines["labels"]):
            f.write(word + " " + label + "\n")
        f.write("\n")

with open("/Users/fran.yang/Documents/tmp/data/test.conll", "w") as f:
    for lines in test_data:
        for word, label in zip(lines["words"], lines["labels"]):
            f.write(word + " " + label + "\n")
        f.write("\n")



