import ast
import numpy as np


val_acc_results = []

for i in range(10):
    with open('../data/history/history_' + str(i) + '.txt') as file:
        history = file.read()
        dict = ast.literal_eval(history)
        val_acc_results.append(dict['val_accuracy'][19])

print(val_acc_results)
print(np.mean(val_acc_results))

