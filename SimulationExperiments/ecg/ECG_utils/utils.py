import time

import numpy as np


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def snomed2hot(snomed, HASH_TABLE):
    y = np.zeros((len(HASH_TABLE), 1)).astype(np.float32)
    for kk, p in enumerate(HASH_TABLE):
        for lbl_i in snomed:
            if lbl_i.find(p) > -1:
                y[kk] = 1

    return y


def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table) - 1
    if num_rows < 1:
        raise Exception('The table {} is empty.'.format(table_file))

    num_cols = set(len(table[i]) - 1 for i in range(num_rows))
    if len(num_cols) != 1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(num_cols)
    if num_cols < 1:
        raise Exception('The table {} is empty.'.format(table_file))

    # Find the row and column labels.
    rows = [table[0][j + 1] for j in range(num_rows)]
    cols = [table[i + 1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i + 1][j + 1]
            if is_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values


def load_weights(weight_file, classes):
    # Load the weight matrix.
    rows, cols, values = load_table(weight_file)

    assert (rows == cols)
    num_rows = len(rows)

    # Assign the entries of the weight matrix with rows and columns corresponding to the classes.
    num_classes = len(classes)
    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i, a in enumerate(rows):
        if a in classes:
            k = classes.index(a)
            for j, b in enumerate(rows):
                if b in classes:
                    l = classes.index(b)
                    weights[k, l] = values[i, j]

    return weights


class AdjustLearningRateAndLoss():
    def __init__(self, optimizer, learning_rates_list, lr_changes_list, loss_functions):
        self.optimizer = optimizer
        self.learning_rates_list = learning_rates_list
        self.lr_changes_list = lr_changes_list
        self.loss_functions = loss_functions

        self.actual_loss = self.loss_functions[0]
        self.actual_lr = self.learning_rates_list[0]
        self.lr_changes_cumulative = np.cumsum([0] + self.lr_changes_list)
        self.epoch = 0

    def step(self):
        self.epoch = self.epoch + 1

        try:
            with open('put_here_to_aply/lr_change.txt', 'r') as f:
                x = f.readlines()
                lr = float(x[0])
            time.sleep(1)
            os.remove('put_here_to_aply/lr_change.txt')

            print('lr was set to: ' + str(x))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        except:
            pass

        for ind in range(len(self.lr_changes_cumulative) - 1):
            value1 = self.lr_changes_cumulative[ind]
            value2 = self.lr_changes_cumulative[ind + 1]
            if self.epoch >= value1 and self.epoch < value2:
                self.actual_loss = self.loss_functions[ind]
                self.actual_lr = self.learning_rates_list[ind]

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.actual_lr
