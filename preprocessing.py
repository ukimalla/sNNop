import numpy as np
import tqdm



"""
Purpose: Convert CHUNK sized dataframe into X and y numpy arrays that can be read my the model
Arguments: 

seq_data: Input dataframe in the format sample_1, sample_2, ...,  sample_chunk_size, "key name"
v_map

"""


def clean_data(seq_data, v_map, chunk_size):
    X = []
    y = []

    blank_counter = 0
    empty_chunk_limit = 1
    # math.ceil(RATE/CHUNK * KPS)
    print("Empty CHUNK limit: ", empty_chunk_limit)

    print("Starting data cleanse")
    # Iterating through the input dataframe to crate X[] and y without any silence
    for i in tqdm.tqdm(range(int(len(seq_data.iloc[:, -1:].values)))):
        # If a key is detected
        if str(seq_data.iloc[:, -1:].values[i][0]) != "NONE":
            # and there were no blanks above
            if blank_counter == 0:
                inputVals = seq_data.iloc[i:i + 1, :-1].values
                target = [v_map[item[0]] for item in seq_data.iloc[i:i + 1, -1:].values]

            # and there were blanks above within empty_chunk_limits
            elif blank_counter < empty_chunk_limit:
                inputVals = seq_data.iloc[(i - blank_counter):i + 1, :-1].values
                target = [v_map[item[0]] for item in seq_data.iloc[(i - blank_counter):i + 1, -1:].values]
            # and there were blank spaces more than empty_chunk_limit
            else:
                inputVals = seq_data.iloc[(i - empty_chunk_limit):i + 1, :-1].values
                target = [v_map[item[0]] for item in seq_data.iloc[(i - empty_chunk_limit):i + 1, -1:].values]

            # If first key
            if len(X) == 0:
                X = inputVals
                y = target
            # Not the first key
            else:
                X = np.append(X, inputVals, axis=0)
                y = np.append(y, target, axis=0)
            blank_counter = 0

        # NO KEY DETECTED
        else:
            blank_counter += 1

    X = np.reshape(X, (len(X), chunk_size, 1))

    return X, y


"""
Purpose: Generates a dictionary that maps each key in a dataframe to an integer.
Returns: key list and key dictonary
"""


def generate_key_map(dataframe):
    y = dataframe.iloc[:, -1:].values
    y = [key[0] for key in y]
    y = list(set(y))

    return y, {n: i for i, n in enumerate(y)}