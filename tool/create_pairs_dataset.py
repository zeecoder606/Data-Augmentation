import pandas as pd
import pose_utils
from itertools import permutations

LABELS = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
               'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']

MISSING_VALUE = -1

def give_name_to_keypoints(array):
    res = {}
    for i, name in enumerate(LABELS):
        if array[i][0] != MISSING_VALUE and array[i][1] != MISSING_VALUE:
            res[name] = array[i][::-1]
    return res


def pose_check_valid(kp_array):
    kp = give_name_to_keypoints(kp_array)
    return check_keypoints_present(kp, ['Rhip', 'Lhip', 'Lsho', 'Rsho'])


def check_keypoints_present(kp, kp_names):
    result = True
    for name in kp_names:
        result = result and (name in kp)
    return result

def filter_not_valid(df_keypoints):
    def check_valid(x):
        kp_array = pose_utils.load_pose_cords_from_strings(x['keypoints_y'], x['keypoints_x'])
        distractor = x['name'].startswith('-1') or x['name'].startswith('0000')
        return pose_check_valid(kp_array) and not distractor
    return df_keypoints[df_keypoints.apply(check_valid, axis=1)].copy()


def make_pairs(df):
    persons = df.apply(lambda x: '_'.join(x['name'].split('_')[0:1]), axis=1)
    df['person'] = persons
    fr, to = [], []
    for person in pd.unique(persons):
        pairs = zip(*list(permutations(df[df['person'] == person]['name'], 2)))
        if len(pairs) != 0:
            fr += list(pairs[0])
            to += list(pairs[1])
    pair_df = pd.DataFrame(index=range(len(fr)))
    pair_df['from'] = fr
    pair_df['to'] = to
    return pair_df

def create_pairs(annotations_file_train, pairs_file_train):
    df_keypoints = pd.read_csv(annotations_file_train, sep=':')
    df = filter_not_valid(df_keypoints)
    print ('Compute pair dataset for train...')
    pairs_df_train = make_pairs(df)
    print ('Number of pairs: %s' % len(pairs_df_train))
    pairs_df_train.to_csv(pairs_file_train, index=False)

    

