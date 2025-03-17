
### UTILS.PY ###
# This file contains all the necessary imports and functions for main.py

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")


from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re



def select_random_rows(data, num_rows):
    games_ix = np.random.choice(data.index, num_rows, replace=False)
    games_sample = data.loc[games_ix]
    return games_sample



def get_game_length(game):
    return len(re.findall(r'[\d]+\. ', game))



def flatten(xss):
    flat_list = [
    x
    for xs in xss
    for x in xs
]
    return flat_list


def get_moves(game_string):
    '''Take in a game as AN, convert into list of moves. Works on dirty rows, with {%eval}, too.
    
    Ex: "1. e4 e6 2. d4 d5 3. e5 c5" -> ['e4', 'e6', 'd4', 'd5', 'e5', 'c5']'''
    
    moves_double = re.split(r'[\d]+\. ', game_string)
    has_eval = len(re.findall(r'\[%eval.+\]', game_string)) > 0
    
    outcomes = ['0-1', '1-0', '1/2-1/2']

    # Handle "dirty" games with {eval} annotations 
    if has_eval:
        # print('has eval')
        moves_single_dirty = [[m.strip() for m in re.split(r'[\d]+\.\.+', md)] for md in moves_double[1:]]
        moves_single_dirty = flatten(moves_single_dirty)
        moves_single = [re.sub(r'\{.+\}', '', ms).strip() for ms in moves_single_dirty]
        
        for outcome in outcomes:
            if outcome in moves_single[-1]:
                moves_single[-1] = moves_single[-1].split(outcome)[0].strip()
                break
            
    # Handle "clean" games without {eval} annotations 
    else:
        moves_single = flatten([md.strip().split(' ') for md in moves_double[1:]])
        
        if moves_single[-1] in outcomes:
            moves_single = moves_single[:-1]
            
    # Remove ?!+ characters
    moves_single = [re.sub(r'[?!+]', '', ms) for ms in moves_single]
        
    return moves_single



def clean_event_col(df):
    '''Input df, outputs a df with OHE Event columns.'''
    
    df = df.copy()
    df['Event'] = df['Event'].apply(lambda x: x.lower().strip().split(' ')[0])
    
    ## NOTE: removes correspondence games
    dummy_cols = pd.get_dummies(df['Event'])
    df = pd.concat([dummy_cols, df], axis=1)
    df = df.drop(columns=['Event'])
    
    return df



def add_move_features(data, n):
    '''
    INPUTS:
    DATA is a df. 
    N is the number of rounds (a single move counts as both white moving + black moving). 
    
    OUTPUT:
    Return copy of DATA with an additional 3*2*N new columns. Every column is read as a move for exactly one side.
    
    "sw1" means white's first move square, 
    "sb1" means black's first move square. 
    "pw1" means white's first move piece, 
    "pb1" means black's first move piece. 
    "cw1" is boolean if white's first move had a capture, 
    "cb1" is boolean if black's first move had a capture. 
    '''
    
    data = data.copy()
    data['moves'] = data['AN'].apply(get_moves)
    
    max_single_moves = data['moves'].apply(lambda l: len(list(l))).max()
    num_columns = min(2*n, max_single_moves)
    
    ### MAKE ALL LIST LENGTHS THE SAME ###
    empty_move = ''
    data['moves'] = data['moves'].apply(lambda l: l + [empty_move]*(num_columns - len(l)))
    data['moves'] = data['moves'].apply(lambda l: l[:num_columns])
    
    
    ### GET ONLY THE SQUARES ###
    def get_square_value(x):
        
        promotions = ['=Q', '=N', '=B', '=R']
        
        if '*' in x:
            x = x.strip('*')
        
        # For not castles
        if 'O-O' not in x:
            x = x.strip('#')
            for promotion in promotions:
                x = x.strip(promotion)
            return x[-2:]
        
        # For castles
        elif 'O-O' in x:
            x = x.strip('#')
            return x
        
        else:
            return ''
    
    ### GET ONLY THE PIECE MOVED ###
    def get_piece(x):
        
        promotions = ['=Q', '=N', '=B', '=R']
        
        if '*' in x:
            x = x.strip('*')
        
        # For not castles
        if 'O-O' not in x:
            for promotion in promotions:
                if promotion in x:
                    return promotion.strip('=')
                
            pieces = ['R', 'N', 'B', 'Q', 'K']
            for piece in pieces:
                if piece in x:
                    return piece
            return 'P'
        
        # For castles
        elif 'O-O' in x:
            return 'K'
        
        else:
            return ''
        
    ### GET IF A MOVE WAS A CAPTURE ###
    def get_capture(x):
        return 'x' in x
        
    data['squares'] = data['moves'].apply(lambda l: [get_square_value(x) for x in l])
    data['pieces'] = data['moves'].apply(lambda l: [get_piece(x) for x in l])
    data['captures'] = data['moves'].apply(lambda l: [get_capture(x) for x in l])
    
    ### SEPERATE INTO COLUMNS ###
    cols = []
    for i in range(num_columns):
        
        move_num = i//2 + 1
        
        if i%2 == 0:
            cols.append((data['squares'].apply(lambda l: l[i])).rename(f'sw{move_num}')) # Add white move square
            cols.append((data['pieces'].apply(lambda l: l[i])).rename(f'pw{move_num}').astype('category')) # Add white move piece
            cols.append((data['captures'].apply(lambda l: l[i])).rename(f'cw{move_num}')) # Add white move capture boolean
            
        if i%2 == 1:
            cols.append((data['squares'].apply(lambda l: l[i])).rename(f'sb{move_num}')) # Add black move square
            cols.append((data['pieces'].apply(lambda l: l[i])).rename(f'pb{move_num}').astype('category')) # Add black move piece
            cols.append((data['captures'].apply(lambda l: l[i])).rename(f'cb{move_num}')) # Add black move capture boolean
        
    return pd.concat([data] + cols, axis=1)


white_chess_dict = {
    '': 0,
    'a1': 1,
    'a2': 2,
    'a3': 3,
    'a4': 4,
    'a5': 5,
    'a6': 6,
    'a7': 7,
    'a8': 8,
    'b1': 9,
    'b2': 10,
    'b3': 11,
    'b4': 12,
    'b5': 13,
    'b6': 14,
    'b7': 15,
    'b8': 16,
    'c1': 17,
    'c2': 18,
    'c3': 19,
    'c4': 20,
    'c5': 21,
    'c6': 22,
    'c7': 23,
    'c8': 24,
    'd1': 25,
    'd2': 26,
    'd3': 27,
    'd4': 28,
    'd5': 29,
    'd6': 30,
    'd7': 31,
    'd8': 32,
    'e1': 33,
    'e2': 34,
    'e3': 35,
    'e4': 36,
    'e5': 37,
    'e6': 38,
    'e7': 39,
    'e8': 40,
    'f1': 41,
    'f2': 42,
    'f3': 43,
    'f4': 44,
    'f5': 45,
    'f6': 46,
    'f7': 47,
    'f8': 48,
    'g1': 49,
    'g2': 50,
    'g3': 51,
    'g4': 52,
    'g5': 53,
    'g6': 54,
    'g7': 55,
    'g8': 56,
    'h1': 57,
    'h2': 58,
    'h3': 59,
    'h4': 60,
    'h5': 61,
    'h6': 62,
    'h7': 63,
    'h8': 64,
    'O-O': 49,
    'O-O-O': 17 
}


black_chess_dict = {
    '': 0,
    'a1': 1,
    'a2': 2,
    'a3': 3,
    'a4': 4,
    'a5': 5,
    'a6': 6,
    'a7': 7,
    'a8': 8,
    'b1': 9,
    'b2': 10,
    'b3': 11,
    'b4': 12,
    'b5': 13,
    'b6': 14,
    'b7': 15,
    'b8': 16,
    'c1': 17,
    'c2': 18,
    'c3': 19,
    'c4': 20,
    'c5': 21,
    'c6': 22,
    'c7': 23,
    'c8': 24,
    'd1': 25,
    'd2': 26,
    'd3': 27,
    'd4': 28,
    'd5': 29,
    'd6': 30,
    'd7': 31,
    'd8': 32,
    'e1': 33,
    'e2': 34,
    'e3': 35,
    'e4': 36,
    'e5': 37,
    'e6': 38,
    'e7': 39,
    'e8': 40,
    'f1': 41,
    'f2': 42,
    'f3': 43,
    'f4': 44,
    'f5': 45,
    'f6': 46,
    'f7': 47,
    'f8': 48,
    'g1': 49,
    'g2': 50,
    'g3': 51,
    'g4': 52,
    'g5': 53,
    'g6': 54,
    'g7': 55,
    'g8': 56,
    'h1': 57,
    'h2': 58,
    'h3': 59,
    'h4': 60,
    'h5': 61,
    'h6': 62,
    'h7': 63,
    'h8': 64,
    'O-O': 56,
    'O-O-O': 24 
}



def clean_games(big_game_data, n, num_rows = 10000):
    '''
    INPUTS:
    BIG_GAME_DATA is a df of all games.
    N is the number of rounds (a single move counts as both white moving + black moving).
    NUM_ROWS is the number of rows to sample from BIG_GAME_DATA.
    
    OUTPUT:
    Returns a filtered df, each row representing a game with N rounds. 
    '''
    
    games = select_random_rows(big_game_data, num_rows)
    
    # Remove correspondence
    games = clean_event_col(games)
    games = games[games['correspondence'] == False].drop(columns=['correspondence'])
    
    # Include only finished games
    games = games[games['Result'] != '*']
    
    # Add game length
    games['Length'] = games['AN'].apply(get_game_length)
    
    # Add move features (only gets squares)
    games = add_move_features(games, n)
    
    # Filter out games with less than n moves
    games = games[games['Length'] >= n]
    
    # Use dicts
    for col in games.columns[-(6*n):]:
        if 'sw' in col:
            games[col] = games[col].apply(lambda k: white_chess_dict[k]).astype('int32')
        if 'sb' in col:
            games[col] = games[col].apply(lambda k: black_chess_dict[k]).astype('int32')
    
    return games



def get_train_test(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    X_train = train_data.drop(columns=['White', 'Black', 'Result', 'UTCDate', 'UTCTime', 'AN', 'Length', 'WhiteRatingDiff', 'BlackRatingDiff', 'ECO', 'Opening', 'TimeControl','Termination', 'AN', 'Length', 'moves', 'squares', 'pieces', 'captures'])

    X_test = test_data.drop(columns=['White', 'Black', 'Result', 'UTCDate', 'UTCTime', 'AN', 'Length', 'WhiteRatingDiff', 'BlackRatingDiff', 'ECO', 'Opening', 'TimeControl','Termination', 'AN', 'Length', 'moves', 'squares', 'pieces', 'captures'])

    y_train = train_data['Result']

    y_test = test_data['Result']
    
    return X_train, X_test, y_train, y_test



def prep_svc(n, df):
    """Additionally prepares the data for SVC by One hot encoding the chess pieces"""
    res = df.copy()
    ohe_columns = [f'pw{i}' for i in range(1, n+1)] + [f'pb{i}' for i in range(1, n+1)]
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    fit_data = ohe.fit_transform(res[ohe_columns])
    ohe_df = pd.DataFrame(fit_data, columns=ohe.get_feature_names_out())
    ohe_df.index = res.index
    res = res.drop(columns=ohe_columns)
    return pd.concat([res, ohe_df], axis=1)



def make_cm(results_df, normalize = False):
    '''Returns a resulting confusion matrix that has the structure:
    Pred B [ ] [ ] [ ]
    Pred W [ ] [ ] [ ]
    Pred D [ ] [ ] [ ]
            B   W   D
    '''
    
    results_df = results_df.copy()
    
    cm = np.zeros((3, 3))
    
    labels = {'0-1': 0, '1-0': 1, '1/2-1/2': 2}
    
    results_df['value'] = (3 * results_df['Predicted'].apply(lambda x: labels[x])) + results_df['Actual'].apply(lambda x: labels[x])
    
    for i in range(9):
        cm[i//3][i%3] = results_df[results_df['value'] == i].shape[0]
        
    if normalize:
        cm = cm/results_df.shape[0]
        
    return cm



def train_test_svc(X_train, X_test, y_train, y_test, C):
    '''Trains a support vector classifier model on the data'''
    
    svc = SVC(C=C)
    svc.fit(X_train, y_train)
    preds= svc.predict(X_test)
    values = y_test.to_numpy().ravel(), preds
    return values, svc



def train_test_histgrad(X_train, X_test, y_train, y_test):
    '''Trains a HistGradBoost classifier model on the data'''
    
    rf = HistGradientBoostingClassifier(categorical_features='from_dtype', max_iter=25, max_depth=4, max_leaf_nodes = 20, learning_rate=0.1)
    rf.fit(X_train, y_train)
    preds= rf.predict(X_test)
    values = y_test.to_numpy().ravel(), preds
    return values, rf



def evaluate_SVC(big_game_data, sights = [10, 20, 30, 40], num_rows = 10000):
    '''Evaluates the SVC model on the data. Returns confusion matrices, models, and result dataframes FOR EACH value of sight.'''
    # A model will only work when a test game has AT LEAST as many moves as sight.
    
    fig, ax = plt.subplots(2, 2, figsize = (15, 15))
    confs = []
    models = []
    results_dfs = []

    k = 4
    for ix, sight in enumerate(sights[:k]):
        print(f"SVC: Training sight = {sight}")
        i = ix // 2
        j = ix % 2
        
        labels = ['0-1', '1-0', '1/2-1/2']
        
        data = clean_games(big_game_data, sight, num_rows = num_rows)
        data_svc = prep_svc(sight, data)
        X_train, X_test, y_train, y_test = get_train_test(data_svc)
        
        results, model = train_test_svc(X_train, X_test, y_train, y_test, 10) 
        results_df = pd.DataFrame({'Actual': results[0], 'Predicted': results[1]})
        cm = make_cm(results_df, normalize=True).T
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax[i][j], xticks_rotation = 45, cmap="GnBu")
        
        ax[i][j].set_title(f'CM for Sight = {sight}')
        
        confs.append(cm)
        models.append(model)
        results_dfs.append(results_df)
        
    return confs, models, results_dfs


def evaluate_histgrad(big_game_data, sights = [10, 20, 30, 40], num_rows = 10000):
    '''Evaluates the HistGradBoost model on the data. Returns confusion matrices, models, and result dataframes FOR EACH value of sight.'''
    # A model will only work when a test game has AT LEAST as many moves as sight.

    fig, ax = plt.subplots(2, 2, figsize = (15, 15))
    confs = []
    results_dfs = []
    models = []

    k = 4
    for ix, sight in enumerate(sights[:k]):
        print(f"HistGrad: Training sight = {sight}")
        i = ix // 2
        j = ix % 2
        
        labels = ['0-1', '1-0', '1/2-1/2']
        
        data = clean_games(big_game_data, sight, num_rows = num_rows)
        X_train, X_test, y_train, y_test = get_train_test(data)
        
        results, model = train_test_histgrad(X_train, X_test, y_train, y_test) 
        results_df = pd.DataFrame({'Actual': results[0], 'Predicted': results[1]})
        cm = make_cm(results_df, normalize=True).T
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax[i][j], xticks_rotation = 45)
        
        ax[i][j].set_title(f'CM for Sight = {sight}')
        
        confs.append(cm)
        models.append(model)
        results_dfs.append(results_df)
        
    return confs, models, results_dfs
        
def print_metrics(results_dfs, label):
    '''Prints the accuracy, precision, recall, and f1 score for each sight. Uses label to make terminal output clearer.'''
    
    for ix, results_df in enumerate(results_dfs):
        y_act = results_df['Actual'].to_numpy()
        y_pred = results_df['Predicted'].to_numpy()
        print(f'\nACCURACY for {label}, SIGHT = {(ix + 1)* 10}:\n', accuracy_score(y_act, y_pred))
        print(f'\nPRECISION for {label}, SIGHT = {(ix + 1) * 10}:\n', precision_score(y_act, y_pred,average='macro'))
        print(f'\nRECALL for {label}, SIGHT = {(ix + 1)* 10}:\n', recall_score(y_act, y_pred, average='macro'))
        print(f'\nF1_SCORE for {label}, SIGHT = {(ix + 1)* 10}:\n', f1_score(y_act, y_pred, average='macro'))
  
    