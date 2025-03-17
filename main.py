import utils
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

def main(num_rows = 10000):
    
    ### STEP 1: READ DATA ###
    print('READING GAME DATA ###')
    big_game_data = pd.read_csv('data/chess_games.csv')
    
    
    ### STEP 2: EVALUATE MODELS ###
    print('METHOD 1: EVALUATING SVC ###')
    svc_data = utils.evaluate_SVC(big_game_data, num_rows=num_rows)
    print('METHOD 2: EVALUATING HISTGRAD ###')
    histgrad_data = utils.evaluate_histgrad(big_game_data, num_rows=num_rows)
    
    
    ### STEP 3: PRINT METRICS ###
    svc_cms, svc_models, svc_results_dfs = svc_data
    histgrad_cms, histgrad_models, histgrad_results_dfs = histgrad_data
    utils.print_metrics(svc_results_dfs, label = 'SVC')
    utils.print_metrics(histgrad_results_dfs, label = 'HISTGRAD')
    print('A tuple of confusion matrices, models, and dataframes has been returned. See preliminary_work/chess.ipynb for more details and visualizations. See README.md for more information.')
    
    return svc_data, histgrad_data

if __name__ == '__main__':
    main(10000)
    

    