from datasets import load_dataset
from transformers import pipeline
import seaborn as sns
import pandas as pd
import csv


'''Function used to create a plot (and save it)'''
def create_plot(name, df, xVal, yVal=None, hueVal=None):
    sns.set_theme(style='whitegrid', palette='muted')

    # Check for y-axis
    if yVal:
        # Check for hue variable
        if hueVal:
            ax = sns.stripplot(data=df, x=xVal, y=yVal, hue=hueVal, alpha=0.2)
        else:
            ax = sns.stripplot(data=df, x=xVal, y=yVal, alpha=0.2)
    # Only x-axis
    else:
        # Check for hue variable
        if hueVal:
            ax = sns.stripplot(data=df, x=xVal, hue=hueVal, alpha=0.2)
        else:
            ax = sns.stripplot(data=df, x=xVal, alpha=0.2)
    
    # Plot modifications
    ax.set(
        title=name[:-12],
        ylabel='Mark', 
        xlabel='Probability'
        )
    ax.legend([], [], frameon=False)
    fig = ax.get_figure()
    fig.savefig(name)

    print('Created', name)

    # Clear current figure for generating next one
    fig.clf()

'''Function for obtaining and processing/formatting the model's scores'''
def model_scores(results):
    # Obtain adjusted scores between 0-1
    adjusted_score = [] # unacceptable = 0, acceptable = 1
    adjusted_score_applied = []
    model_label = []
    model_certainty = []
    # Loop through results
    for result in results:
        model_label.append(result['label'])
        model_certainty.append(result['score'])
        # For acceptable classified items
        if result['label'] == 'LABEL_1':
            adjusted_score.append(result['score'])
            adjusted_score_applied.append(0)
        # Adjustment for unacceptable classified items
        elif result['label'] == 'LABEL_0':
            adjusted_score.append(1 - result['score'])
            adjusted_score_applied.append(1)
        # To keep the len of list same, if for some reason the label is different/missing
        else: 
            adjusted_score.append(0)
            adjusted_score_applied.append(0)
    
    d = {
            'model_score_adjusted': adjusted_score,
            'model_label': model_label,
            'model_score': model_certainty,
            'score_adjusted': adjusted_score_applied
        }
    
    return d

'''Function that calculates the median value'''
def median(listx):
    if (len(listx) % 2) == 0: # even lenght list
        n1 = int((len(listx) / 2) - 1)
        n2 = int(len(listx) / 2)
        med = (list(listx)[n1] + list(listx)[n2]) / 2
    else: # odd lenght list
        n = int(len(listx) / 2)
        med = list(listx)[n]
    
    return med

'''Function for printing scores per marking,
which can be used for analysis per marking'''
def mark_analysis(df):
    for mark in set(df['original_mark']):
        df_x = df.loc[df['original_mark'] == mark]
        avg_mark = sum(df_x['score_adj']) / len(df_x['score_adj'])
        min_mark = min(df_x['score_adj'])
        max_mark = max(df_x['score_adj'])
        med_mark = median(df_x['score_adj'])
        std_mark = df_x['score_adj'].std()

        print(mark, '\trange:', min_mark, '-', max_mark)
        print('\tavg:', avg_mark)
        print('\tmedian:', med_mark)
        print('\tstd:', std_mark)
        print()


def main():
    # Load dataset / models
    test_dataset = load_dataset('GroNLP/dutch-cola', split='test')
    bertje_CoLA = pipeline('text-classification', model='HylkeBr/bertje_dutch-cola')
    robbert_CoLA = pipeline('text-classification', model='HylkeBr/robbert_dutch-cola')
    # Downloaded intermediate.csv from HF:GroNLP/dutch-cola
    with open('intermediate.csv') as intermediate_annotations:
        csv_reader = csv.reader(intermediate_annotations, delimiter=',')

        # Obtain original scores from test set
        sents = []
        original_score = []
        original_mark = []
        for row in test_dataset:
            sents.append(row['Sentence'])
            original_score.append(row['Acceptability'])
            marking = row['Original annotation']
            if marking == None:
                original_mark.append(0)
            else: # marking is *|*?|??|?
                original_mark.append(marking)

        # Obtain original scores from intermediate set
        csv_line = 0
        for csvrow in csv_reader:
            if csv_line != 0: # do not collect column names
                acceptability = csvrow[2]
                annotation = csvrow[3]
                sent = csvrow[4]
                sents.append(sent)
                original_score.append(acceptability)
                if annotation == None:
                    original_mark.append(0)
                else: # marking is *|*?|??|?
                    original_mark.append(annotation)
            csv_line += 1


    # Obtain model predictions
    bertje_results = bertje_CoLA(sents)
    robbert_results = robbert_CoLA(sents)

    # Extract results
    bertje_data = model_scores(bertje_results)
    robbert_data = model_scores(robbert_results)

    # Create dataframe
    d_bertje = {
        'original_score': original_score, 
        'original_mark': original_mark, 
        'sentence': sents,
        'score_adj': bertje_data['model_score_adjusted']
        }
    d_robbert = {
        'original_score': original_score, 
        'original_mark': original_mark, 
        'sentence': sents,
        'score_adj': robbert_data['model_score_adjusted']
        }

    df_bertje = pd.DataFrame(data=d_bertje)
    df_robbert = pd.DataFrame(data=d_robbert)
    
    # Create visualizations
    create_plot('BERTje_results.png', df_bertje, 'score_adj', 'original_mark', 'original_mark')
    create_plot('RobBERT_results.png', df_robbert, 'score_adj', 'original_mark', 'original_mark')

    # Analysis of different markings
    print('\nBERTje')
    mark_analysis(df_bertje)

    print('RobBERT')
    mark_analysis(df_robbert)

    print('\nFinished')


if __name__ == '__main__':
    main()
