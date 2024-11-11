import pandas as pd
df1 = pd.read_csv('path of new test result')
df2 = pd.read_csv('./test_ocr_result.csv')
def decision(result_classifier,result_text):
    if(result_text == 'no-text'):
        return result_classifier
    if(result_text == 'FETA'):
        return result_text
    if(result_classifier == 'SAINT-NECTAIRE' or result_classifier == 'CHABICHOU'):
        return result_classifier
    return result_text

dict = {'id':[],'label':[]}
for i in range(len(df1)):
    final_result = decision(df1['label'][i],df2['label'][i])
    dict['id'].append(df1['id'][i])
    dict['label'].append(final_result)

df_dict = pd.DataFrame(dict)
df_dict.to_csv('submission_ocr.csv',index = False)