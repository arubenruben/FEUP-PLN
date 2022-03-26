import os
import pandas as pd
  
def convert_xlsx_to_csv(filename):
    path = '.\dataset'
    read_file = pd.read_excel(os.path.join(path, filename + '.xlsx'))
    read_file.to_csv(os.path.join(path, filename + '.csv'), index = None, header = True)
    #df = pd.DataFrame(pd.read_csv(os.path.join(path, filename + '.csv')))
    #print(df)

convert_xlsx_to_csv('OpArticles')
convert_xlsx_to_csv('OpArticles_ADUs')