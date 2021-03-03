#╔╗════════════════════════════════════════╔╗
#╚╝════════════════════════════════════════╚╝
#╔╗══════════════╤══════════════════════════╗
#╠╣═╡ Imports                             ╞═╣
#╚╝══════════════╧══════════════════════════╝
#region
import os
import sys
import pandas as pd
import numpy as np
import Data
import Params
import TF_Controller as TFC
#endregion
#╔╗══════════════╤══════════════════════════╗
#╠╣═╡ Set Initial Parameters              ╞═╣
#╚╝══════════════╧══════════════════════════╝
#region
exp_num = 1
#endregion
#╔╗══════════════╤══════════════════════════╗
#╠╣═╡ Initialize Project                  ╞═╣
#╚╝══════════════╧══════════════════════════╝
#region
current_directory = os.getcwd()
filepaths_dict = Data.get_filepaths(current_directory)
price_data = Data.import_data(filepaths_dict)
program_params = Params.program_params
#endregion
#╔╗══════════════╤══════════════════════════╗
#╠╣═╡ Get Initial Population              ╞═╣
#╚╝══════════════╧══════════════════════════╝
#region
model_params = pd.read_csv(filepaths_dict.get('Results') + '/' + 'Final_Candidate.csv')
model_params = model_params.iloc[0,:]
data = pd.read_csv(filepaths_dict.get('Results') + '/' + 'Dataset1.csv')
labels = pd.read_csv(filepaths_dict.get('Results') + '/' + 'labels1.csv')
#removing this line causes the model to train correctly.  Leaving this line in, the model does not improve in accuracy beyond .5024.
model_params, predictions = TFC.train_model(program_params, model_params, filepaths_dict, data, labels)

#endregion