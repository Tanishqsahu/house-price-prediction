import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def perform_feature_engineering(self,df):
    # 1. House Age logic from your notes
    # Assuming 'date' is a string, convert it first

     import datetime as dt
     df['date'] = pd.to_datetime(df['date'])
    
     df['house_age'] = df['date'].dt.year - df['yr_built']
    # till_renovated
     df['till_renovated'] = np.where(
     df['yr_renovated'] == 0,
     -1,  # sentinel: never renovated
     df['date'].dt.year - df['yr_renovated']
)
    
    # lawn area 
     df['lawn_area'] = df['sqft_lot'] - (df['sqft_living'] / df['floors']) 
     df['land_utility_ratio'] = df['sqft_living'] / df['sqft_lot']

    #footprint
     df['footprint']=df['sqft_living']/df['floors']

    # zip_numeric
     df['zip_numeric'] = pd.to_numeric(df['statezip'].str.split().str[-1], errors='coerce').fillna(0)


# Convert it to a proper integer so the model can read it
     df['zip_numeric'] = pd.to_numeric(df['zip_numeric'])
 
    
    
    
     
     

    #log(price)
     df = df[df['price'] > 0].copy()
     df['log_price'] = np.log1p(df['price'])
    

     df.replace([np.inf, -np.inf], np.nan, inplace=True)
     
    
    # 4. Drop the columns you listed in your notes
     cols_to_drop = ['street','country','statezip','date','city','price', 'yr_renovated', 'sqft_lot','till_renovated','lawn_area','yr_built']
     df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
     return df
    
    

    def get_data_transformer_object(self):
     try:
        # Define which columns are which
        numerical_columns = ['sqft_living', 'footprint', 'land_utility_ratio', 'bedrooms',
       'bathrooms', 'sqft_basement', 'sqft_above', 'condition', 'view',
       'waterfront', 'floors', 'zip_numeric']
       

        # 1. Numerical Pipeline (Impute + Scale)
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")) # Safety net
               
            ]
        )

        # 2. Categorical Pipeline (Impute + OneHot)
        

     
        logging.info(f"Numerical columns: {numerical_columns}")

        # 3. Combine them into a ColumnTransformer
        preprocessor = ColumnTransformer(
            [
                ("num_pipeline", num_pipeline, numerical_columns),
                
            ]
        )

        return preprocessor

     except Exception as e:
        raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
     try:
        # ✅ Directly read — no nested function
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        print(f"After read — train shape: {train_df.shape}")

        train_df = self.perform_feature_engineering(train_df)
        test_df = self.perform_feature_engineering(test_df)

        print(f"After feature eng — train shape: {train_df.shape}")

        logging.info("Obtaining preprocessing object")
        preprocessing_obj = self.get_data_transformer_object()

        target_column_name = "log_price"

        input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
        target_feature_train_df = train_df[target_column_name]

        input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
        target_feature_test_df = test_df[target_column_name]

        logging.info("Applying preprocessing object on training and testing dataframe.")

        input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

        train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
        test_arr  = np.c_[input_feature_test_arr,  np.array(target_feature_test_df)]

        logging.info("Saved preprocessing object.")

        save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessing_obj
        )

        return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path
        )

     except Exception as e:
        raise CustomException(e, sys)

#if __name__=="__main__":
    #obj=DataIngestion()
    #train_data,test_data=obj.inititate_data_ingestion()

    #data_transformation=DataTransformation()
    


