import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import numpy as np

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, 
                 sqft_living: float,
                 sqft_lot: int,
                 date: str,
                 bedrooms: float,
                 bathrooms:float,
                 sqft_basement: float,
                 sqft_above: float,
                 condition: int,
                 view: int,
                 waterfront: int,
                 floors: float,
                 statezip: str,
                 yr_renovated: int,
                 yr_built: int
                 ):
        
        # Mapping arguments to the class object
        self.sqft_living = sqft_living
        self.sqft_lot= sqft_lot
        self.date = date
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms
        self.sqft_basement = sqft_basement
        self.sqft_above = sqft_above
        self.condition = condition
        self.view = view
        self.waterfront = waterfront
        self.floors = floors
        self.statezip = statezip
        self.yr_renovated = yr_renovated
        self.yr_built = yr_built

          
        

    def get_data_as_data_frame(self):
        try:
            # 1. Your existing dictionary mapping
            custom_data_input_dict = {
                "sqft_living": [self.sqft_living],
                "sqft_lot": [self.sqft_lot],
                "date": [self.date],
                "bedrooms": [self.bedrooms],
                "bathrooms": [self.bathrooms],
                "sqft_basement": [self.sqft_basement],
                "sqft_above": [self.sqft_above],
                "condition": [self.condition],
                "view": [self.view],
                "waterfront": [self.waterfront],
                "floors": [self.floors],
                "statezip": [self.statezip],
                "yr_renovated": [self.yr_renovated],
                "yr_built": [self.yr_built],
            }

            df = pd.DataFrame(custom_data_input_dict)

            # 1. Convert the user input date to a pandas datetime object
            df['date'] = pd.to_datetime(df['date'])
            prediction_year = df['date'].dt.year

            # 2. Dynamic Feature Engineering based on USER date
            df['house_age'] = (prediction_year - df['yr_built']).clip(lower=0)
            
            # Handle renovation logic
            df['till_renovated'] = np.where(df['yr_renovated'] == 0, 
                                            df['house_age'], 
                                            (prediction_year - df['yr_renovated']).clip(lower=0))

            # 3. Infinity/Safety Guards
            df['footprint'] = df['sqft_living'] / df['floors'].replace(0, 1)
            df['lawn_area'] = (df['sqft_lot'] - df['footprint']).clip(lower=0)
            df['land_utility_ratio'] = df['sqft_living'] / df['sqft_lot'].replace(0, 1)
            df['zip_numeric'] = df['statezip'].str.extract('(\d+)').astype(float).fillna(0)

            # 4. Final Cleanup for Infinity errors
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

            # 5. Drop columns that your model doesn't expect as final inputs
            # Ensure 'date' is dropped here so it doesn't go to the scaler
            cols_to_drop = ['date', 'yr_renovated', 'sqft_lot', 'yr_built', 'statezip']
            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

            return df

        except Exception as e:
            raise CustomException(e, sys)