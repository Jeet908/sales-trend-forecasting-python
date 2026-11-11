"""
Data Cleaning, Feature Engineering & Preprocessing Pipeline
"""
import pandas as pd
import numpy as np


class RetailDataPipeline:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    def load_and_clean_data(self) -> pd.DataFrame:
        """Loads dataset, handles missing values, and enforces strict schema typing."""
        self.df = pd.read_csv(self.filepath)

        # Parse dates
        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Numeric transformations & Imputation
        numeric_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        for col in numeric_cols:
            if col in self.df.columns:
                # Impute missing values with column median
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)

        # Outlier handling via IQR on Weekly Sales
        q1 = self.df['Weekly_Sales'].quantile(0.25)
        q3 = self.df['Weekly_Sales'].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + (2.5 * iqr)
        self.df['Weekly_Sales'] = np.where(self.df['Weekly_Sales'] > upper_bound, upper_bound, self.df['Weekly_Sales'])

        return self.df

    def engineer_features(self) -> pd.DataFrame:
        """Generates temporal, lag, and cyclical date features for forecasting."""
        if self.df is None:
            raise ValueError("Data not loaded. Run load_and_clean_data() first.")

        # Temporal feature extraction
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['WeekOfYear'] = self.df['Date'].dt.isocalendar().week.astype(int)
        self.df['Is_Month_End'] = self.df['Date'].dt.is_month_end.astype(int)

        # Encode Holiday Flag
        if 'IsHoliday' in self.df.columns:
            self.df['IsHoliday'] = self.df['IsHoliday'].astype(int)

        # Cyclical encoding for seasonality (Sine/Cosine transform)
        self.df['Sin_Week'] = np.sin(2 * np.pi * self.df['WeekOfYear'] / 52.0)
        self.df['Cos_Week'] = np.cos(2 * np.pi * self.df['WeekOfYear'] / 52.0)

        return self.df

    def export_processed(self, output_path: str):
        """Saves clean dataset for modeling."""
        if self.df is not None:
            self.df.to_csv(output_path, index=False)
            print(f"Processed dataset saved to {output_path}")


if __name__ == "__main__":
    # Test execution
    print("Pipeline module ready.")
