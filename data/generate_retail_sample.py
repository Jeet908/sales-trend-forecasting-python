import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
dates = [datetime(2022, 1, 7) + timedelta(days=7 * i) for i in range(104)] # 2 years of weekly data
stores = list(range(1, 11))

records = []
for store in stores:
    base_sales = np.random.uniform(15000, 30000)
    for dt in dates:
        month = dt.month
        is_holiday = 1 if (month == 11 and dt.day >= 20) or (month == 12 and dt.day >= 20) else 0
        holiday_bump = 1.35 if is_holiday else 1.0
        
        cpi = 210.0 + (dt.year - 2022) * 5.2 + np.random.normal(0, 0.5)
        unemp = max(4.0, 7.5 - (dt.year - 2022) * 0.8 + np.random.normal(0, 0.2))
        temp = 60 + 20 * np.sin(2 * np.pi * dt.timetuple().tm_yday / 365) + np.random.normal(0, 5)
        fuel = 3.20 + (dt.year - 2022) * 0.40 + np.random.normal(0, 0.1)

        sales = (base_sales * holiday_bump * (1 + 0.05 * np.sin(2 * np.pi * month / 12))) + np.random.normal(0, 1000)

        records.append({
            "Store": store,
            "Date": dt.strftime("%Y-%m-%d"),
            "Weekly_Sales": round(sales, 2),
            "Holiday_Flag": is_holiday,
            "Temperature": round(temp, 1),
            "Fuel_Price": round(fuel, 2),
            "CPI": round(cpi, 2),
            "Unemployment": round(unemp, 2)
        })

df = pd.DataFrame(records)
df.to_csv("data/retail_sales_sample.csv", index=False)
print("Created data/retail_sales_sample.csv with", len(df), "weekly records!")
