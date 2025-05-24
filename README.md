# Rain in Australia - Next-Day Prediction Model (Data Science)

## Project Overview

## Table of Contents

### Data Source

The data for this project was obtained from the Kaggle dataset titled *Rain in Australia*, which originates from the Australian Bureau of Meteorology's Daily Weather Observations. Further weather-related information for Australia is available through the bureau’s Climate Data Online web application.

### Business Problem

Weather, and our ability to predict it accurately, plays a vital role in numerous aspects of daily life. Whether it's farmers managing crops, families planning weekend getaways, or airlines making logistical decisions, rain is a key factor that can significantly affect plans. In some cases, rainfall can even lead to substantial financial implications. Consequently, a wide range of stakeholders have a vested interest in reliable rain forecasting. This project aims to develop a model that predicts whether it will rain the following day, using the provided dataset. Such a model could be integrated into a weather application to serve the general public.

### Repository Stucture

## 📁 Project Structure

```
├── images/          # Exported images of plots
├── saved_models/    # Saved hyperparameter-tuned models for quick access
├── submissions/     # Files used for the project submissions
├── .gitignore
├── LICENSE
├── README.md
├── notebook.ipynb   # Jupyter notebook containing the analysis and models 
└── weatherAUS.csv   # Data on weather conditions in Australia
```

## Exploratory Data Analysis

### Column Definitions

| **Column Name**   | **Definition**                                                                 | **Units**               |
|-------------------|---------------------------------------------------------------------------------|--------------------------|
| Date              | Date of the observation                                                        | N/A                      |
| Location          | Location of the weather station                                                 | N/A                      |
| MinTemp           | Minimum temperature in the 24 hours to 9am. Sometimes only known to the nearest whole degree | Degrees Celsius         |
| MaxTemp           | Maximum temperature in the 24 hours to 9am. Sometimes only known to the nearest whole degree | Degrees Celsius         |
| Rainfall          | Precipitation (rainfall) in the 24 hours to 9am. Sometimes only known to the nearest whole millimeter | Millimeters             |
| Evaporation       | "Class A" pan evaporation in the 24 hours to 9am                                | Millimeters              |
| Sunshine          | Bright sunshine in the 24 hours to midnight                                     | Hours                    |
| WindGustDir       | Direction of the strongest wind gust in the 24 hours to midnight                | 16 compass points        |
| WindGustSpeed     | Speed of the strongest wind gust in the 24 hours to midnight                    | Kilometers per hour      |
| WindDir9am        | Direction of the wind at 9am                                                    | 16 compass points        |
| WindDir3pm        | Direction of the wind at 3pm                                                    | 16 compass points        |
| WindSpeed9am      | Speed of the wind at 9am                                                        | Kilometers per hour      |
| WindSpeed3pm      | Speed of the wind at 3pm                                                        | Kilometers per hour      |
| Humidity9am       | Relative humidity at 9am                                                        | Percent                  |
| Humidity3pm       | Relative humidity at 3pm                                                        | Percent                  |
| Pressure9am       | Atmospheric pressure reduced to mean sea level at 9am                           | Hectopascals             |
| Pressure3pm       | Atmospheric pressure reduced to mean sea level at 3pm                           | Hectopascals             |
| Cloud9am          | Fraction of sky obscured by cloud at 9am                                        | Eighths                  |
| Cloud3pm          | Fraction of sky obscured by cloud at 3pm                                        | Eighths                  |
| Temp9am           | Temperature at 9am                                                              | Degrees Celsius          |
| Temp3pm           | Temperature at 3pm                                                              | Degrees Celsius          |
| RainToday         | Did the current day receive precipitation exceeding 1mm in the 24 hours to 9am | Binary (0 = No, 1 = Yes) |
| RainTomorrow      | Did the next day receive precipitation exceeding 1mm in the 24 hours to 9am    | Binary (0 = No, 1 = Yes) |


### Insights

![image](https://github.com/user-attachments/assets/50e74093-db38-40fa-853a-f89e95b3a80a)

### Rain Days by Location

![image](https://github.com/user-attachments/assets/1fb2d303-e129-4faf-9cb7-fe23daab0eb3)

### Seasonality of Rainfall

![image](https://github.com/user-attachments/assets/134c2623-1579-4b4f-b848-02fe62ab9695)

### Correlation Matrix

![image](https://github.com/user-attachments/assets/69ad631a-f959-427c-be7f-360af1c5d1be)


















