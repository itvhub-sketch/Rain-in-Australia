# Rain in Australia - Next-Day Prediction Model (Data Science)

## Project Overview

This project aims to predict whether it will rain tomorrow using historical weather data from the Australian Bureau of Meteorology, accessed via Kaggle. The dataset contains various meteorological attributes, including temperature, humidity, wind, and atmospheric pressure, across multiple locations in Australia. To address the business need for accurate weather forecasting, especially for sectors like agriculture and logistics, a **Decision Tree Classifier** was employed due to its simplicity, interpretability, and effectiveness with both categorical and numerical data.

The preprocessing steps involved imputing missing values using `SimpleImputer` for numerical features and replacing missing categorical entries with the label "unknown." The `Date` column was decomposed into new features: `Year`, `Month`, and `Day`, enabling the model to capture seasonal patterns. Categorical variables were transformed using one-hot encoding to prepare the data for training.

Exploratory data analysis revealed seasonal trends and location-based variations in rainfall. The Decision Tree model was selected for its ability to handle nonlinear relationships and fast training time, making it suitable for this structured dataset. The model output can be integrated into a broader weather prediction system for real-time decision-making and planning.



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

# Data Preprocessing

## Missing Values

The SimpleImputer is better suited for this project because it offers a fast, efficient, and straightforward approach to handling missing data, which is ideal when working with structured weather datasets. Since many of the missing values in this dataset can be reasonably estimated using statistical measures like the mean, median, or most frequent value, SimpleImputer provides an effective solution without introducing unnecessary complexity. It ensures that the dataset remains consistent and fully numeric, which is important for downstream analysis such as plotting distributions, computing correlations, or training machine learning models. Unlike more complex imputers such as IterativeImputer, which require multiple passes and model-based predictions, SimpleImputer completes the task in a single operation, reducing computational overhead. This makes it ideal for exploratory data analysis and visualization stages of the project, where simplicity, speed, and interpretability are more important than precision. Additionally, it avoids overfitting or bias that could result from imputation models trained on small or noisy subsets.

## Categorical data

In this project, any categorical columns with missing values are handled by replacing the missing entries with the label `"unknown"`. This method ensures that no data is lost by avoiding the need to drop rows with missing values, which is especially important when preserving the overall structure and size of the dataset. By labeling these missing categories explicitly, the dataset remains complete and consistent, and the `"unknown"` label acts as a clear marker that indicates missing information without distorting the data. This approach is particularly suitable for categorical features where statistical imputation methods, such as using the mean or median, are not applicable.

### ✅ **Explanation (in simple terms):**

This sentence describes how missing values in a dataset were filled in:

* For **numerical (continuous) columns**, a method called **IterativeImputer** was used. This fills in missing numbers by looking at patterns in the other columns and predicting what the missing value should be.
* For **categorical columns** (which contain labels or categories), missing values were filled in using `np.random.choice()`. This randomly selects one of the existing categories, but gives more common values a higher chance of being chosen—based on how frequently they appear in the data.

---

## Extracting the Month

 Day and month might have a significant impact on the prediction. For example in India, it is highly likely to rain in the monsoon than in winter. I broke date into day, month and year and then drop the date column from the dataset

### 🆕 New Columns Created (during preprocessing)

- `Year` — extracted from `Date`
- `Month` — extracted from `Date`
- `Day` — extracted from `Date`

> 🔁 **Note**: The original `Date` column was dropped after extracting `Year`, `Month`, and `Day`.

## Modeling

from sklearn.tree import DecisionTreeClassifier

The **Decision Tree Classifier** is often chosen because it is easy to understand and interpret; the decision-making process can be visualized clearly in a tree structure. It handles both numerical and categorical data effectively without requiring extensive preprocessing. Being a non-parametric model, it makes no assumptions about the underlying data distribution, making it versatile across different types of datasets. Additionally, decision trees can capture complex, nonlinear relationships between features and the target variable. They are also relatively fast to train and predict, especially when working with medium-sized datasets, which makes them suitable for a wide range of classification tasks.

## Conclusion

The **Decision Tree Classifier** can be improved in several key ways. One common issue is overfitting, where the model becomes too complex and captures noise in the training data. This can be addressed through pruning—either by limiting the depth and complexity of the tree before it is built (pre-pruning), or by removing unnecessary nodes after the tree has been created (post-pruning). Another important enhancement involves tuning hyperparameters such as `max_depth`, `min_samples_split`, and the splitting criterion using methods like Grid Search or Random Search.

Improving the quality of input features through effective feature engineering also plays a significant role. This includes creating new features, removing irrelevant ones, and properly encoding categorical variables. A major improvement can be achieved by using ensemble methods such as Random Forests or Gradient Boosting (e.g., XGBoost or LightGBM), which combine multiple decision trees to increase predictive accuracy and reduce overfitting.

Furthermore, employing cross-validation helps ensure that the model generalizes well to unseen data by providing a more reliable evaluation. Lastly, if the dataset is imbalanced—for example, with significantly more "No Rain" than "Rain" days—techniques like oversampling, undersampling, or adjusting class weights can help the model learn both classes effectively.


















