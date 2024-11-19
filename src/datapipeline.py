
# For working with DataFrames
import pandas as pd

# For preprocessing data
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# To perform cleaning, call preprocessing_pipeline(df) > split_data(X, y, test_size)
# Or call transform(datapath: str) for end to end processing

# Feature Selection Function

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature selection based on EDA

    Parameters:
    - df (DataFrame): DataFrame to perform feature selection on
    
    Returns:
    - df (DataFrame): A subset (selected features) of the original DataFrame 
    """
    # Define columns to drop, drop id also it is just a unique key
    cols_to_drop = ['id','citizenship', 'reason_for_unemployment', 'detailed_household_summary_in_household',
                'detailed_occupation_recode','own_business_or_self_employed', 'year', 'region_of_previous_residence', 
                'state_of_previous_residence', 'migration_code_change_in_reg', 'fill_inc_questionnaire_for_veteran_s_admin', 
                'migration_code_move_within_reg', 'live_in_this_house_1_year_ago', 'migration_prev_res_in_sunbelt',
                'major_industry_code','migration_code_change_in_msa'
            ]
    
    # Perform drop columns
    df = df.drop(cols_to_drop, axis=1)

    return df

# Feature Engineering Function

def education_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters:
    - df (DataFrame): dataframe to be transformed

    Returns:
    - df[['education']] (DataFrame): education attainment mapped into 7 levels
    """
    value_to_index = {'Children':0,
                 'Less than 1st grade':0,
                 '1st 2nd 3rd or 4th grade':0,
                 '5th or 6th grade':0,
                 '7th and 8th grade':0,
                 '9th grade':0,
                 '10th grade':0,
                 '11th grade':0,
                 '12th grade no diploma':0,
                 'High school graduate':1,
                 'Some college but no degree':2,
                 'Associates degree-occup /vocational':2,
                 'Associates degree-academic program':2,
                 'Bachelors degree(BA AB BS)':3,
                 'Masters degree(MA MS MEng MEd MSW MBA)':4,
                 'Prof school degree (MD DDS DVM LLB JD)':5,
                 'Doctorate degree(PhD EdD)':6
    }

    df['education'] = df['education'].replace(value_to_index)
    return df[['education']]

# Preprocessing Pipeline Function

def preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes the data cleaning pipeline

    Parameters:
    - df (DataFrame): dataframe to be transformed

    Returns:
    - X_cleaned (DataFrame): cleaned independent features 
    - y_cleaned (DataFrame): cleaned target features
    - class_mapping (Dictionary): encoding for target labels
    """
    # Perform feature selection with custom drop_columns(df) function
    df = drop_columns(df)

    # Define X as input features and y as the outcome variable
    X = df.drop('income_group', axis=1)
    y = df['income_group']

    # Define numerical features
    numerical_features = ['age', 'wage_per_hour', 'num_persons_worked_for_employer', 'weeks_worked_in_year',
                      'capital_gains', 'capital_losses', 'dividends_from_stocks']

    # Define nominal features
    nominal_features = ['enroll_in_edu_inst_last_wk', 'marital_stat', 'race', 'sex', 'country_of_birth_father',
                    'country_of_birth_mother', 'country_of_birth_self', 'class_of_worker', 'detailed_industry_recode',
                    'major_occupation_code', 'hispanic_origin', 'member_of_a_labor_union', 'full_or_part_time_employment_stat',
                    'detailed_household_and_family_stat', 'family_members_under_18', 'veterans_benefits', 'tax_filer_stat']

    # Define ordinal features
    ordinal_features = ['education']

    # Scale numerical features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()) # Scale numeric for better convergence in Log Reg
    ])

    # Build a preprocessing step for nominal features
    nominal_transformer = Pipeline(steps=[
    ('impute missing', SimpleImputer(strategy="constant", fill_value='missing')), # 114 nulls for hispanic_origin column
    ('onehot', OneHotEncoder())
    ])

    # Build a preprocessing step for ordinal features
    ordinal_transformer = Pipeline(steps=[
    ('education_mapping', FunctionTransformer(func=education_map))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numeric_transformer, numerical_features),
        ('nominal', nominal_transformer, nominal_features),
        ('ordinal', ordinal_transformer, ordinal_features)
    ])

    # Transform X and y variables
    X = preprocessor.fit_transform(X) # Process X
    le = LabelEncoder() # Process y
    class_mapping = {l: i for i, l in enumerate(le.fit(y).classes_)} # Dictionary of class mapping for the target label
    y = le.fit_transform(y)

    # Get the column names for the transformed DataFrame
    columns = numerical_features + list(preprocessor.named_transformers_['nominal'].named_steps['onehot'].get_feature_names_out(nominal_features)) + ordinal_features

    # Convert the transformed data back to a DataFrame
    X_cleaned = pd.DataFrame(X.toarray(), columns=columns)
    y_cleaned = pd.DataFrame(y, columns=['income_group'])

    return X_cleaned, y_cleaned, class_mapping

def split_data(X: pd.DataFrame, y:pd.DataFrame, test_size: int = 0.33) -> pd.DataFrame:
    """ 
    Stratify and shuffle are set to TRUE due to imbalanced target var

    Parameters:
    - X (DataFrame): cleaned X data
    - y (DataFrame): cleaned y data
    - test_size (int): an integer indicating the portion of test sample, default is 0.33

    Returns:
    - X_train (DataFrame)
    - X_test (DataFrame)
    - y_train (DataFrame)
    - y_test (DataFrame) 
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y, shuffle=True)

    return X_train, X_test, y_train, y_test

def transform(datapath: str) -> pd.DataFrame:
    """
    End-to-end extraction, loading and preprocessing of data

    Parameters:
    - datapath (string): path to data

    Returns:
    - X_train (DataFrame)
    - X_test (DataFrame)
    - y_train (DataFrame)
    - y_test (DataFrame) 
    """
    # Extract
    df = pd.read_csv(datapath)

    # Preprocess / Transform
    X_cleaned, y_cleaned, class_mapping = preprocessing_pipeline(df)

    # Split
    X_train, X_test, y_train, y_test = split_data(X_cleaned, y_cleaned)

    return X_train, X_test, y_train, y_test

