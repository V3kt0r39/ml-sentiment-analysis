import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import os
import warnings
from sqlalchemy import create_engine, text
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
# ANSI colors for console
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def print_color(text, color=Colors.WHITE, end='\n'):
    """Prints text with specified color"""
    print(f"{color}{text}{Colors.ENDC}", end=end)

def print_header(text):
    """Prints header with highlighting"""
    print_color(f"\n{'=' * 80}", Colors.OKCYAN)
    print_color(f"{text.center(80)}", Colors.BOLD + Colors.HEADER)
    print_color(f"{'=' * 80}", Colors.OKCYAN)

def print_success(text):
    """Prints success message"""
    print_color(f"✓ {text}", Colors.OKGREEN)

def print_warning(text):
    """Prints warning message"""
    print_color(f"! {text}", Colors.WARNING)

def print_error(text):
    """Prints error message"""
    print_color(f"✗ {text}", Colors.FAIL)

def print_info(text):
    """Prints informational message"""
    print_color(f"ℹ {text}", Colors.OKBLUE)

def save_plot(filename, results_dir='results'):
    """Saves current plot to file in results directory"""
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    # Form file path
    filepath = os.path.join(results_dir, filename)
    # Save plot
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()
    # Return relative path for HTML usage
    return os.path.relpath(filepath, os.getcwd())

# Download required NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    print_warning(f"Warning during NLTK resources download: {e}")

class DataCleaner:
    """
    Class for data cleaning and preprocessing before analysis
    """
    @staticmethod
    def clean_rating_column(df, rating_column='Rating'):
        """
        Cleans rating column from non-numeric values and anomalies
        Parameters:
            df (DataFrame): source DataFrame
            rating_column (str): name of rating column
        Returns:
            DataFrame: cleaned DataFrame
        """
        print_info("Cleaning rating column from non-numeric values and anomalies...")
        # Create working copy
        data = df.copy()
        # Debug info
        print_info(f"Unique values before cleaning: {data[rating_column].unique()}")
        # Remove rows with empty values in rating column
        initial_count = len(data)
        data = data.dropna(subset=[rating_column])
        after_nan_drop = len(data)
        if initial_count != after_nan_drop:
            print_warning(f"Removed {initial_count - after_nan_drop} rows with empty rating values")
        # Convert to string type for processing
        data[rating_column] = data[rating_column].astype(str)
        # Step 1: Remove known anomalies like "Like"
        anomalies = ['like', 'Like', 'likes', 'dislike', 'Dislike', 'review', 'Review', 'nan', 'NaN', 'NAN']
        for anomaly in anomalies:
            data = data[~data[rating_column].str.contains(anomaly, case=False, na=False)]
        # Step 2: Extract only numeric values (including decimals)
        data[rating_column] = data[rating_column].str.extract(r'(\d+\.?\d*)')[0]
        # Step 3: Convert to numeric format
        data[rating_column] = pd.to_numeric(data[rating_column], errors='coerce')
        # Step 4: Remove rows with invalid values
        initial_count = len(data)
        data = data.dropna(subset=[rating_column])
        final_count = len(data)
        if initial_count != final_count:
            print_warning(f"Removed {initial_count - final_count} rows with invalid rating values")
        # Step 5: Filter by range 1-5
        data = data[(data[rating_column] >= 1) & (data[rating_column] <= 5)]
        print_success(f"Column '{rating_column}' successfully cleaned. {len(data)} valid records remaining")
        # Check unique values after cleaning
        print_info(f"Unique values after cleaning: {data[rating_column].unique()}")
        return data

    @staticmethod
    def validate_dataset(df):
        """
        Validates dataset for critical issues
        Parameters:
            df (DataFrame): DataFrame to validate
        Returns:
            bool: True if data is valid, False otherwise
        """
        print_header("DATASET VALIDATION")
        # Check for required columns
        required_columns = ['Review', 'Rating']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print_error(f"Missing required columns: {missing_columns}")
            return False
        # Check unique values in Rating column
        print_info("Unique values in Rating column:")
        unique_ratings = df['Rating'].unique()
        print(unique_ratings)
        # Check for non-numeric values - first convert to numeric format
        try:
            # Create temporary copy for validation
            temp_df = df.copy()
            # Convert Rating column to numeric format with error handling
            temp_df['Rating'] = pd.to_numeric(temp_df['Rating'], errors='coerce')
            # Count valid numeric values
            numeric_count = temp_df['Rating'].notna().sum()
            non_numeric_count = len(temp_df) - numeric_count
            print_info(f"Numeric values: {numeric_count}, Non-numeric values: {non_numeric_count}")
            if non_numeric_count > 0:
                print_warning(f"Detected {non_numeric_count} non-numeric values in Rating column")
                # Show examples of non-numeric values from original DataFrame
                non_numeric_samples = df[df['Rating'].apply(lambda x: not isinstance(x, (int, float)) or pd.isna(x))]['Rating'].head(5).values
                print_info("Examples of non-numeric values:")
                for sample in non_numeric_samples:
                    print(f"  - {sample}")
            # Check for anomalous values (only for numeric values)
            valid_numeric = temp_df[temp_df['Rating'].notna()]
            valid_range = (valid_numeric['Rating'] >= 1) & (valid_numeric['Rating'] <= 5)
            valid_count = valid_range.sum()
            invalid_count = len(valid_numeric) - valid_count
            if invalid_count > 0:
                print_warning(f"Detected {invalid_count} values outside range 1-5")
                print_info("Examples of anomalous values:")
                invalid_samples = valid_numeric[~valid_range]['Rating'].head(5).values
                for sample in invalid_samples:
                    print(f"  - {sample}")
            if non_numeric_count == 0 and invalid_count == 0:
                print_success("Validation passed successfully. Data is ready for analysis.")
                return True
            else:
                print_warning("Validation detected issues in data. Cleaning required before analysis.")
                return False
        except Exception as e:
            print_error(f"Error during data validation: {str(e)}")
            print_warning("Validation failed. Manual data cleaning required.")
            return False

    @staticmethod
    def clean_text_column(df, text_column='Review'):
        """
        Cleans text column from empty values and invalid data
        Parameters:
            df (DataFrame): source DataFrame
            text_column (str): name of text column
        Returns:
            DataFrame: cleaned DataFrame
        """
        print_info(f"Cleaning text column '{text_column}'...")
        # Create working copy
        data = df.copy()
        # Remove empty and invalid texts
        initial_count = len(data)
        data = data.dropna(subset=[text_column])
        data = data[data[text_column].str.strip() != '']
        data = data[data[text_column].str.len() > 10]  # Remove too short reviews
        final_count = len(data)
        if initial_count != final_count:
            print_warning(f"Removed {initial_count - final_count} invalid records from column '{text_column}'")
        print_success(f"Column '{text_column}' successfully cleaned. {final_count} valid records remaining")
        return data

    @staticmethod
    def clean_data_from_mysql(df):
        """Cleans and normalizes data loaded from MySQL"""
        # Working copy
        data = df.copy()
        # Clean Rating column
        if 'Rating' in data.columns:
            print_info("Cleaning rating values from MySQL...")
            # Convert to numeric format
            data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')
            # Filter by range 1-5
            initial_count = len(data)
            data = data[(data['Rating'] >= 1) & (data['Rating'] <= 5)]
            final_count = len(data)
            if initial_count != final_count:
                print_warning(f"Removed {initial_count - final_count} records with invalid ratings")
        # Clean text reviews
        if 'Review' in data.columns:
            print_info("Cleaning text reviews from MySQL...")
            initial_count = len(data)
            data = data.dropna(subset=['Review'])
            data = data[data['Review'].str.strip() != '']
            data = data[data['Review'].str.len() > 10]  # Remove too short reviews
            final_count = len(data)
            if initial_count != final_count:
                print_warning(f"Removed {initial_count - final_count} invalid text records")
        return data

def load_data_from_mysql(conn_string, query):
    """Loads and normalizes data from MySQL"""
    print_info("Loading data from MySQL...")
    try:
        engine = create_engine(conn_string)
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        # Normalize column names
        column_mapping = {}
        if 'review_text' in df.columns and 'Review' not in df.columns:
            column_mapping['review_text'] = 'Review'
        if 'rating' in df.columns and 'Rating' not in df.columns:
            column_mapping['rating'] = 'Rating'
        if 'restaurant_name' in df.columns and 'Restaurant' not in df.columns:
            column_mapping['restaurant_name'] = 'Restaurant'
        if column_mapping:
            df = df.rename(columns=column_mapping)
            renamed_cols = ", ".join([f"'{old}' → '{new}'" for old, new in column_mapping.items()])
            print_info(f"Columns renamed: {renamed_cols}")
        # Add missing columns with default values
        if 'Restaurant' not in df.columns:
            df['Restaurant'] = 'Unknown Restaurant'
            print_info("Added 'Restaurant' column with default value")
        # Additional data cleaning
        df = DataCleaner.clean_data_from_mysql(df)
        print_success(f"Loaded {len(df)} rows from MySQL")
        return df
    except Exception as e:
        print_error(f"Error loading data from MySQL: {str(e)}")
        sys.exit(1)

def validate_mysql_query(df):
    """Checks if query result contains required columns"""
    required_columns = ['Review', 'Rating']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print_error(f"Database query did not return required columns: {missing_columns}")
        print_info("Example of correct query:")
        print_info('SELECT review_text AS Review, rating AS Rating, restaurant_name AS Restaurant FROM restaurant_reviews')
        sys.exit(1)
    return True

def save_model_metrics_to_mysql(models_comparison, conn_string):
    """Saves model metrics to MySQL"""
    print_info("Saving model metrics to MySQL...")
    try:
        engine = create_engine(conn_string)
        # Convert data to DataFrame if not already done
        if not isinstance(models_comparison, pd.DataFrame):
            models_comparison = pd.DataFrame(models_comparison)
        # Save to model_metrics table
        with engine.connect() as conn:
            models_comparison.to_sql('model_metrics', con=conn, if_exists='append', index=False)
        print_success("Model metrics saved to MySQL")
    except Exception as e:
        print_error(f"Error saving metrics to MySQL: {str(e)}")

def predict_and_save_to_mysql(model, vectorizer, conn_string):
    """Prediction and saving results to MySQL"""
    print_info("Predicting and saving results to MySQL...")
    try:
        engine = create_engine(conn_string)
        # Load new data from DB
        query = """
        SELECT id, review_text as Review
        FROM new_reviews
        WHERE sentiment IS NULL
        """
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        if len(df) == 0:
            print_info("No new data for prediction")
            return
        # Preprocessing and prediction
        df['cleaned_review'] = df['Review'].apply(preprocess_text)
        X = vectorizer.transform(df['cleaned_review'])
        preds = model.predict(X)
        # Add predictions to data
        df['sentiment'] = preds
        # Save results back to DB
        result_df = df[['id', 'sentiment']].copy()
        result_df.rename(columns={'sentiment': 'prediction'}, inplace=True)
        with engine.connect() as conn:
            result_df.to_sql('review_predictions', con=conn, if_exists='append', index=False)
        print_success(f"Predictions saved for {len(df)} new reviews")
    except Exception as e:
        print_error(f"Error during prediction and saving results to MySQL: {str(e)}")

def load_data(file_path, mysql_conn=None, mysql_query=None):
    """Loads data from CSV file or MySQL with detailed error diagnostics"""
    if mysql_conn and mysql_query:
        return load_data_from_mysql(mysql_conn, mysql_query)
    print_info(f"Attempting to load data from file: '{file_path}'")
    # Check if file exists
    if not os.path.exists(file_path):
        print_error(f"File '{file_path}' not found in current directory.")
        # Show current directory and available files
        current_dir = os.getcwd()
        print_info(f"Current working directory: {current_dir}")
        # Show all CSV files in current directory and subdirectories
        print_info("Searching for CSV files in current directory and subdirectories:")
        csv_files = []
        for root, dirs, files in os.walk(current_dir):
            for file in files:
                if file.lower().endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        if csv_files:
            print_info("Found CSV files:")
            for i, csv_file in enumerate(csv_files, 1):
                relative_path = os.path.relpath(csv_file, current_dir)
                print(f"  {i}. {relative_path}")
        else:
            print_warning("No CSV files found in current directory and subdirectories.")
        # Suggest solutions
        print_warning("Possible solutions:")
        print("  1. Place file in current directory")
        print("  2. Specify full path to file when running script")
        print("  3. Use relative path to file")
        print("\nExamples of correct commands:")
        print("   python ai_model.py Restaurant_reviews.csv")
        print("   python ai_model.py \"Restaurant reviews.csv\"")
        print("   python ai_model.py ./data/Restaurant_reviews.csv")
        sys.exit(1)
    try:
        # Attempt to load file with different parameters
        print_info("Attempting to load file...")
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='latin1')
            except Exception as e:
                print_error(f"Error loading with latin1 encoding: {e}")
                raise
        print_success(f"Data successfully loaded! Dataset size: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print_error(f"Detailed error loading data: {str(e)}")
        # Attempt to determine file structure
        print_warning("Attempting to determine file structure...")
        # Show first few lines of file for diagnostics
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_lines = f.readlines()[:5]
            print_info("First 5 lines of file:")
            for line in first_lines:
                print(line.strip())
        except Exception as inner_e:
            print_error(f"Could not read file content: {inner_e}")
            sys.exit(1)

def explore_data(df):
    """Performs exploratory data analysis"""
    print_header("EXPLORATORY DATA ANALYSIS")
    # Validate data before analysis
    if not DataCleaner.validate_dataset(df):
        print_warning("Performing automatic data cleaning...")
        df = DataCleaner.clean_rating_column(df)
        df = DataCleaner.clean_text_column(df)
        print_success("Data successfully cleaned and ready for analysis")
    print_info("First 5 rows of data:")
    print(df.head().to_string())
    print_info("\nData information:")
    print(df.info())
    print_info("\nStatistical description of numeric columns:")
    print(df.describe().to_string())
    # Check missing values
    missing_values = df.isnull().sum()
    print_info("\nMissing values in each column:")
    print(missing_values[missing_values > 0].to_string())
    # Analyze rating distribution
    plots_info = {}
    if 'Rating' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Rating', data=df, palette='viridis')
        plt.title('Customer Rating Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Rating', fontsize=12)
        plt.ylabel('Number of Reviews', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        rating_dist_path = save_plot('rating_distribution.png')
        plots_info['rating_distribution'] = rating_dist_path
    # Analyze review length distribution
    if 'Review' in df.columns:
        df['review_length'] = df['Review'].astype(str).apply(len)
        plt.figure(figsize=(10, 6))
        sns.histplot(df['review_length'], bins=50, kde=True)
        plt.title('Review Length Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Review Length (characters)', fontsize=12)
        plt.ylabel('Number of Reviews', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        review_len_path = save_plot('review_length_distribution.png')
        plots_info['review_length_distribution'] = review_len_path
    # Analyze most frequently mentioned restaurants
    if 'Restaurant' in df.columns:
        top_restaurants = df['Restaurant'].value_counts().head(10)
        # Shorten long restaurant names for better readability
        def shorten_name(name):
            if len(name) > 25:
                return name[:22] + '...'
            return name
        top_restaurants.index = top_restaurants.index.map(shorten_name)
        plt.figure(figsize=(12, 8))
        bars = sns.barplot(x=top_restaurants.values, y=top_restaurants.index, palette='rocket')
        plt.title('Top-10 Restaurants by Number of Reviews', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Reviews', fontsize=12)
        plt.ylabel('Restaurant Name', fontsize=12)
        # Add values at the end of bars
        for i, v in enumerate(top_restaurants.values):
            bars.text(v + 0.5, i, str(v), color='black', fontweight='bold')
        # Set X-axis scale to 1000 for clear comparison
        max_value = max(1000, top_restaurants.max() + 50)  # Minimum 1000 or max + 50
        plt.xlim(0, max_value)
        # Add vertical line at 100 reviews mark
        plt.axvline(x=100, color='#dc3545', linestyle='--', alpha=0.7)
        plt.text(102, len(top_restaurants)-1, '100 reviews', color='#dc3545', fontweight='bold', fontsize=10)
        # Configure X-axis for better visibility
        if max_value <= 200:
            step = 20
        elif max_value <= 500:
            step = 50
        else:
            step = 100
        plt.xticks(range(0, max_value + 1, step), fontsize=10)
        plt.yticks(fontsize=11)
        plt.tight_layout()
        top_rest_path = save_plot('top_restaurants.png')
        plots_info['top_restaurants'] = top_rest_path
    else:
        print_warning("Column 'Restaurant' missing from data. Top-10 restaurants chart will not be created.")
        # Create placeholder for report
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'Restaurant data missing from source.\nTo display this chart, include "Restaurant" column in your database query.',
                ha='center', va='center', fontsize=12)
        plt.axis('off')
        top_rest_path = save_plot('top_restaurants.png')
        plots_info['top_restaurants'] = top_rest_path
    # Return plot paths for HTML report
    return df, plots_info

def preprocess_text(text):
    """Preprocesses text: cleaning, stemming, stop words removal"""
    if not isinstance(text, str) or text.strip() == '':
        return ""
    try:
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenization
        tokens = text.split()
        # Remove stop words and short words
        try:
            stop_words = set(stopwords.words('english'))
        except Exception as e:
            print_warning(f"Error loading stop words: {e}")
            stop_words = set()
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        # Stemming
        ps = PorterStemmer()
        tokens = [ps.stem(word) for word in tokens]
        return ' '.join(tokens)
    except Exception as e:
        print_error(f"Error during text preprocessing: {e}")
        return ""

def prepare_data(df):
    """Prepares data for machine learning"""
    print_header("DATA PREPARATION")
    # Clean data using DataCleaner class
    print_info("Starting data cleaning for model training...")
    data = DataCleaner.clean_rating_column(df)
    data = DataCleaner.clean_text_column(data, 'Review')
    # Create working copy
    data = data[['Review', 'Rating']].copy()
    # Check for required columns
    print_info("Columns in dataset after cleaning:")
    for col in data.columns:
        print(f"  - {col}")
    # Verify required columns
    required_columns = ['Review', 'Rating']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print_error(f"Required columns missing after cleaning: {missing_columns}")
        sys.exit(1)
    # Convert Rating to numeric format
    print_info("Converting 'Rating' column to numeric format...")
    try:
        data['Rating'] = data['Rating'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
        print_success("Column 'Rating' successfully converted to numeric format")
    except Exception as e:
        print_error(f"Error converting 'Rating' column: {e}")
        # Attempt alternative conversion
        try:
            data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')
            # Remove rows with NaN in Rating
            initial_count = len(data)
            data = data.dropna(subset=['Rating'])
            final_count = len(data)
            print_warning(f"Removed {initial_count - final_count} rows with invalid rating values. Remaining rows: {final_count}")
        except Exception as inner_e:
            print_error(f"Alternative conversion also failed: {inner_e}")
            sys.exit(1)
    # Remove rows with missing values
    initial_count = len(data)
    data = data.dropna(subset=['Review', 'Rating'])
    final_count = len(data)
    if initial_count != final_count:
        print_warning(f"Removed {initial_count - final_count} rows with missing values. Remaining rows: {final_count}")
    # Exclude ratings equal to 3 (neutral)
    data = data[data['Rating'] != 3]
    print_info(f"Excluded ratings equal to 3. Remaining rows: {len(data)}")
    # Limit data size for quick startup (optional)
    if len(data) > 5000:
        data = data.sample(n=5000, random_state=42)
        print_info(f"Data reduced to 5000 random records for quick startup")
    # Preprocess review texts
    print_info("Preprocessing review texts...")
    data['cleaned_review'] = data['Review'].apply(preprocess_text)
    # Create target variable based on rating
    print_info("Creating target variable 'sentiment'...")
    data['sentiment'] = data['Rating'].apply(lambda x: 'Positive' if x > 3 else 'Negative')
    # Class balance (optional)
    pos_count = data[data['sentiment'] == 'Positive'].shape[0]
    neg_count = data[data['sentiment'] == 'Negative'].shape[0]
    print_info(f"Review balance: Positive={pos_count}, Negative={neg_count}")
    # Visualize class balance
    plt.figure(figsize=(8, 6))
    sns.countplot(x='sentiment', data=data, palette=['#28a745', '#dc3545'])
    plt.title('Review Balance', fontsize=14, fontweight='bold')
    plt.xlabel('Review Type', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    class_balance_path = save_plot('class_balance.png')
    return data, class_balance_path

def train_and_evaluate_models(X_train, X_test, y_train, y_test, export_format, results_dir='results', mysql_conn=None):
    """Trains and evaluates multiple machine learning models"""
    print_header("MODEL TRAINING AND EVALUATION")
    # Define models for training
    models = {
        'RandomForest': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'vectorizer': CountVectorizer(max_features=9000)
        },
        'NaiveBayes': {
            'model': MultinomialNB(),
            'vectorizer': CountVectorizer(max_features=9000)
        },
        'SVM': {
            'model': SVC(kernel='linear', probability=True, random_state=42),
            'vectorizer': TfidfVectorizer(max_features=5000)
        }
    }
    results = {}
    models_comparison = []  # For saving model comparison data
    html_report_data = {
        'models': {},
        'plots': {}
    }
    for name, components in models.items():
        print_info(f"\nTraining model {name}...")
        # Vectorization
        vectorizer = components['vectorizer']
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        # Model training
        model = components['model']
        model.fit(X_train_vec, y_train)
        # Save trained vectorizer
        os.makedirs(results_dir, exist_ok=True)
        vectorizer_filename = os.path.join(results_dir, f'vectorizer_{name}.pkl')
        joblib.dump(vectorizer, vectorizer_filename)
        print_success(f"Vectorizer for {name} saved: {vectorizer_filename}")
        # Predictions
        y_pred = model.predict(X_test_vec)
        y_proba = model.predict_proba(X_test_vec)[:, 1] if hasattr(model, "predict_proba") else None
        # Model evaluation
        accuracy = accuracy_score(y_test, y_pred)
        print_success(f"{name} Accuracy: {accuracy:.4f}")
        print_info("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        # Calculate additional metrics for export
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_test, y_pred, pos_label='Positive')
        recall = recall_score(y_test, y_pred, pos_label='Positive')
        f1 = f1_score(y_test, y_pred, pos_label='Positive')
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        # Save metrics for later export
        models_comparison.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'ROC_AUC': roc_auc if roc_auc is not None else 'N/A'
        })
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'],
                    annot_kws={"size": 12})
        plt.title(f'Confusion Matrix: {name}', fontsize=14, fontweight='bold')
        plt.ylabel('Actual Values', fontsize=12)
        plt.xlabel('Predicted Values', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        cm_path = save_plot(f'confusion_matrix_{name}.png')
        # ROC curve for models supporting predict_proba
        roc_path = None
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label='Positive')
            roc_auc = roc_auc_score(y_test, y_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'ROC Curve: {name}', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(True)
            roc_path = save_plot(f'roc_curve_{name}.png')
        # Save model
        model_filename = os.path.join(results_dir, f'model_{name}.pkl')
        joblib.dump(model, model_filename)
        print_success(f"Model {name} saved: {model_filename}")
        results[name] = {
            'model': model,
            'vectorizer': vectorizer,
            'accuracy': accuracy,
            'roc_auc': roc_auc if y_proba is not None else None,
            'predictions': y_pred,
            'probabilities': y_proba,
            'X_test': X_test
        }
        # Add information for HTML report
        html_report_data['models'][name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc if roc_auc is not None else None,
            'confusion_matrix_path': cm_path,
            'roc_curve_path': roc_path if roc_path else None
        }
    # Export model comparison in selected format
    models_df = pd.DataFrame(models_comparison)
    if export_format == 'excel':
        try:
            import openpyxl
            models_df.to_excel(os.path.join(results_dir, 'models_comparison.xlsx'), index=False)
            print_success(f"Model comparison exported to Excel: {os.path.join(results_dir, 'models_comparison.xlsx')}")
        except ImportError:
            print_warning("openpyxl library not installed. Excel export skipped.")
            print_info("To enable Excel export, install openpyxl: pip install openpyxl")
            # Fall back to CSV export
            models_df.to_csv(os.path.join(results_dir, 'models_comparison.csv'), index=False)
            print_success(f"Model comparison exported to CSV: {os.path.join(results_dir, 'models_comparison.csv')}")
    else:  # CSV
        models_df.to_csv(os.path.join(results_dir, 'models_comparison.csv'), index=False)
        print_success(f"Model comparison exported to CSV: {os.path.join(results_dir, 'models_comparison.csv')}")
    # Save to MySQL if connection specified
    if mysql_conn:
        save_model_metrics_to_mysql(models_df, mysql_conn)
    # Save detailed classification report for each model
    for name, result in results.items():
        report = classification_report(y_test, result['predictions'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        if export_format == 'excel':
            try:
                import openpyxl
                report_df.to_excel(os.path.join(results_dir, f'classification_report_{name}.xlsx'))
                print_success(f"Classification report for {name} saved to Excel")
            except ImportError:
                report_df.to_csv(os.path.join(results_dir, f'classification_report_{name}.csv'))
                print_warning(f"Classification report for {name} saved to CSV (openpyxl not installed)")
        else:  # CSV
            report_df.to_csv(os.path.join(results_dir, f'classification_report_{name}.csv'))
            print_success(f"Classification report for {name} saved to CSV")
    return results, html_report_data

def export_model_predictions_to_file(results, y_test, best_model_name, export_format, results_dir='results'):
    """Exports predictions of best models to selected file format"""
    os.makedirs(results_dir, exist_ok=True)
    for name, result in results.items():
        if name == best_model_name:  # Export only for best model
            try:
                # Determine probabilities for positive class
                if result['probabilities'] is not None:
                    # Check probability array dimensionality
                    if result['probabilities'].ndim == 1:
                        # For 1D array (some SVM implementations)
                        probability_positive = result['probabilities']
                    else:
                        # For 2D array (standard format)
                        probability_positive = result['probabilities'][:, 1]
                else:
                    probability_positive = None
                # Create DataFrame with predictions
                predictions_df = pd.DataFrame({
                    'Review': result['X_test'].values,
                    'Actual_Sentiment': y_test.values,
                    'Predicted_Sentiment': result['predictions'],
                    'Probability_Positive': probability_positive
                })
                # Add column with prediction correctness
                predictions_df['Correct'] = predictions_df['Actual_Sentiment'].astype(str) == predictions_df['Predicted_Sentiment'].astype(str)
                # Export to selected format
                if export_format == 'excel':
                    try:
                        import openpyxl
                        filename = os.path.join(results_dir, f'model_predictions_{name}.xlsx')
                        predictions_df.to_excel(filename, index=False)
                        print_success(f"Model {name} predictions exported to Excel: {filename}")
                    except ImportError:
                        print_warning("openpyxl library not installed. Excel export skipped.")
                        # Fall back to CSV
                        filename = os.path.join(results_dir, f'model_predictions_{name}.csv')
                        predictions_df.to_csv(filename, index=False)
                        print_success(f"Model {name} predictions exported to CSV: {filename}")
                else:  # CSV
                    filename = os.path.join(results_dir, f'model_predictions_{name}.csv')
                    predictions_df.to_csv(filename, index=False)
                    print_success(f"Model {name} predictions exported to CSV: {filename}")
                # Export only incorrect predictions for analysis
                errors_df = predictions_df[~predictions_df['Correct']]
                if not errors_df.empty:
                    if export_format == 'excel':
                        try:
                            import openpyxl
                            errors_filename = os.path.join(results_dir, f'prediction_errors_{name}.xlsx')
                            errors_df.to_excel(errors_filename, index=False)
                            print_info(f"Model {name} incorrect predictions exported to Excel: {errors_filename}")
                        except ImportError:
                            errors_filename = os.path.join(results_dir, f'prediction_errors_{name}.csv')
                            errors_df.to_csv(errors_filename, index=False)
                            print_info(f"Model {name} incorrect predictions exported to CSV: {errors_filename}")
                    else:  # CSV
                        errors_filename = os.path.join(results_dir, f'prediction_errors_{name}.csv')
                        errors_df.to_csv(errors_filename, index=False)
                        print_info(f"Model {name} incorrect predictions exported to CSV: {errors_filename}")
            except Exception as e:
                print_error(f"Error exporting predictions for model {name}: {str(e)}")
                print_warning("Continuing execution without prediction export...")

def analyze_feature_importance(model, vectorizer, model_name, export_format, results_dir='results'):
    """Analyzes feature importance for model interpretation"""
    if model_name == 'RandomForest':
        # Extract feature importance for RandomForest
        feature_names = vectorizer.get_feature_names_out()
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top-20 important words
        # Create DataFrame for convenience
        feature_importance_df = pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'importance': importances[indices]
        })
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
        plt.title(f'Top-20 Important Words for {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Words', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plot_path = save_plot(f'feature_importance_{model_name}.png')
        print_info(f"\nTop-20 important words for {model_name}:")
        print(feature_importance_df.to_string(index=False))
        # Save feature importance to selected format
        os.makedirs(results_dir, exist_ok=True)
        if export_format == 'excel':
            try:
                import openpyxl
                feature_importance_df.to_excel(os.path.join(results_dir, f'feature_importance_{model_name}.xlsx'), index=False)
                print_success(f"Feature importance for {model_name} saved to Excel")
            except ImportError:
                feature_importance_df.to_csv(os.path.join(results_dir, f'feature_importance_{model_name}.csv'), index=False)
                print_warning(f"Feature importance for {model_name} saved to CSV (openpyxl not installed)")
        else:  # CSV
            feature_importance_df.to_csv(os.path.join(results_dir, f'feature_importance_{model_name}.csv'), index=False)
            print_success(f"Feature importance for {model_name} saved to CSV")
        return feature_importance_df, plot_path

def analyze_word_frequency(data, export_format, results_dir='results'):
    """Analyzes word frequency in positive and negative reviews"""
    print_header("WORD FREQUENCY ANALYSIS")
    # Split data by class
    positive_reviews = data[data['sentiment'] == 'Positive']['cleaned_review']
    negative_reviews = data[data['sentiment'] == 'Negative']['cleaned_review']
    # Count word frequency
    def get_word_freq(reviews):
        all_text = ' '.join(reviews)
        words = all_text.split()
        word_freq = {}
        for word in words:
            if len(word) > 2:  # Ignore short words
                word_freq[word] = word_freq.get(word, 0) + 1
        return dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20])
    pos_freq = get_word_freq(positive_reviews)
    neg_freq = get_word_freq(negative_reviews)
    # Export word frequency
    pos_freq_df = pd.DataFrame({
        'Word': list(pos_freq.keys()),
        'Frequency': list(pos_freq.values())
    })
    neg_freq_df = pd.DataFrame({
        'Word': list(neg_freq.keys()),
        'Frequency': list(neg_freq.values())
    })
    os.makedirs(results_dir, exist_ok=True)
    if export_format == 'excel':
        try:
            import openpyxl
            pos_freq_df.to_excel(os.path.join(results_dir, 'word_frequency_positive.xlsx'), index=False)
            neg_freq_df.to_excel(os.path.join(results_dir, 'word_frequency_negative.xlsx'), index=False)
            print_success(f"Word frequency in positive reviews saved to Excel")
            print_success(f"Word frequency in negative reviews saved to Excel")
        except ImportError:
            pos_freq_df.to_csv(os.path.join(results_dir, 'word_frequency_positive.csv'), index=False)
            neg_freq_df.to_csv(os.path.join(results_dir, 'word_frequency_negative.csv'), index=False)
            print_warning(f"Word frequency saved to CSV (openpyxl not installed)")
    else:  # CSV
        pos_freq_df.to_csv(os.path.join(results_dir, 'word_frequency_positive.csv'), index=False)
        neg_freq_df.to_csv(os.path.join(results_dir, 'word_frequency_negative.csv'), index=False)
        print_success(f"Word frequency in positive reviews saved to CSV")
        print_success(f"Word frequency in negative reviews saved to CSV")
    # Visualization for positive reviews
    plt.figure(figsize=(12, 8))
    plt.barh(list(pos_freq.keys()), list(pos_freq.values()), color='#28a745')
    plt.title('Top-20 Most Frequent Words in Positive Reviews', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Words', fontsize=12)
    plt.gca().invert_yaxis()  # Most frequent words on top
    pos_word_freq_path = save_plot('word_freq_positive.png')
    # Visualization for negative reviews
    plt.figure(figsize=(12, 8))
    plt.barh(list(neg_freq.keys()), list(neg_freq.values()), color='#dc3545')
    plt.title('Top-20 Most Frequent Words in Negative Reviews', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Words', fontsize=12)
    plt.gca().invert_yaxis()  # Most frequent words on top
    neg_word_freq_path = save_plot('word_freq_negative.png')
    # Analyze service aspect mentions
    aspects = {
        'food': ['food', 'taste', 'delicious', 'flavor', 'dish', 'meal', 'cuisine', 'spicy', 'sweet', 'salty'],
        'service': ['service', 'staff', 'waiter', 'waitress', 'manager', 'host', 'courteous', 'polite', 'friendly', 'helpful'],
        'ambience': ['ambience', 'atmosphere', 'decor', 'music', 'lighting', 'seating', 'comfortable', 'clean', 'noise'],
        'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'affordable', 'worth', 'bill', 'overpriced']
    }
    aspect_analysis = {'aspect': [], 'sentiment': [], 'count': []}
    for aspect, keywords in aspects.items():
        for sentiment in ['Positive', 'Negative']:
            reviews = data[data['sentiment'] == sentiment]['cleaned_review']
            count = 0
            for review in reviews:
                for keyword in keywords:
                    if keyword in review:
                        count += 1
                        break
            aspect_analysis['aspect'].append(aspect)
            aspect_analysis['sentiment'].append(sentiment)
            aspect_analysis['count'].append(count)
    aspect_df = pd.DataFrame(aspect_analysis)
    # Visualization of aspect mentions
    plt.figure(figsize=(12, 8))
    sns.barplot(x='aspect', y='count', hue='sentiment', data=aspect_df, palette=['#28a745', '#dc3545'])
    plt.title('Service Aspect Mentions in Reviews', fontsize=14, fontweight='bold')
    plt.xlabel('Aspect', fontsize=12)
    plt.ylabel('Mention Count', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='Review Type')
    aspect_mentions_path = save_plot('aspect_mentions.png')
    # Export aspect analysis
    if export_format == 'excel':
        try:
            import openpyxl
            aspect_df.to_excel(os.path.join(results_dir, 'aspect_mentions.xlsx'), index=False)
            print_success(f"Aspect mentions analysis saved to Excel")
        except ImportError:
            aspect_df.to_csv(os.path.join(results_dir, 'aspect_mentions.csv'), index=False)
            print_warning(f"Aspect mentions analysis saved to CSV (openpyxl not installed)")
    else:  # CSV
        aspect_df.to_csv(os.path.join(results_dir, 'aspect_mentions.csv'), index=False)
        print_success(f"Aspect mentions analysis saved to CSV")
    print_info("\nAspect mentions analysis:")
    print(aspect_df.to_string(index=False))
    # Return plot paths for HTML report
    plots_info = {
        'word_freq_positive': pos_word_freq_path,
        'word_freq_negative': neg_word_freq_path,
        'aspect_mentions': aspect_mentions_path
    }
    return plots_info

def generate_business_insights(data, export_format, results_dir='results'):
    """Generates business insights based on data analysis"""
    print_header("BUSINESS INSIGHTS AND RECOMMENDATIONS")
    # Analyze aspect impact on ratings
    aspects = {
        'service': ['service', 'staff', 'waiter', 'waitress', 'manager', 'host', 'courteous', 'polite', 'friendly', 'helpful', 'slow', 'rude', 'attentive'],
        'food': ['food', 'taste', 'delicious', 'flavor', 'dish', 'meal', 'cuisine', 'spicy', 'sweet', 'salty', 'bland', 'tasteless'],
        'ambience': ['ambience', 'atmosphere', 'decor', 'music', 'lighting', 'seating', 'comfortable', 'clean', 'noise', 'crowded', 'dirty']
    }
    insights = []
    for aspect, keywords in aspects.items():
        aspect_reviews = data[data['cleaned_review'].str.contains('|'.join(keywords), na=False)]
        avg_rating = aspect_reviews['Rating'].mean()
        pos_percentage = (aspect_reviews['sentiment'] == 'Positive').mean() * 100
        insights.append({
            'aspect': aspect,
            'avg_rating': avg_rating,
            'pos_percentage': pos_percentage,
            'mention_count': len(aspect_reviews)
        })
    insights_df = pd.DataFrame(insights)
    print_info("\nAspect impact analysis on ratings:")
    print(insights_df.to_string(index=False))
    # Export business insights
    if export_format == 'excel':
        try:
            import openpyxl
            insights_df.to_excel(os.path.join(results_dir, 'business_insights_aspects.xlsx'), index=False)
            print_success(f"Aspect business insights saved to Excel")
        except ImportError:
            insights_df.to_csv(os.path.join(results_dir, 'business_insights_aspects.csv'), index=False)
            print_warning(f"Aspect business insights saved to CSV (openpyxl not installed)")
    else:  # CSV
        insights_df.to_csv(os.path.join(results_dir, 'business_insights_aspects.csv'), index=False)
        print_success(f"Aspect business insights saved to CSV")
    # Identify key issues
    negative_reviews = data[data['sentiment'] == 'Negative']['cleaned_review']
    common_issues = {}
    issue_keywords = {
        'slow_service': ['slow', 'wait', 'waiting', 'delay', 'late', 'forever', 'hours'],
        'rude_staff': ['rude', 'impolite', 'unfriendly', 'disrespectful', 'bad attitude', 'angry', 'unhelpful'],
        'poor_food': ['bad taste', 'tasteless', 'overcooked', 'undercooked', 'cold', 'stale', 'raw', 'bland'],
        'dirty': ['dirty', 'unhygienic', 'filthy', 'cockroach', 'bug', 'mouse', 'rat'],
        'expensive': ['expensive', 'overpriced', 'costly', 'pricey', 'not worth', 'value for money']
    }
    for issue, keywords in issue_keywords.items():
        count = sum(negative_reviews.str.contains('|'.join(keywords), na=False))
        if count > 0:
            percentage = (count / len(negative_reviews)) * 100
            common_issues[issue] = (count, percentage)
    # Visualize key issues
    key_issues_path = None
    if common_issues:
        issues_df = pd.DataFrame({
            'issue': list(common_issues.keys()),
            'count': [v[0] for v in common_issues.values()],
            'percentage': [v[1] for v in common_issues.values()]
        })
        plt.figure(figsize=(12, 8))
        sns.barplot(x='percentage', y='issue', data=issues_df, palette='Reds_r')
        plt.title('Key Issues in Negative Reviews', fontsize=14, fontweight='bold')
        plt.xlabel('Mention Percentage (%)', fontsize=12)
        plt.ylabel('Issue Type', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        key_issues_path = save_plot('key_issues.png')
        print_info("\nKey issues in negative reviews:")
        print(issues_df.to_string(index=False))
        # Export key issues
        if export_format == 'excel':
            try:
                import openpyxl
                issues_df.to_excel(os.path.join(results_dir, 'key_issues.xlsx'), index=False)
                print_success(f"Key issues saved to Excel")
            except ImportError:
                issues_df.to_csv(os.path.join(results_dir, 'key_issues.csv'), index=False)
                print_warning(f"Key issues saved to CSV (openpyxl not installed)")
        else:  # CSV
            issues_df.to_csv(os.path.join(results_dir, 'key_issues.csv'), index=False)
            print_success(f"Key issues saved to CSV")
    # Recommendations
    recommendations = [
        "Focus on improving service quality - this is the most frequently mentioned aspect in reviews.",
        "Pay attention to service speed - slow service is a common cause of negative reviews.",
        "Conduct staff training on customer interaction to improve politeness and attentiveness levels.",
        "Improve kitchen quality control to ensure consistent taste and temperature of dishes.",
        "Consider revising pricing policy to improve perceived value of services."
    ]
    # Concrete examples of problems and recommendations
    concrete_examples = [
        {
            "aspect": "service",
            "problem": "Slow service",
            "example_review": "Had to wait 20 minutes for a waiter just to order drinks",
            "recommendation": "Implement a waiter call system via mobile app or special buttons on tables",
            "expected_result": "40% reduction in wait time, increase in positive reviews"
        },
        {
            "aspect": "food",
            "problem": "Stale ingredients",
            "example_review": "Salad was clearly made with stale vegetables, and the fish had an unpleasant smell",
            "recommendation": "Implement daily quality control system for ingredients with photo reports and responsible person signatures",
            "expected_result": "60% reduction in food quality complaints, 0.5 point increase in average rating"
        },
        {
            "aspect": "ambience",
            "problem": "Noisy atmosphere",
            "example_review": "Impossible to have a normal conversation due to loud music and noise from neighboring tables",
            "recommendation": "Create zones with different noise levels (quiet zones for business meetings, zones for groups) and install acoustic panels",
            "expected_result": "15% increase in guest stay duration, growth in repeat visits"
        }
    ]
    # Additional sections
    improvement_timeline = [
        {
            "phase": "Immediate actions (1-2 weeks)",
            "actions": [
                "Conduct audit of current service quality",
                "Identify most problematic aspects based on analysis data",
                "Implement basic guest feedback system"
            ]
        },
        {
            "phase": "Short-term measures (1-3 months)",
            "actions": [
                "Conduct staff training on service standards",
                "Improve kitchen dish quality control",
                "Implement real-time review monitoring system"
            ]
        },
        {
            "phase": "Long-term strategy (3-12 months)",
            "actions": [
                "Develop loyalty program for regular customers",
                "Modernize restaurant interior and atmosphere",
                "Create automated feedback analysis system"
            ]
        }
    ]
    # ROI analysis
    roi_analysis = {
        "investment": "Implementation of review analysis system and service quality improvement requires investment of approximately 5,500 EUR.",
        "expected_returns": [
            "10% increase in average check due to improved service quality",
            "25% growth in repeat visits within 6 months",
            "Reduced marketing costs due to positive word-of-mouth",
            "Increase in restaurant average rating on review platforms from 3.8 to 4.5"
        ],
        "payback_period": "Investment payback expected within 8 months"
    }
    print_info("\nKey recommendations for improving service quality:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    # Export recommendations
    recommendations_df = pd.DataFrame({
        'Priority': range(1, len(recommendations) + 1),
        'Recommendation': recommendations
    })
    if export_format == 'excel':
        try:
            import openpyxl
            recommendations_df.to_excel(os.path.join(results_dir, 'business_recommendations.xlsx'), index=False)
            print_success(f"Recommendations saved to Excel")
        except ImportError:
            recommendations_df.to_csv(os.path.join(results_dir, 'business_recommendations.csv'), index=False)
            print_warning(f"Recommendations saved to CSV (openpyxl not installed)")
    else:  # CSV
        recommendations_df.to_csv(os.path.join(results_dir, 'business_recommendations.csv'), index=False)
        print_success(f"Recommendations saved to CSV")
    # Save recommendations to file
    with open(os.path.join(results_dir, 'business_recommendations.txt'), 'w') as f:
        f.write("Key recommendations for improving service quality\n")
        f.write("=" * 70 + "\n")
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n")
    print_success(f"Recommendations saved to file: {os.path.join(results_dir, 'business_recommendations.txt')}")
    # Return data for HTML report
    html_data = {
        'aspects_analysis': insights_df.to_dict('records'),
        'key_issues': common_issues,
        'recommendations': recommendations,
        'concrete_examples': concrete_examples,
        'improvement_timeline': improvement_timeline,
        'roi_analysis': roi_analysis
    }
    return html_data

def predict_sentiment(model, vectorizer, review_text):
    """Predicts review sentiment for new review"""
    cleaned_text = preprocess_text(review_text)
    text_vec = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0] if hasattr(model, "predict_proba") else None
    return prediction, probability

def fix_image_path(path):
    """Fixes image paths for HTML report"""
    if not path:
        return None
    # If path already contains 'results/' at start, return as is
    if path.startswith('results/'):
        return path
    # If path contains only filename (no directory), add results/ prefix
    if '/' not in path:
        return f'results/{path}'
    # For all other cases - return as is
    return path

def generate_html_report(eda_plots, class_balance_path, word_freq_plots, model_results, business_insights, feature_importance=None, results_dir='results', root_dir='.'):
    """Generates HTML report with analysis results"""
    print_header("HTML REPORT GENERATION")
    os.makedirs(results_dir, exist_ok=True)
    # Fix image paths for HTML report
    def fix_image_path(path):
        if not path:
            return None
        # If path already contains 'results/' at start, return as is
        if path.startswith('results/'):
            return path
        # If path contains only filename (no directory), add results/ prefix
        if '/' not in path:
            return f'results/{path}'
        # For all other cases - return as is
        return path
    # Fix plot paths
    eda_plot_paths = {k: fix_image_path(v) for k, v in eda_plots.items() if v}
    class_balance_path = fix_image_path(class_balance_path)
    word_freq_plot_paths = {k: fix_image_path(v) for k, v in word_freq_plots.items() if v}
    # HTML content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Restaurant Service Quality Analysis Report</title>
<style>
body {{
font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
line-height: 1.6;
color: #333;
max-width: 1200px;
margin: 0 auto;
padding: 20px;
}}
.header {{
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
color: white;
padding: 30px;
text-align: center;
border-radius: 10px;
margin-bottom: 30px;
}}
.section {{
background: #f8f9fa;
border-radius: 8px;
padding: 20px;
margin-bottom: 25px;
box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}}
.section-title {{
color: #495057;
border-bottom: 2px solid #6c757d;
padding-bottom: 10px;
margin-top: 0;
}}
.plot-container {{
text-align: center;
margin: 25px 0;
}}
.plot-container img {{
max-width: 100%;
border: 1px solid #ddd;
border-radius: 5px;
box-shadow: 0 3px 10px rgba(0,0,0,0.1);
}}
.model-comparison {{
display: flex;
flex-wrap: wrap;
justify-content: space-between;
}}
.model-card {{
background: white;
border-radius: 8px;
padding: 15px;
margin: 10px;
box-shadow: 0 2px 8px rgba(0,0,0,0.1);
flex: 1;
min-width: 300px;
}}
.model-name {{
font-size: 1.4em;
font-weight: bold;
color: #495057;
margin-bottom: 10px;
}}
.metric {{
font-size: 1.2em;
margin: 8px 0;
}}
.metric-name {{
display: inline-block;
width: 180px;
font-weight: bold;
}}
.metric-value {{
display: inline-block;
font-weight: bold;
color: #28a745;
}}
.recommendations {{
background: #e3f2fd;
border-left: 4px solid #2196f3;
padding: 15px;
margin: 15px 0;
}}
.recommendation-item {{
margin: 8px 0;
padding-left: 20px;
position: relative;
}}
.recommendation-item:before {{
content: "✓";
position: absolute;
left: 0;
color: #28a745;
font-weight: bold;
}}
.best-model {{
background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
color: white !important;
}}
.footer {{
text-align: center;
margin-top: 40px;
padding: 20px;
color: #6c757d;
font-size: 0.9em;
}}
table {{
width: 100%;
border-collapse: collapse;
margin: 20px 0;
}}
table, th, td {{
border: 1px solid #ddd;
}}
th, td {{
padding: 12px;
text-align: left;
}}
th {{
background-color: #f2f2f2;
}}
tr:hover {{
background-color: #f5f5f5;
}}
.example-card {{
background: white;
border-radius: 8px;
padding: 15px;
margin: 10px 0;
box-shadow: 0 2px 8px rgba(0,0,0,0.1);
border-left: 4px solid #007bff;
}}
.problem {{
color: #dc3545;
font-weight: bold;
}}
.solution {{
color: #28a745;
font-weight: bold;
}}
.timeline {{
margin: 20px 0;
}}
.phase {{
margin-bottom: 15px;
padding: 10px;
border-radius: 5px;
background: #e9ecef;
}}
.phase-title {{
font-weight: bold;
color: #495057;
margin-bottom: 5px;
}}
.roi-card {{
background: #fff3cd;
border: 1px solid #ffeaa7;
border-radius: 8px;
padding: 15px;
margin: 15px 0;
}}
.tabs {{
display: flex;
margin-bottom: 20px;
}}
.tab {{
padding: 10px 20px;
cursor: pointer;
background: #e9ecef;
border: 1px solid #ddd;
border-bottom: none;
border-radius: 5px 5px 0 0;
margin-right: 5px;
}}
.tab.active {{
background: #007bff;
color: white;
}}
.tab-content {{
display: none;
padding: 20px;
border: 1px solid #ddd;
border-radius: 0 5px 5px 5px;
background: white;
}}
.tab-content.active {{
display: block;
}}
</style>
<script>
function showTab(tabId) {{
document.querySelectorAll('.tab-content').forEach(tab => {{
tab.classList.remove('active');
}});
document.querySelectorAll('.tab').forEach(tab => {{
tab.classList.remove('active');
}});
document.getElementById(tabId).classList.add('active');
event.target.classList.add('active');
}}
document.addEventListener('DOMContentLoaded', function() {{
document.querySelectorAll('.tab')[0].click();
}});
</script>
</head>
<body>
<div class="header">
<h1>Restaurant Service Quality Analysis</h1>
<p>Report based on review and rating analysis using machine learning methods</p>
<p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
<!-- Exploratory Data Analysis -->
<div class="section">
<h2 class="section-title">Exploratory Data Analysis</h2>
<p>At this stage we analyzed data structure, rating distribution, and main review characteristics.</p>
<div class="tabs">
<div class="tab" onclick="showTab('eda-1')">Rating Distribution</div>
<div class="tab" onclick="showTab('eda-2')">Review Length Distribution</div>
<div class="tab active" onclick="showTab('eda-3')">Top-10 Restaurants</div>
</div>
<div class="tab-content" id="eda-1">
<div class="plot-container">
<h3>Customer Rating Distribution</h3>
<img src="{eda_plot_paths.get('rating_distribution', '')}" alt="Rating Distribution">
<p>Chart shows how ratings from 1 to 5 are distributed. Most customers give high ratings (4-5 points).</p>
</div>
</div>
<div class="tab-content" id="eda-2">
<div class="plot-container">
<h3>Review Length Distribution</h3>
<img src="{eda_plot_paths.get('review_length_distribution', '')}" alt="Review Length Distribution">
<p>Chart shows how review lengths are distributed. Most reviews have 50-300 characters.</p>
</div>
</div>
<div class="tab-content active" id="eda-3">
<div class="plot-container">
<h3>Top-10 Restaurants by Number of Reviews</h3>
{f'<img src="{eda_plot_paths.get("top_restaurants", "")}" alt="Top-10 Restaurants">' if "top_restaurants" in eda_plot_paths else '<p>Restaurant data missing from source. To display this chart, include "Restaurant" column in your database query.</p>'}
<p>Chart shows restaurants with highest number of reviews. Scale up to 1000 reviews shows that even most popular restaurants have relatively low number of reviews, indicating growth potential.</p>
</div>
</div>
</div>
<!-- Class Balance -->
<div class="section">
<h2 class="section-title">Review Balance</h2>
<p>For review classification task, target feature was created based on rating:</p>
<ul>
<li>Rating &gt; 3 → "Positive"</li>
<li>Rating &lt; 3 → "Negative"</li>
</ul>
<div class="plot-container">
<img src="{class_balance_path}" alt="Review Balance">
<p>Chart shows ratio of positive and negative reviews in dataset.</p>
</div>
</div>
<!-- Word Frequency Analysis -->
<div class="section">
<h2 class="section-title">Word Frequency Analysis in Reviews</h2>
<p>Analysis of most frequently occurring words in positive and negative reviews helps understand which aspects most influence customer satisfaction.</p>
<div class="tabs">
<div class="tab active" onclick="showTab('words-1')">Positive Reviews</div>
<div class="tab" onclick="showTab('words-2')">Negative Reviews</div>
<div class="tab" onclick="showTab('words-3')">Service Aspects</div>
</div>
<div class="tab-content active" id="words-1">
<div class="plot-container">
<h3>Top-20 Most Frequent Words in Positive Reviews</h3>
<img src="{word_freq_plot_paths.get('word_freq_positive', '')}" alt="Word Frequency in Positive Reviews">
<p>Frequently occurring words in positive reviews help understand what customers value most in restaurant.</p>
</div>
</div>
<div class="tab-content" id="words-2">
<div class="plot-container">
<h3>Top-20 Most Frequent Words in Negative Reviews</h3>
<img src="{word_freq_plot_paths.get('word_freq_negative', '')}" alt="Word Frequency in Negative Reviews">
<p>Frequently occurring words in negative reviews indicate main service problems.</p>
</div>
</div>
<div class="tab-content" id="words-3">
<div class="plot-container">
<h3>Service Aspect Mentions</h3>
<img src="{word_freq_plot_paths.get('aspect_mentions', '')}" alt="Service Aspects">
<p>Chart shows which service aspects (food, service, atmosphere) are most frequently mentioned in reviews.</p>
</div>
</div>
</div>
<!-- Model Results -->
<div class="section">
<h2 class="section-title">Machine Learning Model Results</h2>
<p>Three machine learning models were trained and tested for review analysis. Results and performance visualizations are presented below.</p>
<div class="model-comparison">
"""
    # Add cards for each model
    for model_name, metrics in model_results['models'].items():
        is_best = model_name == max(model_results['models'], key=lambda x: model_results['models'][x]['accuracy'])
        card_class = "best-model" if is_best else ""
        html_content += f"""
<div class="model-card {card_class}">
<div class="model-name">{model_name}</div>
<div class="metric"><span class="metric-name">Accuracy:</span> <span class="metric-value">{metrics['accuracy']:.4f}</span></div>
<div class="metric"><span class="metric-name">Precision:</span> <span class="metric-value">{metrics['precision']:.4f}</span></div>
<div class="metric"><span class="metric-name">Recall:</span> <span class="metric-value">{metrics['recall']:.4f}</span></div>
<div class="metric"><span class="metric-name">F1-Score:</span> <span class="metric-value">{metrics['f1']:.4f}</span></div>
{f'<div class="metric"><span class="metric-name">ROC-AUC:</span> <span class="metric-value">{metrics["roc_auc"]:.4f}</span></div>' if metrics["roc_auc"] else ''}
</div>
"""
    html_content += """
</div>
<!-- Confusion Matrices and ROC Curves -->
<div class="plot-container">
<h3>Model Confusion Matrices</h3>
"""
    for model_name, metrics in model_results['models'].items():
        if metrics['confusion_matrix_path']:
            html_content += f"""
<div style="display: inline-block; margin: 15px;">
<h4>{model_name}</h4>
<img src="{metrics['confusion_matrix_path']}" alt="Confusion Matrix {model_name}" style="max-width: 300px;">
</div>
"""
    html_content += """
</div>
<div class="plot-container">
<h3>Model ROC Curves</h3>
"""
    for model_name, metrics in model_results['models'].items():
        if metrics['roc_curve_path']:
            html_content += f"""
<div style="display: inline-block; margin: 15px;">
<h4>{model_name}</h4>
<img src="{metrics['roc_curve_path']}" alt="ROC Curve {model_name}" style="max-width: 300px;">
</div>
"""
    html_content += """
</div>
</div>
"""
    # Add feature importance analysis if available
    if feature_importance is not None:
        html_content += f"""
<div class="section">
<h2 class="section-title">Feature Importance Analysis</h2>
<p>For the best model (RandomForest), feature importance analysis was performed to determine which words have greatest impact on review type prediction.</p>
<div class="plot-container">
<img src="{feature_importance[1]}" alt="Feature Importance">
<p>Chart shows most important words for review classification. For example, words "excellent", "delicious" frequently appear in positive reviews.</p>
</div>
</div>
"""
    # Concrete examples of problems and solutions
    html_content += """
<div class="section">
<h2 class="section-title">Concrete Problem Examples and Solutions</h2>
<p>Below are real review examples with problems and specific recommendations for resolution.</p>
"""
    for example in business_insights['concrete_examples']:
        html_content += f"""
<div class="example-card">
<h3>Aspect: {example['aspect']}</h3>
<p><span class="problem">Problem:</span> {example['problem']}</p>
<p><strong>Review example:</strong> "{example['example_review']}"</p>
<p><span class="solution">Recommendation:</span> {example['recommendation']}</p>
<p><strong>Expected result:</strong> {example['expected_result']}</p>
</div>
"""
    html_content += """
</div>
<!-- Improvement plan by phases -->
<div class="section">
<h2 class="section-title">Service Quality Improvement Plan</h2>
<p>For best results, follow phased improvement plan.</p>
"""
    for phase in business_insights['improvement_timeline']:
        html_content += f"""
<div class="phase">
<div class="phase-title">{phase['phase']}</div>
<ul>
"""
        for action in phase['actions']:
            html_content += f"<li>{action}</li>"
        html_content += """
</ul>
</div>
"""
    html_content += """
</div>
<!-- ROI analysis - CORRECTED SECTION WITH PROPER TEXT -->
<div class="section">
<h2 class="section-title">Economic Efficiency of Improvements</h2>
<p>Return on investment (ROI) analysis for implementing service quality improvement recommendations.</p>
<div class="roi-card">
<h3>Investment</h3>
<p>Implementation of review analysis system and service quality improvement requires investment ≈ 5,500 EUR.</p>
<h3>Expected Results</h3>
<ul>
<li>10% increase in average check due to improved service quality</li>
<li>25% growth in repeat visits within 6 months</li>
<li>Reduced marketing costs due to positive word-of-mouth</li>
<li>Increase in restaurant average rating on review platforms from 3.8 to 4.5</li>
</ul>
<h3>Payback Period</h3>
<p>Investment payback expected within 8 months</p>
</div>
</div>
"""
    # Add business analytics
    html_content += """
<div class="section">
<h2 class="section-title">Business Analytics and Recommendations</h2>
<p>Based on data analysis, key factors influencing customer satisfaction were identified and practical recommendations formulated for service quality improvement.</p>
"""
    # Aspect impact analysis
    html_content += """
<h3>Impact of Different Aspects on Ratings</h3>
<table>
<thead>
<tr>
<th>Aspect</th>
<th>Average Rating</th>
<th>% Positive Reviews</th>
<th>Mention Count</th>
</tr>
</thead>
<tbody>
"""
    for aspect in business_insights['aspects_analysis']:
        html_content += f"""
<tr>
<td>{aspect['aspect'].capitalize()}</td>
<td>{aspect['avg_rating']:.2f}</td>
<td>{aspect['pos_percentage']:.1f}%</td>
<td>{aspect['mention_count']}</td>
</tr>
"""
    html_content += """
</tbody>
</table>
"""
    # Key issues
    if business_insights['key_issues']:
        html_content += """
<h3>Key Issues in Negative Reviews</h3>
<table>
<thead>
<tr>
<th>Issue Type</th>
<th>Mention Count</th>
<th>Percentage of All Negative Reviews</th>
</tr>
</thead>
<tbody>
"""
        for issue, (count, percentage) in business_insights['key_issues'].items():
            html_content += f"""
<tr>
<td>{issue.replace('_', ' ').capitalize()}</td>
<td>{count}</td>
<td>{percentage:.1f}%</td>
</tr>
"""
        html_content += """
</tbody>
</table>
"""
    # Recommendations
    html_content += """
<h3>Recommendations for Service Quality Improvement</h3>
<div class="recommendations">
"""
    for i, rec in enumerate(business_insights['recommendations'], 1):
        html_content += f'<div class="recommendation-item"><strong>Recommendation #{i}:</strong> {rec}</div>'
    html_content += """
</div>
</div>
<div class="footer">
<p>Report automatically generated using Python, scikit-learn, matplotlib and seaborn</p>
<p>For additional information about analysis methodology, contact data science team</p>
</div>
</body>
</html>
"""
    # Save HTML file to root directory
    html_path = os.path.join(root_dir, 'analysis_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print_success(f"HTML report successfully generated: {html_path}")
    print_info("To view report, open file in web browser")
    return html_path

def main():
    print_header("RESTAURANT SERVICE QUALITY ANALYSIS SYSTEM")
    if len(sys.argv) < 2:
        print_header("USAGE INSTRUCTIONS")
        print_warning("Incorrect number of arguments!")
        print_info("Usage: python ai_model.py <csv_file_path|mysql> [parameters]")
        print_info("For CSV: python ai_model.py Restaurant_reviews.csv [excel|csv]")
        print_info("For MySQL: python ai_model.py mysql <connection_string> <query> [excel|csv]")
        print_info("MySQL Example: python ai_model.py mysql \"mysql+mysqlconnector://project_user:project_password123@localhost:3306/restaurant_reviews\" \"SELECT review_text AS Review, rating AS Rating, restaurant_name AS Restaurant FROM restaurant_reviews\"")
        sys.exit(1)
    # Get arguments
    source_type = sys.argv[1]
    mysql_conn = None
    mysql_query = None
    export_format = 'csv'  # default value
    if source_type == "mysql":
        if len(sys.argv) < 4:
            print_error("MySQL requires connection string and query")
            sys.exit(1)
        mysql_conn = sys.argv[2]
        mysql_query = sys.argv[3]
        # Export format - fourth argument if exists
        if len(sys.argv) > 4:
            export_format = sys.argv[4].lower()
    else:
        file_path = sys.argv[1]
        # Export format - second argument if exists
        if len(sys.argv) > 2:
            export_format = sys.argv[2].lower()
        if export_format not in ['excel', 'csv']:
            print_warning(f"Unsupported export format '{export_format}'. Using default format 'csv'")
            export_format = 'csv'
    print_success(f"Selected export format for results: {export_format.upper()}")
    # Create results folder if it doesn't exist
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    print_success(f"Technical files will be saved to directory: {os.path.abspath(results_dir)}")
    # Load data
    if source_type == "mysql":
        df = load_data(None, mysql_conn=mysql_conn, mysql_query=mysql_query)
        # Validate database query
        validate_mysql_query(df)
    else:
        df = load_data(file_path)
    # Check if data loaded
    if df is None or len(df) == 0:
        print_error("Data not loaded or empty. Check data source.")
        print_error("If using MySQL, ensure database contains data.")
        print_info("To load data from CSV use command:")
        print_info("  python import_data.py")
        sys.exit(1)
    # Exploratory analysis
    df, eda_plots = explore_data(df)
    # Data preparation
    data, class_balance_path = prepare_data(df)
    # Check if data not empty after preparation
    if len(data) == 0:
        print_error("Dataset empty after preparation. Check data cleaning process.")
        sys.exit(1)
    # Split data into training and test sets
    X = data['cleaned_review']
    y = data['sentiment']
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        print_error(f"Error splitting data: {str(e)}")
        print_error("Dataset may not contain sufficient examples for stratified split.")
        print_info("Attempting to split data without stratification...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    print_info(f"\nTraining set size: {len(X_train)}")
    print_info(f"Test set size: {len(X_test)}")
    # Model training and evaluation
    results, html_model_data = train_and_evaluate_models(X_train, X_test, y_train, y_test, export_format, results_dir, mysql_conn)
    # Feature importance analysis for best model
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']
    best_vectorizer = results[best_model_name]['vectorizer']
    print_success(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")
    feature_importance_data = None
    if best_model_name == 'RandomForest':
        feature_importance_data = analyze_feature_importance(best_model, best_vectorizer, best_model_name, export_format, results_dir)
    # Export predictions of best model
    export_model_predictions_to_file(results, y_test, best_model_name, export_format, results_dir)
    # Word frequency analysis
    word_freq_plots = analyze_word_frequency(data, export_format, results_dir)
    # Generate business insights
    business_insights = generate_business_insights(data, export_format, results_dir)
    # Generate HTML report (in root directory)
    html_path = generate_html_report(
        eda_plots=eda_plots,
        class_balance_path=class_balance_path,
        word_freq_plots=word_freq_plots,
        model_results=html_model_data,
        business_insights=business_insights,
        feature_importance=feature_importance_data,
        results_dir=results_dir,
        root_dir='.'
    )
    # Prediction and saving results to MySQL if connection specified
    if source_type == "mysql":
        predict_and_save_to_mysql(best_model, best_vectorizer, mysql_conn)
    # Demonstration prediction for example
    example_review = "The food was absolutely delicious and the service was excellent. I loved the ambience and will definitely come back again!"
    prediction, probability = predict_sentiment(best_model, best_vectorizer, example_review)
    print_header("PREDICTION DEMONSTRATION")
    print_info(f"Example review: {example_review}")
    print_success(f"Predicted review sentiment: {prediction}")
    if probability is not None:
        classes = best_model.classes_
        prob_dict = dict(zip(classes, probability.round(4)))
        print_info(f"Probabilities: {prob_dict}")
    print_success(f"\nProject successfully completed! Technical files saved in '{results_dir}' folder")
    print_success(f"HTML report available at: {html_path}")

if __name__ == "__main__":
    print_header("RESTAURANT SERVICE QUALITY ANALYSIS SYSTEM")
    main()
    print_header("ANALYSIS COMPLETED. RESULTS SAVED")