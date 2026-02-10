import pandas as pd
import numpy as np
import re
import sys
import os
from sqlalchemy import create_engine, text

try:
    # Database connection
    conn_string = "mysql+mysqlconnector://project_user:your_password@localhost:3306/restaurant_reviews"
    engine = create_engine(conn_string)
    
    # Load data from CSV
    print("🔍 Loading data from CSV file...")
    df = pd.read_csv('Restaurant_reviews.csv')
    print(f"📊 Original dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Clean rating column
    rating_column = None
    if 'Rating' in df.columns:
        rating_column = 'Rating'
        print(f"📈 Found rating column: '{rating_column}'")
    elif 'rating' in df.columns:
        rating_column = 'rating'
        print(f"📈 Found rating column: '{rating_column}'")
    
    if rating_column:
        print("🧹 Cleaning rating column...")
        # Remove rows with empty values
        initial_count = len(df)
        df = df.dropna(subset=[rating_column])
        after_nan_drop = len(df)
        if initial_count != after_nan_drop:
            print(f"🗑️  Removed {initial_count - after_nan_drop} rows with empty rating values")
        
        # Convert to string type for processing
        df[rating_column] = df[rating_column].astype(str)
        
        # Remove known anomalies like "Like"
        anomalies = ['like', 'Like', 'likes', 'dislike', 'Dislike', 'review', 'Review', 'nan', 'NaN', 'NAN', 'unlike']
        for anomaly in anomalies:
            mask = df[rating_column].str.contains(anomaly, case=False, na=False)
            count_anomalies = mask.sum()
            if count_anomalies > 0:
                df = df[~mask]
                print(f"🧹 Removed {count_anomalies} rows with anomaly '{anomaly}'")
        
        # Extract only numeric values (including decimals)
        df[rating_column] = df[rating_column].str.extract(r'(\d+\.?\d*)')[0]
        
        # Convert to numeric format
        df[rating_column] = pd.to_numeric(df[rating_column], errors='coerce')
        
        # Remove rows with invalid values
        initial_count = len(df)
        df = df.dropna(subset=[rating_column])
        final_count = len(df)
        if initial_count != final_count:
            print(f"🗑️  Removed {initial_count - final_count} rows with invalid rating values")
        
        # Filter by range 1-5
        df = df[(df[rating_column] >= 1) & (df[rating_column] <= 5)]
        print(f"✅ {len(df)} valid records remaining after rating cleanup")
    
    # Clean text column
    text_column = None
    if 'Review' in df.columns:
        text_column = 'Review'
        print(f"📝 Found text column: '{text_column}'")
    elif 'review_text' in df.columns:
        text_column = 'review_text'
        print(f"📝 Found text column: '{text_column}'")
    
    if text_column:
        print("🧹 Cleaning text column...")
        # Remove empty and invalid texts
        initial_count = len(df)
        df = df.dropna(subset=[text_column])
        df = df[df[text_column].str.strip() != '']
        df = df[df[text_column].str.len() > 10]  # Remove too short reviews
        final_count = len(df)
        if initial_count != final_count:
            print(f"🗑️  Removed {initial_count - final_count} invalid records from column '{text_column}'")
        print(f"✅ {final_count} valid records remaining after text cleanup")
    
    # Map column names to match database schema
    column_mapping = {}
    if 'Review' in df.columns:
        column_mapping['Review'] = 'review_text'
    if 'Rating' in df.columns:
        column_mapping['Rating'] = 'rating'
    if 'Restaurant' in df.columns:
        column_mapping['Restaurant'] = 'restaurant_name'
    df = df.rename(columns=column_mapping)
    
    # Add restaurant_name column if missing
    if 'restaurant_name' not in df.columns:
        df['restaurant_name'] = 'Unknown Restaurant'
        print("🏨 Added 'restaurant_name' column with default value")
    
    # Select only required columns
    required_columns = ['review_text', 'rating', 'restaurant_name']
    available_columns = [col for col in required_columns if col in df.columns]
    if len(available_columns) < 2:  # Must have at least review_text and rating
        print("❌ Error: not enough columns found for database upload")
        print(f"Available columns: {df.columns.tolist()}")
        print("Try using a different CSV file or check its structure")
        sys.exit(1)
    
    df = df[available_columns]
    
    # Remove remaining empty values
    initial_count = len(df)
    df = df.dropna(subset=['review_text', 'rating'])
    final_count = len(df)
    if initial_count != final_count:
        print(f"🗑️  Removed {initial_count - final_count} rows with empty values in required fields")
    
    print(f"\n📤 Uploading {len(df)} records to database...")
    df.to_sql('restaurant_reviews', con=engine, if_exists='append', index=False)
    print(f"✅ Successfully uploaded {len(df)} records to database!")
    
except FileNotFoundError:
    print("❌ Error: file 'Restaurant_reviews.csv' not found in current directory")
    print("Place the CSV file in the current directory or specify the full path to the file")
    print(f"Current directory: {os.getcwd()}")
    sys.exit(1)
    
except pd.errors.EmptyDataError:
    print("❌ Error: CSV file is empty")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("Additional debugging information:")
    print(f"- Error type: {type(e).__name__}")
    print(f"- Error description: {str(e)}")
    sys.exit(1)