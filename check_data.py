from sqlalchemy import create_engine, text
import pandas as pd

# Connection string
conn_string = "mysql+mysqlconnector://project_user:your_password@localhost:3306/restaurant_reviews"

try:
    # Create engine
    engine = create_engine(conn_string)
    print("✅ Successfully connected to the database")
    
    # Execute various checks
    with engine.connect() as conn:
        # Check total record count
        result = conn.execute(text("SELECT COUNT(*) as count FROM restaurant_reviews"))
        count = result.fetchone()[0]
        print(f"📊 Total number of records: {count}")
        
        if count > 0:
            # View first 5 records
            print("\n📋 First 5 records:")
            query = text("SELECT id, review_text, rating, restaurant_name FROM restaurant_reviews LIMIT 5")
            df = pd.read_sql(query, conn)
            print(df.to_string(index=False))
            
            # Check rating distribution
            print("\n📈 Rating distribution:")
            ratings_query = text("""
                SELECT
                    rating,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM restaurant_reviews), 1) as percentage
                FROM restaurant_reviews
                GROUP BY rating
                ORDER BY rating
            """)
            ratings_df = pd.read_sql(ratings_query, conn)
            print(ratings_df.to_string(index=False))
            
            # Check unique restaurants
            print("\n🏨 Unique restaurants:")
            restaurants_query = text("""
                SELECT
                    restaurant_name,
                    COUNT(*) as review_count
                FROM restaurant_reviews
                GROUP BY restaurant_name
                ORDER BY review_count DESC
                LIMIT 10
            """)
            restaurants_df = pd.read_sql(restaurants_query, conn)
            print(restaurants_df.to_string(index=False))
            
            print(f"\n✅ Data successfully loaded and ready for analysis!")
        else:
            print("❗ Database contains no records. Check the data loading process.")
            
except Exception as e:
    print(f"❌ Connection or query error: {e}")