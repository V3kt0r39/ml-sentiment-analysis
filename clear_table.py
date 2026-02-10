from sqlalchemy import create_engine, text

# Connection parameters
db_user = "project_user"
db_password = "your_password"
db_host = "localhost"
db_port = "3306"
db_name = "restaurant_reviews"
table_name = "model_metrics"  # or any other table name

# Create connection string
conn_string = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

try:
    # Create engine
    engine = create_engine(conn_string)
    
    with engine.connect() as conn:
        # First check current record count
        count_query = text(f"SELECT COUNT(*) as count FROM {table_name}")
        result = conn.execute(count_query)
        current_count = result.fetchone()[0]
        print(f"Current number of records in table '{table_name}': {current_count}")
        
        # Clear the table
        print(f"Clearing table '{table_name}'...")
        truncate_query = text(f"TRUNCATE TABLE {table_name}")
        conn.execute(truncate_query)
        conn.commit()
        
        # Verify result
        result = conn.execute(count_query)
        new_count = result.fetchone()[0]
        print(f"Records after cleanup: {new_count}")
        print(f"✅ Table '{table_name}' successfully cleared!")
        
except Exception as e:
    print(f"❌ Error while clearing table: {e}")