from sqlalchemy import create_engine, text

try:
    # Connection string to MariaDB/MySQL
    conn_string = "mysql+mysqlconnector://project_user:your_password@localhost:3306/restaurant_reviews"
    
    # Create engine
    engine = create_engine(conn_string)
    
    # Test connection
    with engine.connect() as conn:
        print("✅ Successfully connected to the database!")
        
        # Test query using text() for SQLAlchemy 2.0 compatibility
        result = conn.execute(text("SELECT DATABASE()"))
        db_name = result.fetchone()[0]
        print(f"Current database: {db_name}")
        
        # Check existing tables
        tables_result = conn.execute(text("SHOW TABLES"))
        tables = tables_result.fetchall()
        print("Existing tables:")
        for table in tables:
            print(f"  - {table[0]}")
            
except Exception as e:
    print(f"❌ Connection error: {e}")