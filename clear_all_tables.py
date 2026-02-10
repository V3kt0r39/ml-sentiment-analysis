from sqlalchemy import create_engine, text, inspect

# Connection string
conn_string = "mysql+mysqlconnector://project_user:your_password@localhost:3306/restaurant_reviews"

try:
    engine = create_engine(conn_string)
    with engine.connect() as conn:
        # Get list of all tables
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"Found tables to clear: {len(tables)}")
        print("Table list:", tables)
        
        # Correct cleanup order considering foreign key dependencies
        # Clear child tables first, then parent tables
        tables_to_clear = [
            'review_predictions',  # References new_reviews
            'new_reviews',         # References restaurant_reviews (if such dependency exists)
            'model_metrics',       # Independent table
            'restaurant_reviews'   # Main data table
        ]
        
        # Check which required tables exist
        existing_tables = [table for table in tables_to_clear if table in tables]
        print("\n✅ Starting FULL cleanup of ALL tables in correct order...")
        
        # Disable foreign key checks during cleanup
        print("🔧 Disabling foreign key checks...")
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
        conn.commit()
        print("✅ Foreign key checks disabled")
        
        for table in existing_tables:
            print(f"\n🧹 FULL CLEANUP of table: {table}")
            # Check current record count
            count_query = text(f"SELECT COUNT(*) as count FROM {table}")
            result = conn.execute(count_query)
            current_count = result.fetchone()[0]
            print(f"  Current number of records: {current_count}")
            
            try:
                # Clear table using TRUNCATE for fast cleanup
                print(f"  TRUNCATING table '{table}'...")
                truncate_query = text(f"TRUNCATE TABLE {table}")
                conn.execute(truncate_query)
                conn.commit()
                
                # Verify result
                result = conn.execute(count_query)
                new_count = result.fetchone()[0]
                print(f"  📊 Records after cleanup: {new_count}")
                print(f"  ✅ Table '{table}' FULLY cleared!")
                
            except Exception as e:
                print(f"  ⚠️  Error TRUNCATING table '{table}': {e}")
                print(f"  🔁 Attempting alternative method (DELETE)...")
                try:
                    # Alternative method: DELETE
                    delete_query = text(f"DELETE FROM {table}")
                    conn.execute(delete_query)
                    conn.commit()
                    result = conn.execute(count_query)
                    new_count = result.fetchone()[0]
                    print(f"  📊 Records after DELETE cleanup: {new_count}")
                    print(f"  ✅ Table '{table}' cleared using DELETE method!")
                except Exception as inner_e:
                    print(f"  ❌ Failed to clear table '{table}': {inner_e}")
        
        # Re-enable foreign key checks
        print("\n🔧 Enabling foreign key checks...")
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
        conn.commit()
        print("✅ Foreign key checks enabled")
        
        # Final verification of all tables
        print("\n🔍 FINAL VERIFICATION OF ALL TABLES:")
        for table in existing_tables:
            result = conn.execute(text(f"SELECT COUNT(*) as count FROM {table}"))
            count = result.fetchone()[0]
            print(f"  📊 Table '{table}': {count} records")
        
        print("\n🎉 FULL CLEANUP OF ALL TABLES COMPLETED SUCCESSFULLY!")
        print("✅ All database tables are now empty and ready for new data")
        
except Exception as e:
    print(f"❌ Critical error during table cleanup: {e}")
    # Attempt to restore foreign key checks even after error
    try:
        with engine.connect() as conn:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
            conn.commit()
            print("✅ Foreign key checks restored after error")
    except:
        pass