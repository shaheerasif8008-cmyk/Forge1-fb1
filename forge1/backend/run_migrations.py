#!/usr/bin/env python3
"""
Database Migration Runner for Forge1

Runs all database migrations to set up the AI Employee Lifecycle schema.
"""

import asyncio
import os
import sys
from pathlib import Path
import asyncpg
from datetime import datetime

# Add the forge1 backend to the Python path
sys.path.insert(0, str(Path(__file__).parent))


async def run_migrations():
    """Run all database migrations"""
    print("🗄️  Starting Forge1 Database Migrations...")
    
    # Database connection parameters
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', 5432)),
        'database': os.getenv('POSTGRES_DB', 'forge1'),
        'user': os.getenv('POSTGRES_USER', 'forge1_user'),
        'password': os.getenv('POSTGRES_PASSWORD', 'forge1_db_pass')
    }
    
    print(f"📡 Connecting to database: {db_config['host']}:{db_config['port']}/{db_config['database']}")
    
    try:
        # Connect to database
        conn = await asyncpg.connect(**db_config)
        print("✅ Database connection established")
        
        # Get list of migration files
        migrations_dir = Path(__file__).parent / "migrations"
        migration_files = sorted(migrations_dir.glob("*.sql"))
        
        print(f"📁 Found {len(migration_files)} migration files")
        
        # Create migrations tracking table if it doesn't exist
        await conn.execute("""
            CREATE SCHEMA IF NOT EXISTS forge1_migrations;
            CREATE TABLE IF NOT EXISTS forge1_migrations.schema_migrations (
                id SERIAL PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL,
                applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        # Check which migrations have already been applied
        applied_migrations = await conn.fetch("""
            SELECT filename FROM forge1_migrations.schema_migrations
        """)
        applied_filenames = {row['filename'] for row in applied_migrations}
        
        # Run each migration
        for migration_file in migration_files:
            filename = migration_file.name
            
            if filename in applied_filenames:
                print(f"⏭️  Skipping {filename} (already applied)")
                continue
            
            print(f"🔄 Running migration: {filename}")
            
            try:
                # Read migration file
                migration_sql = migration_file.read_text()
                
                # Execute migration
                await conn.execute(migration_sql)
                
                # Record migration as applied
                await conn.execute("""
                    INSERT INTO forge1_migrations.schema_migrations (filename)
                    VALUES ($1)
                """, filename)
                
                print(f"✅ Migration {filename} completed successfully")
                
            except Exception as e:
                print(f"❌ Migration {filename} failed: {e}")
                raise
        
        # Verify schema setup
        print(f"\n🔍 Verifying schema setup...")
        
        # Check if main tables exist
        tables_to_check = [
            'forge1_employees.clients',
            'forge1_employees.employees', 
            'forge1_employees.employee_interactions',
            'forge1_employees.employee_memory_summaries',
            'forge1_employees.employee_knowledge'
        ]
        
        for table in tables_to_check:
            exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = $1 AND table_name = $2
                )
            """, table.split('.')[0], table.split('.')[1])
            
            if exists:
                print(f"  ✅ Table {table} exists")
            else:
                print(f"  ❌ Table {table} missing")
        
        # Check if extensions are installed
        extensions = await conn.fetch("""
            SELECT extname FROM pg_extension 
            WHERE extname IN ('uuid-ossp', 'pgcrypto', 'vector')
        """)
        
        extension_names = {row['extname'] for row in extensions}
        
        for ext in ['uuid-ossp', 'pgcrypto']:
            if ext in extension_names:
                print(f"  ✅ Extension {ext} installed")
            else:
                print(f"  ⚠️  Extension {ext} not found (may need manual installation)")
        
        if 'vector' in extension_names:
            print(f"  ✅ Extension vector installed (pgvector for embeddings)")
        else:
            print(f"  ⚠️  Extension vector not found (pgvector needed for embeddings)")
            print(f"     Note: Vector operations will use JSONB fallback")
        
        print(f"\n🎉 Database migrations completed successfully!")
        print(f"📊 Database is ready for AI Employee Lifecycle operations")
        
    except Exception as e:
        print(f"💥 Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        if 'conn' in locals():
            await conn.close()
            print("🔌 Database connection closed")


async def main():
    """Main function"""
    print("🚀 Forge1 Database Migration Runner\n")
    
    # Check environment variables
    required_vars = ['POSTGRES_HOST', 'POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("   Please set these in your .env file")
        return
    
    await run_migrations()


if __name__ == "__main__":
    asyncio.run(main())