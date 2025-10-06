#!/usr/bin/env python3
"""
Database Migration Script for Employee Lifecycle System

Handles database schema migrations, data migrations, and rollbacks
with proper validation and backup procedures.

Requirements: 5.3, 5.4, 8.5
"""

import os
import sys
import asyncio
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import asyncpg
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from forge1.core.database_config import DatabaseManager
from forge1.core.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


class MigrationManager:
    """Manages database migrations for the Employee Lifecycle System"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.migrations_dir = Path(__file__).parent.parent / "migrations"
        self.db_manager = None
    
    async def initialize(self):
        """Initialize database connection"""
        self.db_manager = DatabaseManager(self.database_url)
        await self.db_manager.initialize()
    
    async def close(self):
        """Close database connection"""
        if self.db_manager:
            await self.db_manager.close()
    
    def get_migration_files(self) -> List[Path]:
        """Get all migration files sorted by version"""
        migration_files = []
        
        for file_path in self.migrations_dir.glob("*.sql"):
            if file_path.name.startswith(("000", "001", "002", "003", "004", "005", "006", "007", "008", "009")):
                migration_files.append(file_path)
        
        return sorted(migration_files)
    
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of migration file"""
        with open(file_path, 'rb') as f:
            content = f.read()
            return hashlib.md5(content).hexdigest()
    
    async def ensure_migrations_table(self):
        """Ensure the schema_migrations table exists"""
        async with self.db_manager.get_connection() as conn:
            # Check if migrations table exists
            exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'schema_migrations'
                )
            """)
            
            if not exists:
                console.print("Creating schema_migrations table...", style="yellow")
                
                # Read and execute the migrations table creation script
                migrations_table_sql = self.migrations_dir / "schema_migrations.sql"
                if migrations_table_sql.exists():
                    with open(migrations_table_sql, 'r') as f:
                        await conn.execute(f.read())
                else:
                    # Fallback creation
                    await conn.execute("""
                        CREATE TABLE schema_migrations (
                            version VARCHAR(20) PRIMARY KEY,
                            applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            applied_by VARCHAR(100) DEFAULT current_user,
                            description TEXT,
                            checksum VARCHAR(64)
                        )
                    """)
    
    async def get_applied_migrations(self) -> Dict[str, Dict]:
        """Get list of applied migrations"""
        async with self.db_manager.get_connection() as conn:
            rows = await conn.fetch("""
                SELECT version, applied_at, applied_by, description, checksum
                FROM schema_migrations
                ORDER BY version
            """)
            
            return {
                row['version']: {
                    'applied_at': row['applied_at'],
                    'applied_by': row['applied_by'],
                    'description': row['description'],
                    'checksum': row['checksum']
                }
                for row in rows
            }
    
    async def get_pending_migrations(self) -> List[Tuple[str, Path]]:
        """Get list of pending migrations"""
        applied_migrations = await self.get_applied_migrations()
        migration_files = self.get_migration_files()
        
        pending = []
        for file_path in migration_files:
            version = self.extract_version(file_path)
            if version not in applied_migrations:
                pending.append((version, file_path))
        
        return pending
    
    def extract_version(self, file_path: Path) -> str:
        """Extract version number from migration filename"""
        filename = file_path.stem
        if '_' in filename:
            return filename.split('_')[0]
        return filename
    
    def extract_description(self, file_path: Path) -> str:
        """Extract description from migration filename"""
        filename = file_path.stem
        if '_' in filename:
            parts = filename.split('_', 1)
            if len(parts) > 1:
                return parts[1].replace('_', ' ').title()
        return "Migration"
    
    async def validate_migration(self, file_path: Path) -> bool:
        """Validate migration file syntax"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Basic SQL syntax validation
            if not content.strip():
                console.print(f"❌ Migration {file_path.name} is empty", style="red")
                return False
            
            # Check for dangerous operations in production
            if settings.ENVIRONMENT == "production":
                dangerous_keywords = ['DROP TABLE', 'DROP DATABASE', 'TRUNCATE', 'DELETE FROM']
                content_upper = content.upper()
                
                for keyword in dangerous_keywords:
                    if keyword in content_upper:
                        console.print(f"⚠️  Migration {file_path.name} contains dangerous operation: {keyword}", style="yellow")
                        if not click.confirm("Continue with this migration?"):
                            return False
            
            return True
            
        except Exception as e:
            console.print(f"❌ Error validating migration {file_path.name}: {e}", style="red")
            return False
    
    async def backup_database(self) -> Optional[str]:
        """Create database backup before migration"""
        if settings.ENVIRONMENT == "production":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"backup_before_migration_{timestamp}.sql"
            
            console.print(f"Creating database backup: {backup_file}", style="yellow")
            
            # Use pg_dump to create backup
            import subprocess
            
            try:
                # Parse database URL
                from urllib.parse import urlparse
                parsed = urlparse(self.database_url)
                
                cmd = [
                    "pg_dump",
                    "-h", parsed.hostname,
                    "-p", str(parsed.port or 5432),
                    "-U", parsed.username,
                    "-d", parsed.path.lstrip('/'),
                    "--format=custom",
                    "--compress=9",
                    "--file", backup_file
                ]
                
                env = os.environ.copy()
                if parsed.password:
                    env["PGPASSWORD"] = parsed.password
                
                result = subprocess.run(cmd, env=env, capture_output=True, text=True)
                
                if result.returncode == 0:
                    console.print(f"✅ Backup created successfully: {backup_file}", style="green")
                    return backup_file
                else:
                    console.print(f"❌ Backup failed: {result.stderr}", style="red")
                    return None
                    
            except Exception as e:
                console.print(f"❌ Backup error: {e}", style="red")
                return None
        
        return None
    
    async def apply_migration(self, version: str, file_path: Path) -> bool:
        """Apply a single migration"""
        try:
            console.print(f"Applying migration {version}: {file_path.name}", style="blue")
            
            # Validate migration
            if not await self.validate_migration(file_path):
                return False
            
            # Read migration content
            with open(file_path, 'r') as f:
                migration_sql = f.read()
            
            # Calculate checksum
            checksum = self.calculate_checksum(file_path)
            description = self.extract_description(file_path)
            
            # Apply migration in transaction
            async with self.db_manager.get_connection() as conn:
                async with conn.transaction():
                    # Execute migration SQL
                    await conn.execute(migration_sql)
                    
                    # Record migration as applied
                    await conn.execute("""
                        INSERT INTO schema_migrations (version, description, checksum)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (version) DO UPDATE SET
                            applied_at = NOW(),
                            applied_by = current_user,
                            description = $2,
                            checksum = $3
                    """, version, description, checksum)
            
            console.print(f"✅ Migration {version} applied successfully", style="green")
            return True
            
        except Exception as e:
            console.print(f"❌ Error applying migration {version}: {e}", style="red")
            logger.exception(f"Migration {version} failed")
            return False
    
    async def rollback_migration(self, version: str) -> bool:
        """Rollback a migration (if rollback script exists)"""
        rollback_file = self.migrations_dir / f"{version}_rollback.sql"
        
        if not rollback_file.exists():
            console.print(f"❌ No rollback script found for migration {version}", style="red")
            return False
        
        try:
            console.print(f"Rolling back migration {version}", style="yellow")
            
            # Read rollback content
            with open(rollback_file, 'r') as f:
                rollback_sql = f.read()
            
            # Apply rollback in transaction
            async with self.db_manager.get_connection() as conn:
                async with conn.transaction():
                    # Execute rollback SQL
                    await conn.execute(rollback_sql)
                    
                    # Remove migration record
                    await conn.execute("""
                        DELETE FROM schema_migrations WHERE version = $1
                    """, version)
            
            console.print(f"✅ Migration {version} rolled back successfully", style="green")
            return True
            
        except Exception as e:
            console.print(f"❌ Error rolling back migration {version}: {e}", style="red")
            logger.exception(f"Rollback {version} failed")
            return False
    
    async def migrate_up(self, target_version: Optional[str] = None) -> bool:
        """Apply pending migrations up to target version"""
        await self.ensure_migrations_table()
        
        pending_migrations = await self.get_pending_migrations()
        
        if not pending_migrations:
            console.print("✅ No pending migrations", style="green")
            return True
        
        # Filter by target version if specified
        if target_version:
            pending_migrations = [
                (version, path) for version, path in pending_migrations
                if version <= target_version
            ]
        
        console.print(f"Found {len(pending_migrations)} pending migrations", style="blue")
        
        # Create backup in production
        backup_file = await self.backup_database()
        
        # Apply migrations
        success_count = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Applying migrations...", total=len(pending_migrations))
            
            for version, file_path in pending_migrations:
                progress.update(task, description=f"Applying {version}...")
                
                if await self.apply_migration(version, file_path):
                    success_count += 1
                else:
                    console.print(f"❌ Migration {version} failed, stopping", style="red")
                    break
                
                progress.advance(task)
        
        if success_count == len(pending_migrations):
            console.print(f"✅ All {success_count} migrations applied successfully", style="green")
            return True
        else:
            console.print(f"⚠️  Applied {success_count}/{len(pending_migrations)} migrations", style="yellow")
            if backup_file:
                console.print(f"Database backup available: {backup_file}", style="blue")
            return False
    
    async def migrate_down(self, target_version: str) -> bool:
        """Rollback migrations down to target version"""
        applied_migrations = await self.get_applied_migrations()
        
        # Find migrations to rollback (in reverse order)
        to_rollback = [
            version for version in sorted(applied_migrations.keys(), reverse=True)
            if version > target_version
        ]
        
        if not to_rollback:
            console.print(f"✅ Already at or below version {target_version}", style="green")
            return True
        
        console.print(f"Rolling back {len(to_rollback)} migrations", style="yellow")
        
        # Create backup in production
        backup_file = await self.backup_database()
        
        # Rollback migrations
        success_count = 0
        for version in to_rollback:
            if await self.rollback_migration(version):
                success_count += 1
            else:
                console.print(f"❌ Rollback {version} failed, stopping", style="red")
                break
        
        if success_count == len(to_rollback):
            console.print(f"✅ All {success_count} rollbacks completed successfully", style="green")
            return True
        else:
            console.print(f"⚠️  Completed {success_count}/{len(to_rollback)} rollbacks", style="yellow")
            if backup_file:
                console.print(f"Database backup available: {backup_file}", style="blue")
            return False
    
    async def show_status(self):
        """Show migration status"""
        await self.ensure_migrations_table()
        
        applied_migrations = await self.get_applied_migrations()
        pending_migrations = await self.get_pending_migrations()
        
        # Create status table
        table = Table(title="Migration Status")
        table.add_column("Version", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Applied At", style="blue")
        table.add_column("Description", style="white")
        
        # Add applied migrations
        for version, info in applied_migrations.items():
            table.add_row(
                version,
                "✅ Applied",
                info['applied_at'].strftime("%Y-%m-%d %H:%M:%S") if info['applied_at'] else "Unknown",
                info['description'] or "No description"
            )
        
        # Add pending migrations
        for version, file_path in pending_migrations:
            description = self.extract_description(file_path)
            table.add_row(
                version,
                "⏳ Pending",
                "-",
                description
            )
        
        console.print(table)
        
        # Summary
        console.print(f"\nSummary:", style="bold")
        console.print(f"Applied migrations: {len(applied_migrations)}", style="green")
        console.print(f"Pending migrations: {len(pending_migrations)}", style="yellow")
    
    async def verify_integrity(self) -> bool:
        """Verify database integrity after migrations"""
        try:
            async with self.db_manager.get_connection() as conn:
                # Check table existence
                tables = await conn.fetch("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """)
                
                expected_tables = {
                    'clients', 'employees', 'interactions', 'memories',
                    'knowledge_sources', 'analytics_events', 'performance_metrics',
                    'audit_logs', 'schema_migrations'
                }
                
                actual_tables = {row['table_name'] for row in tables}
                missing_tables = expected_tables - actual_tables
                
                if missing_tables:
                    console.print(f"❌ Missing tables: {missing_tables}", style="red")
                    return False
                
                # Check indexes
                indexes = await conn.fetch("""
                    SELECT indexname 
                    FROM pg_indexes 
                    WHERE schemaname = 'public'
                    AND indexname LIKE 'idx_%'
                """)
                
                if len(indexes) < 10:  # Should have many indexes
                    console.print(f"⚠️  Only {len(indexes)} indexes found, expected more", style="yellow")
                
                # Check RLS policies
                policies = await conn.fetch("""
                    SELECT tablename, policyname
                    FROM pg_policies
                    WHERE schemaname = 'public'
                """)
                
                if len(policies) < 8:  # Should have tenant isolation policies
                    console.print(f"⚠️  Only {len(policies)} RLS policies found", style="yellow")
                
                console.print("✅ Database integrity check passed", style="green")
                return True
                
        except Exception as e:
            console.print(f"❌ Integrity check failed: {e}", style="red")
            return False


# CLI Commands
@click.group()
@click.option('--database-url', default=None, help='Database URL (defaults to settings)')
@click.pass_context
def cli(ctx, database_url):
    """Database migration tool for Employee Lifecycle System"""
    ctx.ensure_object(dict)
    ctx.obj['database_url'] = database_url or settings.DATABASE_URL


@cli.command()
@click.option('--target', help='Target migration version')
@click.pass_context
def up(ctx, target):
    """Apply pending migrations"""
    async def run():
        manager = MigrationManager(ctx.obj['database_url'])
        try:
            await manager.initialize()
            success = await manager.migrate_up(target)
            sys.exit(0 if success else 1)
        finally:
            await manager.close()
    
    asyncio.run(run())


@cli.command()
@click.argument('target_version')
@click.pass_context
def down(ctx, target_version):
    """Rollback migrations to target version"""
    async def run():
        manager = MigrationManager(ctx.obj['database_url'])
        try:
            await manager.initialize()
            success = await manager.migrate_down(target_version)
            sys.exit(0 if success else 1)
        finally:
            await manager.close()
    
    asyncio.run(run())


@cli.command()
@click.pass_context
def status(ctx):
    """Show migration status"""
    async def run():
        manager = MigrationManager(ctx.obj['database_url'])
        try:
            await manager.initialize()
            await manager.show_status()
        finally:
            await manager.close()
    
    asyncio.run(run())


@cli.command()
@click.pass_context
def verify(ctx):
    """Verify database integrity"""
    async def run():
        manager = MigrationManager(ctx.obj['database_url'])
        try:
            await manager.initialize()
            success = await manager.verify_integrity()
            sys.exit(0 if success else 1)
        finally:
            await manager.close()
    
    asyncio.run(run())


@cli.command()
@click.pass_context
def reset(ctx):
    """Reset database (WARNING: Destructive operation)"""
    if not click.confirm("This will destroy all data. Are you sure?"):
        return
    
    if settings.ENVIRONMENT == "production":
        console.print("❌ Reset not allowed in production", style="red")
        sys.exit(1)
    
    async def run():
        manager = MigrationManager(ctx.obj['database_url'])
        try:
            await manager.initialize()
            
            # Drop all tables
            async with manager.db_manager.get_connection() as conn:
                await conn.execute("""
                    DROP SCHEMA public CASCADE;
                    CREATE SCHEMA public;
                    GRANT ALL ON SCHEMA public TO postgres;
                    GRANT ALL ON SCHEMA public TO public;
                """)
            
            console.print("✅ Database reset completed", style="green")
            
            # Apply all migrations
            await manager.migrate_up()
            
        finally:
            await manager.close()
    
    asyncio.run(run())


if __name__ == "__main__":
    cli()