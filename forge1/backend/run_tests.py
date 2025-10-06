#!/usr/bin/env python3
"""
Test runner script for Employee Lifecycle System

Provides convenient commands for running different types of tests
with appropriate configurations and reporting.

Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional


class TestRunner:
    """Test runner for Employee Lifecycle System"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / "tests"
    
    def run_unit_tests(self, verbose: bool = False, coverage: bool = True) -> int:
        """Run unit tests"""
        print("🧪 Running Unit Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir / "unit"),
            "-m", "unit",
            "--tb=short"
        ]
        
        if verbose:
            cmd.extend(["-v", "-s"])
        
        if coverage:
            cmd.extend([
                "--cov=forge1",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/unit"
            ])
        
        return subprocess.run(cmd, cwd=self.project_root).returncode
    
    def run_integration_tests(self, verbose: bool = False, coverage: bool = True) -> int:
        """Run integration tests"""
        print("🔗 Running Integration Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir / "integration"),
            "-m", "integration",
            "--tb=short"
        ]
        
        if verbose:
            cmd.extend(["-v", "-s"])
        
        if coverage:
            cmd.extend([
                "--cov=forge1",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/integration"
            ])
        
        return subprocess.run(cmd, cwd=self.project_root).returncode
    
    def run_e2e_tests(self, verbose: bool = False, coverage: bool = False) -> int:
        """Run end-to-end tests"""
        print("🌐 Running End-to-End Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir / "e2e"),
            "-m", "e2e",
            "--tb=short",
            "--timeout=60"
        ]
        
        if verbose:
            cmd.extend(["-v", "-s"])
        
        if coverage:
            cmd.extend([
                "--cov=forge1",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/e2e"
            ])
        
        return subprocess.run(cmd, cwd=self.project_root).returncode
    
    def run_performance_tests(self, verbose: bool = False) -> int:
        """Run performance tests"""
        print("⚡ Running Performance Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "-m", "performance",
            "--tb=short",
            "--benchmark-only",
            "--benchmark-sort=mean"
        ]
        
        if verbose:
            cmd.extend(["-v", "-s"])
        
        return subprocess.run(cmd, cwd=self.project_root).returncode
    
    def run_security_tests(self, verbose: bool = False) -> int:
        """Run security tests"""
        print("🔒 Running Security Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "-m", "security",
            "--tb=short"
        ]
        
        if verbose:
            cmd.extend(["-v", "-s"])
        
        return subprocess.run(cmd, cwd=self.project_root).returncode
    
    def run_all_tests(self, verbose: bool = False, coverage: bool = True, skip_e2e: bool = False) -> int:
        """Run all tests"""
        print("🚀 Running All Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "--tb=short"
        ]
        
        if skip_e2e:
            cmd.extend(["-m", "not e2e"])
        
        if verbose:
            cmd.extend(["-v", "-s"])
        
        if coverage:
            cmd.extend([
                "--cov=forge1",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/all",
                "--cov-report=xml",
                "--cov-fail-under=80"
            ])
        
        return subprocess.run(cmd, cwd=self.project_root).returncode
    
    def run_specific_test(self, test_path: str, verbose: bool = False) -> int:
        """Run a specific test file or test function"""
        print(f"🎯 Running Specific Test: {test_path}")
        
        cmd = [
            "python", "-m", "pytest",
            test_path,
            "--tb=short"
        ]
        
        if verbose:
            cmd.extend(["-v", "-s"])
        
        return subprocess.run(cmd, cwd=self.project_root).returncode
    
    def generate_coverage_report(self) -> int:
        """Generate comprehensive coverage report"""
        print("📊 Generating Coverage Report...")
        
        # Run tests with coverage
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "-m", "not e2e",  # Skip E2E for coverage
            "--cov=forge1",
            "--cov-report=html:htmlcov",
            "--cov-report=xml",
            "--cov-report=term-missing",
            "--cov-fail-under=80"
        ]
        
        result = subprocess.run(cmd, cwd=self.project_root)
        
        if result.returncode == 0:
            print("✅ Coverage report generated successfully!")
            print(f"📁 HTML report: {self.project_root}/htmlcov/index.html")
            print(f"📄 XML report: {self.project_root}/coverage.xml")
        
        return result.returncode
    
    def lint_and_format(self) -> int:
        """Run linting and formatting checks"""
        print("🧹 Running Linting and Formatting...")
        
        # Run black for formatting
        black_cmd = ["python", "-m", "black", "--check", "forge1", "tests"]
        black_result = subprocess.run(black_cmd, cwd=self.project_root)
        
        # Run flake8 for linting
        flake8_cmd = ["python", "-m", "flake8", "forge1", "tests"]
        flake8_result = subprocess.run(flake8_cmd, cwd=self.project_root)
        
        # Run mypy for type checking
        mypy_cmd = ["python", "-m", "mypy", "forge1"]
        mypy_result = subprocess.run(mypy_cmd, cwd=self.project_root)
        
        return max(black_result.returncode, flake8_result.returncode, mypy_result.returncode)
    
    def setup_test_environment(self) -> int:
        """Set up test environment"""
        print("🔧 Setting up Test Environment...")
        
        # Install test dependencies
        install_cmd = [
            "pip", "install", "-e", ".",
            "pytest", "pytest-asyncio", "pytest-cov", "pytest-mock",
            "pytest-benchmark", "pytest-timeout", "httpx", "black", "flake8", "mypy"
        ]
        
        result = subprocess.run(install_cmd, cwd=self.project_root)
        
        if result.returncode == 0:
            print("✅ Test environment set up successfully!")
        
        return result.returncode
    
    def clean_test_artifacts(self) -> int:
        """Clean test artifacts and cache"""
        print("🧽 Cleaning Test Artifacts...")
        
        import shutil
        
        artifacts_to_clean = [
            self.project_root / "htmlcov",
            self.project_root / "coverage.xml",
            self.project_root / ".coverage",
            self.project_root / ".pytest_cache",
            self.project_root / "__pycache__"
        ]
        
        for artifact in artifacts_to_clean:
            if artifact.exists():
                if artifact.is_dir():
                    shutil.rmtree(artifact)
                else:
                    artifact.unlink()
                print(f"🗑️  Removed: {artifact}")
        
        print("✅ Test artifacts cleaned!")
        return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test runner for Employee Lifecycle System")
    parser.add_argument("command", choices=[
        "unit", "integration", "e2e", "performance", "security", "all",
        "coverage", "lint", "setup", "clean", "run"
    ], help="Test command to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage reporting")
    parser.add_argument("--skip-e2e", action="store_true", help="Skip end-to-end tests")
    parser.add_argument("--test-path", help="Specific test path to run")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    # Set environment variables for testing
    os.environ["TESTING"] = "1"
    os.environ["LOG_LEVEL"] = "INFO"
    
    try:
        if args.command == "unit":
            return runner.run_unit_tests(args.verbose, not args.no_coverage)
        elif args.command == "integration":
            return runner.run_integration_tests(args.verbose, not args.no_coverage)
        elif args.command == "e2e":
            return runner.run_e2e_tests(args.verbose, not args.no_coverage)
        elif args.command == "performance":
            return runner.run_performance_tests(args.verbose)
        elif args.command == "security":
            return runner.run_security_tests(args.verbose)
        elif args.command == "all":
            return runner.run_all_tests(args.verbose, not args.no_coverage, args.skip_e2e)
        elif args.command == "coverage":
            return runner.generate_coverage_report()
        elif args.command == "lint":
            return runner.lint_and_format()
        elif args.command == "setup":
            return runner.setup_test_environment()
        elif args.command == "clean":
            return runner.clean_test_artifacts()
        elif args.command == "run":
            if not args.test_path:
                print("❌ --test-path is required for 'run' command")
                return 1
            return runner.run_specific_test(args.test_path, args.verbose)
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
    
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())