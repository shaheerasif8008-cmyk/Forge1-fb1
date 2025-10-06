#!/usr/bin/env python3
"""
Comprehensive test suite for the compliance implementation
Tests SOC2, GDPR, HIPAA, and unified compliance management
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the forge1 directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'forge1'))

from forge1.core.performance_monitor import MetricsCollector
from forge1.core.memory_manager import MemoryManager
from forge1.core.secret_manager import SecretManager
from forge1.compliance.soc2 import SOC2ComplianceManager, SOC2Principle, ComplianceStatus
from forge1.compliance.gdpr import GDPRComplianceManager, LegalBasis, DataCategory, ProcessingPurpose, DataSubjectRightType
from forge1.compliance.hipaa import HIPAAComplianceManager, PHICategory, IncidentSeverity, SafeguardType
from forge1.compliance.manager import UnifiedComplianceManager, ComplianceFramework, ComplianceRisk


class TestResults:
    """Track test results"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def assert_test(self, condition: bool, test_name: str, error_msg: str = ""):
        """Assert a test condition"""
        if condition:
            self.passed += 1
            print(f"✅ {test_name}")
        else:
            self.failed += 1
            error = f"❌ {test_name}: {error_msg}"
            self.errors.append(error)
            print(error)
    
    def print_summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {(self.passed/total*100):.1f}%" if total > 0 else "0%")
        
        if self.errors:
            print(f"\nFAILED TESTS:")
            for error in self.errors:
                print(f"  {error}")
        
        return self.failed == 0


async def test_soc2_compliance():
    """Test SOC2 compliance implementation"""
    print("\n" + "="*60)
    print("TESTING SOC2 COMPLIANCE")
    print("="*60)
    
    results = TestResults()
    
    try:
        # Initialize components
        memory_manager = MemoryManager()
        metrics_collector = MetricsCollector()
        secret_manager = SecretManager()
        
        # Initialize SOC2 manager
        soc2_manager = SOC2ComplianceManager(memory_manager, metrics_collector, secret_manager)
        
        # Test 1: Verify SOC2 controls initialization
        results.assert_test(
            len(soc2_manager.controls) > 0,
            "SOC2 controls initialization",
            f"Expected controls to be initialized, got {len(soc2_manager.controls)}"
        )
        
        # Test 2: Verify control categories
        security_controls = [c for c in soc2_manager.controls.values() if c.principle == SOC2Principle.SECURITY]
        results.assert_test(
            len(security_controls) > 0,
            "Security principle controls",
            "No security controls found"
        )
        
        # Test 3: Execute a control
        control_id = list(soc2_manager.controls.keys())[0]
        execution = await soc2_manager.execute_control(control_id, "test_user")
        results.assert_test(
            execution.control_id == control_id,
            "Control execution",
            f"Expected control_id {control_id}, got {execution.control_id}"
        )
        
        # Test 4: Log audit event
        audit_id = await soc2_manager.log_audit_event(
            user_id="test_user",
            event_type="data_access",
            resource_type="customer_data",
            resource_id="cust_123",
            action="read"
        )
        results.assert_test(
            audit_id.startswith("audit_"),
            "Audit event logging",
            f"Expected audit ID to start with 'audit_', got {audit_id}"
        )
        
        # Test 5: Run compliance assessment
        assessment = await soc2_manager.run_compliance_assessment()
        results.assert_test(
            "assessment_id" in assessment,
            "Compliance assessment",
            "Assessment missing required fields"
        )
        
        # Test 6: Get compliance dashboard
        dashboard = soc2_manager.get_compliance_dashboard()
        results.assert_test(
            "compliance_overview" in dashboard,
            "Compliance dashboard",
            "Dashboard missing compliance overview"
        )
        
        print(f"\nSOC2 Compliance Tests: {results.passed} passed, {results.failed} failed")
        
    except Exception as e:
        results.assert_test(False, "SOC2 compliance test suite", str(e))
    
    return results


async def test_gdpr_compliance():
    """Test GDPR compliance implementation"""
    print("\n" + "="*60)
    print("TESTING GDPR COMPLIANCE")
    print("="*60)
    
    results = TestResults()
    
    try:
        # Initialize components
        memory_manager = MemoryManager()
        metrics_collector = MetricsCollector()
        secret_manager = SecretManager()
        
        # Initialize GDPR manager
        gdpr_manager = GDPRComplianceManager(memory_manager, metrics_collector, secret_manager)
        
        # Test 1: Verify processing records initialization
        results.assert_test(
            len(gdpr_manager.processing_records) > 0,
            "GDPR processing records initialization",
            f"Expected processing records to be initialized, got {len(gdpr_manager.processing_records)}"
        )
        
        # Test 2: Record consent
        consent_id = await gdpr_manager.record_consent(
            data_subject_id="user_123",
            processing_purposes=[ProcessingPurpose.SERVICE_PROVISION],
            data_categories=[DataCategory.BASIC_IDENTITY, DataCategory.CONTACT_DETAILS],
            consent_method="web_form",
            consent_text="I consent to processing of my personal data"
        )
        results.assert_test(
            consent_id.startswith("consent_"),
            "Consent recording",
            f"Expected consent ID to start with 'consent_', got {consent_id}"
        )
        
        # Test 3: Submit data subject request
        request_id = await gdpr_manager.submit_data_subject_request(
            data_subject_id="user_123",
            request_type=DataSubjectRightType.ACCESS,
            description="Request for access to my personal data",
            contact_email="user@example.com"
        )
        results.assert_test(
            request_id.startswith("dsr_"),
            "Data subject request submission",
            f"Expected request ID to start with 'dsr_', got {request_id}"
        )
        
        # Test 4: Process data subject request
        processed = await gdpr_manager.process_data_subject_request(
            request_id=request_id,
            assigned_to="privacy_officer"
        )
        results.assert_test(
            processed,
            "Data subject request processing",
            "Failed to process data subject request"
        )
        
        # Test 5: Report data breach
        breach_id = await gdpr_manager.report_data_breach(
            description="Unauthorized access to customer database",
            breach_type="confidentiality",
            severity="high",
            discovered_at=datetime.utcnow(),
            reported_by="security_team",
            data_subjects_affected=100,
            data_categories_affected=[DataCategory.BASIC_IDENTITY, DataCategory.CONTACT_DETAILS]
        )
        results.assert_test(
            breach_id.startswith("breach_"),
            "Data breach reporting",
            f"Expected breach ID to start with 'breach_', got {breach_id}"
        )
        
        # Test 6: Get consent status
        consent_status = await gdpr_manager.get_consent_status("user_123")
        results.assert_test(
            consent_status["has_active_consent"],
            "Consent status retrieval",
            "Expected active consent for user_123"
        )
        
        # Test 7: Get GDPR dashboard
        dashboard = gdpr_manager.get_gdpr_dashboard()
        results.assert_test(
            "overview" in dashboard,
            "GDPR dashboard",
            "Dashboard missing overview section"
        )
        
        print(f"\nGDPR Compliance Tests: {results.passed} passed, {results.failed} failed")
        
    except Exception as e:
        results.assert_test(False, "GDPR compliance test suite", str(e))
    
    return results


async def test_hipaa_compliance():
    """Test HIPAA compliance implementation"""
    print("\n" + "="*60)
    print("TESTING HIPAA COMPLIANCE")
    print("="*60)
    
    results = TestResults()
    
    try:
        # Initialize components
        memory_manager = MemoryManager()
        metrics_collector = MetricsCollector()
        secret_manager = SecretManager()
        
        # Initialize HIPAA manager
        hipaa_manager = HIPAAComplianceManager(memory_manager, metrics_collector, secret_manager)
        
        # Test 1: Verify HIPAA safeguards initialization
        results.assert_test(
            len(hipaa_manager.safeguards) > 0,
            "HIPAA safeguards initialization",
            f"Expected safeguards to be initialized, got {len(hipaa_manager.safeguards)}"
        )
        
        # Test 2: Verify safeguard types
        admin_safeguards = [s for s in hipaa_manager.safeguards.values() if s.safeguard_type == SafeguardType.ADMINISTRATIVE]
        results.assert_test(
            len(admin_safeguards) > 0,
            "Administrative safeguards",
            "No administrative safeguards found"
        )
        
        # Test 3: Log PHI access
        access_log_id = await hipaa_manager.log_phi_access(
            user_id="doctor_123",
            patient_id="patient_456",
            action="view",
            phi_categories=[PHICategory.NAMES, PHICategory.MEDICAL_RECORD_NUMBERS],
            data_accessed="Patient medical record",
            purpose="Treatment"
        )
        results.assert_test(
            access_log_id.startswith("phi_access_"),
            "PHI access logging",
            f"Expected access log ID to start with 'phi_access_', got {access_log_id}"
        )
        
        # Test 4: Report security incident
        incident_id = await hipaa_manager.report_security_incident(
            title="Unauthorized PHI Access",
            description="Employee accessed PHI without authorization",
            severity=IncidentSeverity.HIGH,
            incident_type="privacy_violation",
            discovered_at=datetime.utcnow(),
            reported_by="security_officer",
            phi_involved=True,
            phi_categories=[PHICategory.NAMES, PHICategory.MEDICAL_RECORD_NUMBERS],
            patients_affected=1
        )
        results.assert_test(
            incident_id.startswith("incident_"),
            "Security incident reporting",
            f"Expected incident ID to start with 'incident_', got {incident_id}"
        )
        
        # Test 5: Assess safeguard compliance
        safeguard_id = list(hipaa_manager.safeguards.keys())[0]
        compliance_status = await hipaa_manager.assess_safeguard_compliance(
            safeguard_id=safeguard_id,
            assessed_by="compliance_officer"
        )
        results.assert_test(
            isinstance(compliance_status, type(hipaa_manager.safeguards[safeguard_id].status)),
            "Safeguard compliance assessment",
            f"Expected ComplianceStatus enum, got {type(compliance_status)}"
        )
        
        # Test 6: Create Business Associate Agreement
        baa_id = await hipaa_manager.create_business_associate_agreement(
            business_associate_name="Cloud Provider Inc.",
            contact_information="contact@cloudprovider.com",
            services_provided=["Data hosting", "Backup services"],
            phi_categories_accessed=[PHICategory.NAMES, PHICategory.MEDICAL_RECORD_NUMBERS],
            agreement_date=datetime.utcnow(),
            effective_date=datetime.utcnow(),
            created_by="legal_team"
        )
        results.assert_test(
            baa_id.startswith("baa_"),
            "Business Associate Agreement creation",
            f"Expected BAA ID to start with 'baa_', got {baa_id}"
        )
        
        # Test 7: Generate HIPAA compliance report
        compliance_report = await hipaa_manager.generate_hipaa_compliance_report()
        results.assert_test(
            "report_id" in compliance_report,
            "HIPAA compliance report generation",
            "Compliance report missing required fields"
        )
        
        # Test 8: Get HIPAA dashboard
        dashboard = hipaa_manager.get_hipaa_dashboard()
        results.assert_test(
            "overview" in dashboard,
            "HIPAA dashboard",
            "Dashboard missing overview section"
        )
        
        print(f"\nHIPAA Compliance Tests: {results.passed} passed, {results.failed} failed")
        
    except Exception as e:
        results.assert_test(False, "HIPAA compliance test suite", str(e))
    
    return results


async def test_unified_compliance():
    """Test unified compliance management"""
    print("\n" + "="*60)
    print("TESTING UNIFIED COMPLIANCE MANAGEMENT")
    print("="*60)
    
    results = TestResults()
    
    try:
        # Initialize components
        memory_manager = MemoryManager()
        metrics_collector = MetricsCollector()
        secret_manager = SecretManager()
        
        # Initialize unified compliance manager
        unified_manager = UnifiedComplianceManager(memory_manager, metrics_collector, secret_manager)
        
        # Test 1: Verify framework managers initialization
        results.assert_test(
            unified_manager.soc2_manager is not None,
            "SOC2 manager initialization",
            "SOC2 manager not initialized"
        )
        
        results.assert_test(
            unified_manager.gdpr_manager is not None,
            "GDPR manager initialization",
            "GDPR manager not initialized"
        )
        
        results.assert_test(
            unified_manager.hipaa_manager is not None,
            "HIPAA manager initialization",
            "HIPAA manager not initialized"
        )
        
        # Test 2: Create compliance alert
        alert_id = await unified_manager.create_compliance_alert(
            framework=ComplianceFramework.SOC2,
            severity=ComplianceRisk.HIGH,
            title="Test Compliance Alert",
            description="This is a test alert for compliance monitoring",
            alert_type="test_alert",
            source_component="test_suite"
        )
        results.assert_test(
            alert_id.startswith("alert_"),
            "Compliance alert creation",
            f"Expected alert ID to start with 'alert_', got {alert_id}"
        )
        
        # Test 3: Resolve compliance alert
        resolved = await unified_manager.resolve_compliance_alert(
            alert_id=alert_id,
            resolved_by="compliance_officer",
            resolution_notes="Test alert resolved successfully"
        )
        results.assert_test(
            resolved,
            "Compliance alert resolution",
            "Failed to resolve compliance alert"
        )
        
        # Test 4: Get compliance dashboard
        dashboard = await unified_manager.get_compliance_dashboard()
        results.assert_test(
            "overview" in dashboard,
            "Unified compliance dashboard",
            "Dashboard missing overview section"
        )
        
        results.assert_test(
            "framework_status" in dashboard,
            "Framework status in dashboard",
            "Dashboard missing framework status"
        )
        
        # Test 5: Run comprehensive assessment
        assessment = await unified_manager.run_comprehensive_assessment()
        results.assert_test(
            assessment.overall_compliance_score >= 0,
            "Comprehensive compliance assessment",
            f"Invalid compliance score: {assessment.overall_compliance_score}"
        )
        
        # Test 6: Generate executive report
        executive_report = await unified_manager.generate_executive_report()
        results.assert_test(
            "executive_summary" in executive_report,
            "Executive report generation",
            "Executive report missing summary"
        )
        
        results.assert_test(
            "key_metrics" in executive_report,
            "Executive report metrics",
            "Executive report missing key metrics"
        )
        
        print(f"\nUnified Compliance Tests: {results.passed} passed, {results.failed} failed")
        
    except Exception as e:
        results.assert_test(False, "Unified compliance test suite", str(e))
    
    return results


async def test_integration_scenarios():
    """Test integration scenarios across compliance frameworks"""
    print("\n" + "="*60)
    print("TESTING INTEGRATION SCENARIOS")
    print("="*60)
    
    results = TestResults()
    
    try:
        # Initialize components
        memory_manager = MemoryManager()
        metrics_collector = MetricsCollector()
        secret_manager = SecretManager()
        
        # Initialize unified manager
        unified_manager = UnifiedComplianceManager(memory_manager, metrics_collector, secret_manager)
        
        # Scenario 1: Healthcare data processing with multiple compliance requirements
        print("\nScenario 1: Healthcare Data Processing")
        
        # GDPR: Record consent for healthcare data processing
        consent_id = await unified_manager.gdpr_manager.record_consent(
            data_subject_id="patient_789",
            processing_purposes=[ProcessingPurpose.SERVICE_PROVISION],
            data_categories=[DataCategory.BASIC_IDENTITY, DataCategory.SPECIAL_CATEGORY],
            consent_method="web_form",
            consent_text="I consent to processing of my health data for treatment purposes"
        )
        
        # HIPAA: Log PHI access
        phi_access_id = await unified_manager.hipaa_manager.log_phi_access(
            user_id="doctor_456",
            patient_id="patient_789",
            action="create",
            phi_categories=[PHICategory.NAMES, PHICategory.MEDICAL_RECORD_NUMBERS],
            data_accessed="Patient treatment record",
            purpose="Treatment"
        )
        
        # SOC2: Log audit event for the same action
        audit_id = await unified_manager.soc2_manager.log_audit_event(
            user_id="doctor_456",
            event_type="phi_access",
            resource_type="patient_record",
            resource_id="patient_789",
            action="create"
        )
        
        results.assert_test(
            all([consent_id, phi_access_id, audit_id]),
            "Multi-framework healthcare data processing",
            "Failed to process healthcare data across all frameworks"
        )
        
        # Scenario 2: Security incident affecting multiple compliance areas
        print("\nScenario 2: Cross-Framework Security Incident")
        
        # Report incident in HIPAA
        hipaa_incident_id = await unified_manager.hipaa_manager.report_security_incident(
            title="Database Breach",
            description="Unauthorized access to patient database",
            severity=IncidentSeverity.CRITICAL,
            incident_type="breach",
            discovered_at=datetime.utcnow(),
            reported_by="security_team",
            phi_involved=True,
            patients_affected=500
        )
        
        # Report breach in GDPR
        gdpr_breach_id = await unified_manager.gdpr_manager.report_data_breach(
            description="Database breach affecting patient records",
            breach_type="confidentiality",
            severity="critical",
            discovered_at=datetime.utcnow(),
            reported_by="security_team",
            data_subjects_affected=500
        )
        
        # Create unified compliance alert
        alert_id = await unified_manager.create_compliance_alert(
            framework=ComplianceFramework.SOC2,
            severity=ComplianceRisk.CRITICAL,
            title="Multi-Framework Security Breach",
            description="Security breach affecting HIPAA and GDPR compliance",
            alert_type="security_breach",
            source_component="security_monitoring"
        )
        
        results.assert_test(
            all([hipaa_incident_id, gdpr_breach_id, alert_id]),
            "Cross-framework security incident handling",
            "Failed to handle security incident across frameworks"
        )
        
        # Scenario 3: Comprehensive compliance assessment
        print("\nScenario 3: Comprehensive Assessment")
        
        assessment = await unified_manager.run_comprehensive_assessment()
        
        results.assert_test(
            assessment.overall_compliance_score > 0,
            "Comprehensive assessment execution",
            f"Invalid overall compliance score: {assessment.overall_compliance_score}"
        )
        
        results.assert_test(
            len(assessment.soc2_compliance) > 0,
            "SOC2 assessment data",
            "Missing SOC2 assessment data"
        )
        
        results.assert_test(
            len(assessment.gdpr_compliance) > 0,
            "GDPR assessment data",
            "Missing GDPR assessment data"
        )
        
        results.assert_test(
            len(assessment.hipaa_compliance) > 0,
            "HIPAA assessment data",
            "Missing HIPAA assessment data"
        )
        
        print(f"\nIntegration Scenario Tests: {results.passed} passed, {results.failed} failed")
        
    except Exception as e:
        results.assert_test(False, "Integration scenarios test suite", str(e))
    
    return results


async def main():
    """Run all compliance tests"""
    print("FORGE 1 COMPLIANCE IMPLEMENTATION TEST SUITE")
    print("=" * 80)
    
    all_results = []
    
    # Run individual framework tests
    all_results.append(await test_soc2_compliance())
    all_results.append(await test_gdpr_compliance())
    all_results.append(await test_hipaa_compliance())
    all_results.append(await test_unified_compliance())
    all_results.append(await test_integration_scenarios())
    
    # Calculate overall results
    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    total_tests = total_passed + total_failed
    
    print("\n" + "="*80)
    print("OVERALL TEST RESULTS")
    print("="*80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Success Rate: {(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "0%")
    
    if total_failed > 0:
        print(f"\n❌ {total_failed} tests failed. See details above.")
        return False
    else:
        print(f"\n✅ All {total_passed} tests passed!")
        return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)