// forge1/frontend/src/pages/DashboardPage.tsx
/**
 * Premium Client Dashboard
 * 
 * Comprehensive monitoring and analytics interface with compliance monitoring.
 */

import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ComplianceDashboard } from '@/components/dashboard/compliance-dashboard';
import { ComplianceMonitorWidget } from '@/components/dashboard/compliance-monitor-widget';
import { PerformanceMetricsWidget } from '@/components/dashboard/performance-metrics-widget';
import { ROICalculatorWidget } from '@/components/dashboard/roi-calculator-widget';
import { RealTimeMonitor } from '@/components/dashboard/real-time-monitor';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export const DashboardPage: React.FC = () => {
  return (
    <div className="dashboard-page p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Premium Client Dashboard</h1>
          <p className="text-muted-foreground">
            Monitor AI employee performance, ROI metrics, and compliance status
          </p>
        </div>
      </div>
      
      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="compliance">Compliance</TabsTrigger>
          <TabsTrigger value="roi">ROI & Analytics</TabsTrigger>
          <TabsTrigger value="monitoring">Real-time</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <PerformanceMetricsWidget />
            <ComplianceMonitorWidget />
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ROICalculatorWidget />
            <RealTimeMonitor />
          </div>
        </TabsContent>

        {/* Performance Tab */}
        <TabsContent value="performance" className="space-y-6">
          <div className="grid grid-cols-1 gap-6">
            <PerformanceMetricsWidget />
            <Card>
              <CardHeader>
                <CardTitle>Performance Analytics</CardTitle>
                <CardDescription>
                  Detailed performance metrics and trends for AI employees
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center p-8 text-muted-foreground">
                  Advanced performance analytics will be implemented in future tasks
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Compliance Tab */}
        <TabsContent value="compliance" className="space-y-6">
          <ComplianceDashboard />
        </TabsContent>

        {/* ROI & Analytics Tab */}
        <TabsContent value="roi" className="space-y-6">
          <div className="grid grid-cols-1 gap-6">
            <ROICalculatorWidget />
            <Card>
              <CardHeader>
                <CardTitle>Advanced Analytics</CardTitle>
                <CardDescription>
                  Comprehensive ROI analysis and business intelligence
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center p-8 text-muted-foreground">
                  Advanced analytics dashboard will be implemented in future tasks
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Real-time Monitoring Tab */}
        <TabsContent value="monitoring" className="space-y-6">
          <div className="grid grid-cols-1 gap-6">
            <RealTimeMonitor />
            <Card>
              <CardHeader>
                <CardTitle>System Health</CardTitle>
                <CardDescription>
                  Real-time system health and performance monitoring
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center p-8 text-muted-foreground">
                  Advanced system monitoring will be implemented in future tasks
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};