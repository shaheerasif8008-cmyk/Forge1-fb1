'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { 
  BarChart3, 
  TrendingUp, 
  TrendingDown,
  Users, 
  Bot, 
  Zap, 
  DollarSign,
  Clock,
  Target,
  AlertTriangle,
  CheckCircle,
  Activity,
  Eye,
  Settings,
  Download,
  RefreshCw,
  Calendar,
  Filter
} from 'lucide-react'
import { useAuth } from '@/hooks/use-auth'
import { PerformanceMetricsWidget } from '@/components/dashboard/performance-metrics-widget'
import { AIEmployeeStatusGrid } from '@/components/dashboard/ai-employee-status-grid'
import { RealTimeActivityFeed } from '@/components/dashboard/real-time-activity-feed'
import { PerformanceTrendsChart } from '@/components/dashboard/performance-trends-chart'
import { ROICalculatorWidget } from '@/components/dashboard/roi-calculator-widget'
import { ComplianceMonitorWidget } from '@/components/dashboard/compliance-monitor-widget'
import { PredictiveAnalyticsWidget } from '@/components/dashboard/predictive-analytics-widget'

const timeRanges = [
  { value: '1h', label: 'Last Hour' },
  { value: '24h', label: 'Last 24 Hours' },
  { value: '7d', label: 'Last 7 Days' },
  { value: '30d', label: 'Last 30 Days' },
  { value: '90d', label: 'Last 90 Days' }
]

const dashboardMetrics = {
  overview: {
    totalEmployees: 47,
    activeEmployees: 43,
    totalTasks: 15420,
    completedTasks: 14891,
    averageAccuracy: 97.8,
    averageSpeed: 12.4, // x faster than human
    totalCostSavings: 2847392,
    uptime: 99.94
  },
  performance: {
    accuracy: { current: 97.8, trend: 2.3, target: 95 },
    speed: { current: 12.4, trend: 8.7, target: 5 },
    availability: { current: 99.94, trend: 0.12, target: 99.9 },
    customerSatisfaction: { current: 94.2, trend: 5.1, target: 90 },
    errorRate: { current: 2.2, trend: -15.3, target: 5 },
    throughput: { current: 1847, trend: 23.4, target: 1000 }
  },
  alerts: [
    {
      id: 1,
      type: 'warning',
      title: 'High CPU Usage',
      description: 'Customer Service Agent #3 showing elevated CPU usage',
      timestamp: '2 minutes ago',
      severity: 'medium'
    },
    {
      id: 2,
      type: 'success',
      title: 'Performance Target Exceeded',
      description: 'Data Analyst team exceeded accuracy target by 12%',
      timestamp: '15 minutes ago',
      severity: 'low'
    },
    {
      id: 3,
      type: 'info',
      title: 'Scheduled Maintenance',
      description: 'System maintenance scheduled for tonight at 2 AM UTC',
      timestamp: '1 hour ago',
      severity: 'low'
    }
  ]
}

export default function DashboardPage() {
  const { user } = useAuth()
  const [timeRange, setTimeRange] = useState('24h')
  const [refreshing, setRefreshing] = useState(false)
  const [lastUpdated, setLastUpdated] = useState(new Date())

  const handleRefresh = async () => {
    setRefreshing(true)
    // Simulate data refresh
    await new Promise(resolve => setTimeout(resolve, 1000))
    setLastUpdated(new Date())
    setRefreshing(false)
  }

  const getMetricTrendIcon = (trend: number) => {
    if (trend > 0) return <TrendingUp className="h-4 w-4 text-green-500" />
    if (trend < 0) return <TrendingDown className="h-4 w-4 text-red-500" />
    return <Activity className="h-4 w-4 text-muted-foreground" />
  }

  const getMetricTrendColor = (trend: number) => {
    if (trend > 0) return 'text-green-600'
    if (trend < 0) return 'text-red-600'
    return 'text-muted-foreground'
  }

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount)
  }

  const formatNumber = (num: number, decimals: number = 1) => {
    return new Intl.NumberFormat('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    }).format(num)
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container-forge flex h-16 items-center justify-between">
          <div className="flex items-center space-x-4">
            <BarChart3 className="h-8 w-8 text-forge-500" />
            <div>
              <h1 className="text-xl font-bold">Performance Dashboard</h1>
              <p className="text-sm text-muted-foreground">
                Welcome back, {user?.name}
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-sm text-muted-foreground">
              <Clock className="h-4 w-4" />
              <span>Last updated: {lastUpdated.toLocaleTimeString()}</span>
            </div>
            
            <Select value={timeRange} onValueChange={setTimeRange}>
              <SelectTrigger className="w-40">
                <Calendar className="h-4 w-4 mr-2" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {timeRanges.map((range) => (
                  <SelectItem key={range.value} value={range.value}>
                    {range.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              disabled={refreshing}
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
            
            <Button variant="outline" size="sm">
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container-forge py-6 space-y-6">
        {/* Key Metrics Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Active AI Employees</p>
                    <p className="text-2xl font-bold">
                      {dashboardMetrics.overview.activeEmployees}
                      <span className="text-sm font-normal text-muted-foreground">
                        /{dashboardMetrics.overview.totalEmployees}
                      </span>
                    </p>
                  </div>
                  <Bot className="h-8 w-8 text-forge-500" />
                </div>
                <div className="flex items-center mt-2">
                  <Badge variant="secondary" className="text-xs">
                    91% Active
                  </Badge>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Tasks Completed</p>
                    <p className="text-2xl font-bold">
                      {formatNumber(dashboardMetrics.overview.completedTasks, 0)}
                    </p>
                  </div>
                  <CheckCircle className="h-8 w-8 text-green-500" />
                </div>
                <div className="flex items-center mt-2">
                  <TrendingUp className="h-4 w-4 text-green-500 mr-1" />
                  <span className="text-sm text-green-600">+23% vs last period</span>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Average Performance</p>
                    <p className="text-2xl font-bold">
                      {formatNumber(dashboardMetrics.overview.averageSpeed)}x
                    </p>
                  </div>
                  <Zap className="h-8 w-8 text-yellow-500" />
                </div>
                <div className="flex items-center mt-2">
                  <Badge variant="default" className="text-xs bg-gradient-forge">
                    Superhuman
                  </Badge>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Cost Savings</p>
                    <p className="text-2xl font-bold">
                      {formatCurrency(dashboardMetrics.overview.totalCostSavings)}
                    </p>
                  </div>
                  <DollarSign className="h-8 w-8 text-green-500" />
                </div>
                <div className="flex items-center mt-2">
                  <TrendingUp className="h-4 w-4 text-green-500 mr-1" />
                  <span className="text-sm text-green-600">+47% ROI</span>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Performance Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {Object.entries(dashboardMetrics.performance).map(([key, metric], index) => (
            <motion.div
              key={key}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * (index + 5) }}
            >
              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-semibold capitalize">
                      {key.replace(/([A-Z])/g, ' $1').trim()}
                    </h3>
                    {getMetricTrendIcon(metric.trend)}
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex items-baseline space-x-2">
                      <span className="text-2xl font-bold">
                        {formatNumber(metric.current)}
                        {key === 'speed' ? 'x' : '%'}
                      </span>
                      <span className={`text-sm ${getMetricTrendColor(metric.trend)}`}>
                        {metric.trend > 0 ? '+' : ''}{formatNumber(metric.trend)}%
                      </span>
                    </div>
                    
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">
                        Target: {formatNumber(metric.target)}{key === 'speed' ? 'x' : '%'}
                      </span>
                      <Badge 
                        variant={metric.current >= metric.target ? "default" : "secondary"}
                        className="text-xs"
                      >
                        {metric.current >= metric.target ? 'On Target' : 'Below Target'}
                      </Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Dashboard Tabs */}
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="employees">AI Employees</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
            <TabsTrigger value="alerts">Alerts</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            <div className="grid lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <PerformanceTrendsChart timeRange={timeRange} />
              </div>
              <div>
                <RealTimeActivityFeed />
              </div>
            </div>
            
            <div className="grid md:grid-cols-2 gap-6">
              <ROICalculatorWidget />
              <ComplianceMonitorWidget />
            </div>
          </TabsContent>

          {/* AI Employees Tab */}
          <TabsContent value="employees" className="space-y-6">
            <AIEmployeeStatusGrid timeRange={timeRange} />
          </TabsContent>

          {/* Performance Tab */}
          <TabsContent value="performance" className="space-y-6">
            <PerformanceMetricsWidget timeRange={timeRange} />
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-6">
            <PredictiveAnalyticsWidget timeRange={timeRange} />
          </TabsContent>

          {/* Alerts Tab */}
          <TabsContent value="alerts" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <AlertTriangle className="h-5 w-5" />
                  <span>System Alerts</span>
                  <Badge variant="secondary">{dashboardMetrics.alerts.length}</Badge>
                </CardTitle>
                <CardDescription>
                  Monitor system health and performance alerts
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {dashboardMetrics.alerts.map((alert) => (
                    <div
                      key={alert.id}
                      className="flex items-start space-x-4 p-4 border rounded-lg"
                    >
                      <div className={`p-2 rounded-full ${
                        alert.type === 'warning' ? 'bg-yellow-100 text-yellow-600' :
                        alert.type === 'success' ? 'bg-green-100 text-green-600' :
                        'bg-blue-100 text-blue-600'
                      }`}>
                        {alert.type === 'warning' ? <AlertTriangle className="h-4 w-4" /> :
                         alert.type === 'success' ? <CheckCircle className="h-4 w-4" /> :
                         <Activity className="h-4 w-4" />}
                      </div>
                      
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <h4 className="font-medium">{alert.title}</h4>
                          <Badge 
                            variant={alert.severity === 'high' ? 'destructive' : 'secondary'}
                            className="text-xs"
                          >
                            {alert.severity}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mt-1">
                          {alert.description}
                        </p>
                        <p className="text-xs text-muted-foreground mt-2">
                          {alert.timestamp}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}