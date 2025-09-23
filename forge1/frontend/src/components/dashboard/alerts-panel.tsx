'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { 
  AlertTriangle, 
  CheckCircle, 
  Info, 
  XCircle,
  Clock,
  TrendingDown,
  TrendingUp,
  Zap,
  Bot,
  Shield,
  Activity,
  Bell,
  X,
  Eye,
  Archive,
  Filter
} from 'lucide-react'

interface DashboardMetrics {
  totalEmployees: number
  activeEmployees: number
  totalTasks: number
  completedTasks: number
  averageAccuracy: number
  averageSpeed: number
  totalCostSavings: number
  roiPercentage: number
  uptime: number
  errorRate: number
  customerSatisfaction: number
  performanceImprovement: number
}

interface AlertsPanelProps {
  metrics: DashboardMetrics
}

interface Alert {
  id: string
  type: 'error' | 'warning' | 'info' | 'success'
  title: string
  message: string
  timestamp: Date
  source: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  status: 'active' | 'acknowledged' | 'resolved'
  actionRequired?: boolean
  relatedEmployee?: string
  category: 'performance' | 'system' | 'security' | 'maintenance' | 'compliance'
}

const generateMockAlerts = (): Alert[] => [
  {
    id: 'alert-1',
    type: 'warning',
    title: 'Performance Degradation Detected',
    message: 'Sarah (Customer Service) showing 15% decrease in response time over the last hour',
    timestamp: new Date(Date.now() - 5 * 60 * 1000),
    source: 'Performance Monitor',
    severity: 'medium',
    status: 'active',
    actionRequired: true,
    relatedEmployee: 'Sarah',
    category: 'performance'
  },
  {
    id: 'alert-2',
    type: 'success',
    title: 'Accuracy Milestone Achieved',
    message: 'Alex (Data Analyst) has maintained 99%+ accuracy for 7 consecutive days',
    timestamp: new Date(Date.now() - 15 * 60 * 1000),
    source: 'Quality Assurance',
    severity: 'low',
    status: 'active',
    relatedEmployee: 'Alex',
    category: 'performance'
  },
  {
    id: 'alert-3',
    type: 'error',
    title: 'Task Queue Overflow',
    message: 'Customer service queue has exceeded capacity. 25 tasks pending assignment.',
    timestamp: new Date(Date.now() - 30 * 60 * 1000),
    source: 'Task Manager',
    severity: 'high',
    status: 'acknowledged',
    actionRequired: true,
    category: 'system'
  },
  {
    id: 'alert-4',
    type: 'info',
    title: 'Scheduled Maintenance',
    message: 'Riley (HR Assistant) will undergo routine maintenance in 2 hours',
    timestamp: new Date(Date.now() - 45 * 60 * 1000),
    source: 'Maintenance Scheduler',
    severity: 'low',
    status: 'active',
    relatedEmployee: 'Riley',
    category: 'maintenance'
  },
  {
    id: 'alert-5',
    type: 'warning',
    title: 'Unusual Activity Pattern',
    message: 'Jordan (Sales Assistant) has been idle for 45 minutes during peak hours',
    timestamp: new Date(Date.now() - 60 * 60 * 1000),
    source: 'Activity Monitor',
    severity: 'medium',
    status: 'active',
    actionRequired: true,
    relatedEmployee: 'Jordan',
    category: 'performance'
  },
  {
    id: 'alert-6',
    type: 'success',
    title: 'Cost Savings Target Exceeded',
    message: 'Monthly cost savings target of $250K exceeded by 14%',
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
    source: 'Financial Analytics',
    severity: 'low',
    status: 'resolved',
    category: 'performance'
  },
  {
    id: 'alert-7',
    type: 'warning',
    title: 'Security Scan Required',
    message: 'Quarterly security assessment due for all AI employees',
    timestamp: new Date(Date.now() - 3 * 60 * 60 * 1000),
    source: 'Security Manager',
    severity: 'medium',
    status: 'active',
    actionRequired: true,
    category: 'security'
  },
  {
    id: 'alert-8',
    type: 'info',
    title: 'Performance Report Generated',
    message: 'Weekly performance report is ready for review',
    timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000),
    source: 'Report Generator',
    severity: 'low',
    status: 'active',
    category: 'system'
  }
]

export function AlertsPanel({ metrics }: AlertsPanelProps) {
  const [alerts, setAlerts] = useState<Alert[]>(generateMockAlerts())
  const [filterType, setFilterType] = useState<string>('all')
  const [filterSeverity, setFilterSeverity] = useState<string>('all')
  const [filterStatus, setFilterStatus] = useState<string>('active')

  // Simulate real-time alerts
  useEffect(() => {
    const interval = setInterval(() => {
      // Randomly generate new alerts
      if (Math.random() < 0.3) { // 30% chance every 30 seconds
        const alertTypes: Alert['type'][] = ['info', 'warning', 'success', 'error']
        const severities: Alert['severity'][] = ['low', 'medium', 'high']
        const categories: Alert['category'][] = ['performance', 'system', 'security', 'maintenance']
        
        const newAlert: Alert = {
          id: `alert-${Date.now()}`,
          type: alertTypes[Math.floor(Math.random() * alertTypes.length)],
          title: 'New System Alert',
          message: 'Automated system notification generated',
          timestamp: new Date(),
          source: 'System Monitor',
          severity: severities[Math.floor(Math.random() * severities.length)],
          status: 'active',
          category: categories[Math.floor(Math.random() * categories.length)]
        }
        
        setAlerts(prev => [newAlert, ...prev.slice(0, 19)]) // Keep last 20 alerts
      }
    }, 30000) // Check every 30 seconds

    return () => clearInterval(interval)
  }, [])

  const getAlertIcon = (type: Alert['type']) => {
    switch (type) {
      case 'error':
        return <XCircle className="h-5 w-5 text-red-500" />
      case 'warning':
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />
      case 'info':
        return <Info className="h-5 w-5 text-blue-500" />
      case 'success':
        return <CheckCircle className="h-5 w-5 text-green-500" />
    }
  }

  const getAlertColor = (type: Alert['type']) => {
    switch (type) {
      case 'error':
        return 'border-red-200 bg-red-50'
      case 'warning':
        return 'border-yellow-200 bg-yellow-50'
      case 'info':
        return 'border-blue-200 bg-blue-50'
      case 'success':
        return 'border-green-200 bg-green-50'
    }
  }

  const getSeverityColor = (severity: Alert['severity']) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-500 text-white'
      case 'high':
        return 'bg-red-100 text-red-800'
      case 'medium':
        return 'bg-yellow-100 text-yellow-800'
      case 'low':
        return 'bg-gray-100 text-gray-800'
    }
  }

  const getStatusColor = (status: Alert['status']) => {
    switch (status) {
      case 'active':
        return 'bg-blue-100 text-blue-800'
      case 'acknowledged':
        return 'bg-yellow-100 text-yellow-800'
      case 'resolved':
        return 'bg-green-100 text-green-800'
    }
  }

  const getCategoryIcon = (category: Alert['category']) => {
    switch (category) {
      case 'performance':
        return <TrendingUp className="h-4 w-4" />
      case 'system':
        return <Activity className="h-4 w-4" />
      case 'security':
        return <Shield className="h-4 w-4" />
      case 'maintenance':
        return <Bot className="h-4 w-4" />
      case 'compliance':
        return <CheckCircle className="h-4 w-4" />
    }
  }

  const filteredAlerts = alerts.filter(alert => {
    const matchesType = filterType === 'all' || alert.type === filterType
    const matchesSeverity = filterSeverity === 'all' || alert.severity === filterSeverity
    const matchesStatus = filterStatus === 'all' || alert.status === filterStatus
    return matchesType && matchesSeverity && matchesStatus
  })

  const handleAcknowledge = (alertId: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, status: 'acknowledged' as const } : alert
    ))
  }

  const handleResolve = (alertId: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, status: 'resolved' as const } : alert
    ))
  }

  const handleDismiss = (alertId: string) => {
    setAlerts(prev => prev.filter(alert => alert.id !== alertId))
  }

  const alertCounts = {
    total: alerts.length,
    active: alerts.filter(a => a.status === 'active').length,
    critical: alerts.filter(a => a.severity === 'critical').length,
    actionRequired: alerts.filter(a => a.actionRequired && a.status === 'active').length
  }

  return (
    <div className="space-y-6">
      {/* Alert Summary */}
      <div className="grid md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Alerts</p>
                <p className="text-2xl font-bold">{alertCounts.total}</p>
              </div>
              <Bell className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Active Alerts</p>
                <p className="text-2xl font-bold text-blue-600">{alertCounts.active}</p>
              </div>
              <Activity className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Critical</p>
                <p className="text-2xl font-bold text-red-600">{alertCounts.critical}</p>
              </div>
              <AlertTriangle className="h-8 w-8 text-red-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Action Required</p>
                <p className="text-2xl font-bold text-orange-600">{alertCounts.actionRequired}</p>
              </div>
              <Zap className="h-8 w-8 text-orange-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Bell className="h-5 w-5 text-blue-500" />
            <span>System Alerts & Notifications</span>
          </CardTitle>
          <CardDescription>
            Monitor system health and performance alerts
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col md:flex-row gap-4">
            <Select value={filterType} onValueChange={setFilterType}>
              <SelectTrigger className="w-48">
                <Filter className="h-4 w-4 mr-2" />
                <SelectValue placeholder="Filter by type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="error">Errors</SelectItem>
                <SelectItem value="warning">Warnings</SelectItem>
                <SelectItem value="info">Information</SelectItem>
                <SelectItem value="success">Success</SelectItem>
              </SelectContent>
            </Select>

            <Select value={filterSeverity} onValueChange={setFilterSeverity}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="Filter by severity" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Severities</SelectItem>
                <SelectItem value="critical">Critical</SelectItem>
                <SelectItem value="high">High</SelectItem>
                <SelectItem value="medium">Medium</SelectItem>
                <SelectItem value="low">Low</SelectItem>
              </SelectContent>
            </Select>

            <Select value={filterStatus} onValueChange={setFilterStatus}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="Filter by status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="acknowledged">Acknowledged</SelectItem>
                <SelectItem value="resolved">Resolved</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Alerts List */}
      <div className="space-y-4">
        <AnimatePresence>
          {filteredAlerts.map((alert, index) => (
            <motion.div
              key={alert.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
            >
              <Card className={`${getAlertColor(alert.type)} border-l-4`}>
                <CardContent className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-3 flex-1">
                      {getAlertIcon(alert.type)}
                      
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <h4 className="font-semibold">{alert.title}</h4>
                          <Badge className={getSeverityColor(alert.severity)}>
                            {alert.severity.toUpperCase()}
                          </Badge>
                          <Badge className={getStatusColor(alert.status)}>
                            {alert.status.toUpperCase()}
                          </Badge>
                          {alert.actionRequired && (
                            <Badge className="bg-orange-100 text-orange-800">
                              <Zap className="h-3 w-3 mr-1" />
                              Action Required
                            </Badge>
                          )}
                        </div>
                        
                        <p className="text-sm text-muted-foreground mb-2">
                          {alert.message}
                        </p>
                        
                        <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                          <div className="flex items-center space-x-1">
                            <Clock className="h-3 w-3" />
                            <span>{alert.timestamp.toLocaleString()}</span>
                          </div>
                          <div className="flex items-center space-x-1">
                            {getCategoryIcon(alert.category)}
                            <span>{alert.category}</span>
                          </div>
                          <span>Source: {alert.source}</span>
                          {alert.relatedEmployee && (
                            <span>Employee: {alert.relatedEmployee}</span>
                          )}
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2 ml-4">
                      {alert.status === 'active' && (
                        <>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleAcknowledge(alert.id)}
                          >
                            <Eye className="h-4 w-4 mr-1" />
                            Acknowledge
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleResolve(alert.id)}
                          >
                            <CheckCircle className="h-4 w-4 mr-1" />
                            Resolve
                          </Button>
                        </>
                      )}
                      {alert.status === 'acknowledged' && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleResolve(alert.id)}
                        >
                          <CheckCircle className="h-4 w-4 mr-1" />
                          Resolve
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleDismiss(alert.id)}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </AnimatePresence>
        
        {filteredAlerts.length === 0 && (
          <Card>
            <CardContent className="text-center py-12">
              <CheckCircle className="h-16 w-16 mx-auto mb-4 text-green-500" />
              <h3 className="text-lg font-semibold mb-2">No alerts found</h3>
              <p className="text-muted-foreground">
                {filterStatus === 'active' 
                  ? 'All systems are running smoothly!' 
                  : 'No alerts match your current filters.'
                }
              </p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}