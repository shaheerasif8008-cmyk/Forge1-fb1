'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { 
  Shield, 
  CheckCircle, 
  AlertTriangle, 
  XCircle,
  FileText,
  Clock,
  Users,
  Lock,
  Eye,
  Download,
  Calendar,
  Activity,
  TrendingUp,
  TrendingDown,
  Search,
  Filter,
  RefreshCw,
  Bell,
  AlertCircle,
  CheckSquare,
  BarChart3,
  PieChart,
  LineChart
} from 'lucide-react'

interface ComplianceFramework {
  name: string
  full_name: string
  status: string
  score: number
  last_audit: string
  next_audit: string
  requirements: Array<{
    id: string
    name: string
    status: string
    score: number
  }>
}

interface AuditEntry {
  id: string
  timestamp: string
  framework: string
  event_type: string
  user_id?: string
  resource: string
  action: string
  result: string
  ip_address?: string
  user_agent?: string
  details?: Record<string, any>
}

interface ComplianceAlert {
  id: string
  type: string
  framework: string
  title: string
  description: string
  priority: string
  due_date?: string
  created_at: string
  resolved_at?: string
  assigned_to?: string
  status: string
}

interface DashboardSummary {
  overall_score: number
  frameworks: {
    total: number
    compliant: number
    warning: number
    non_compliant: number
  }
  alerts: {
    total: number
    open: number
    critical: number
  }
  recent_activity: {
    audits_this_month: number
    reports_generated: number
    violations_resolved: number
  }
  trends: {
    score_change: string
    alerts_trend: string
    compliance_trend: string
  }
}

export function ComplianceDashboard() {
  const [frameworks, setFrameworks] = useState<ComplianceFramework[]>([])
  const [auditTrail, setAuditTrail] = useState<AuditEntry[]>([])
  const [alerts, setAlerts] = useState<ComplianceAlert[]>([])
  const [summary, setSummary] = useState<DashboardSummary | null>(null)
  const [selectedFramework, setSelectedFramework] = useState<string>('all')
  const [auditFilter, setAuditFilter] = useState({
    framework: '',
    startDate: '',
    endDate: '',
    limit: 50
  })
  const [alertFilter, setAlertFilter] = useState({
    status: '',
    priority: '',
    framework: ''
  })
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)

  useEffect(() => {
    loadDashboardData()
  }, [])

  const loadDashboardData = async () => {
    try {
      setLoading(true)
      
      // Load all data in parallel
      const [frameworksRes, auditRes, alertsRes, summaryRes] = await Promise.all([
        fetch('/api/v1/compliance/frameworks'),
        fetch('/api/v1/compliance/audit-trail?limit=100'),
        fetch('/api/v1/compliance/alerts?limit=50'),
        fetch('/api/v1/compliance/dashboard/summary')
      ])

      if (frameworksRes.ok) {
        const frameworksData = await frameworksRes.json()
        setFrameworks(frameworksData)
      }

      if (auditRes.ok) {
        const auditData = await auditRes.json()
        setAuditTrail(auditData)
      }

      if (alertsRes.ok) {
        const alertsData = await alertsRes.json()
        setAlerts(alertsData)
      }

      if (summaryRes.ok) {
        const summaryData = await summaryRes.json()
        setSummary(summaryData)
      }

    } catch (error) {
      console.error('Failed to load dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  const refreshData = async () => {
    setRefreshing(true)
    await loadDashboardData()
    setRefreshing(false)
  }

  const resolveAlert = async (alertId: string) => {
    try {
      const response = await fetch(`/api/v1/compliance/alerts/${alertId}/resolve`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          resolution_notes: 'Resolved from dashboard'
        })
      })

      if (response.ok) {
        // Refresh alerts
        const alertsRes = await fetch('/api/v1/compliance/alerts?limit=50')
        if (alertsRes.ok) {
          const alertsData = await alertsRes.json()
          setAlerts(alertsData)
        }
      }
    } catch (error) {
      console.error('Failed to resolve alert:', error)
    }
  }

  const generateReport = async (framework: string) => {
    try {
      const response = await fetch('/api/v1/compliance/reports/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          framework,
          report_type: 'monthly_audit',
          period_start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
          period_end: new Date().toISOString().split('T')[0]
        })
      })

      if (response.ok) {
        const result = await response.json()
        console.log('Report generation started:', result)
        // In production, this would show a success message
      }
    } catch (error) {
      console.error('Failed to generate report:', error)
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'compliant': return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'warning': return <AlertTriangle className="h-4 w-4 text-yellow-500" />
      case 'non_compliant': return <XCircle className="h-4 w-4 text-red-500" />
      default: return <Activity className="h-4 w-4 text-gray-500" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'compliant': return 'text-green-600'
      case 'warning': return 'text-yellow-600'
      case 'non_compliant': return 'text-red-600'
      default: return 'text-gray-600'
    }
  }

  const getScoreColor = (score: number) => {
    if (score >= 95) return 'text-green-600'
    if (score >= 90) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'warning': return <AlertTriangle className="h-4 w-4 text-yellow-500" />
      case 'error': return <XCircle className="h-4 w-4 text-red-500" />
      case 'success': return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'info': return <Activity className="h-4 w-4 text-blue-500" />
      default: return <Activity className="h-4 w-4 text-gray-500" />
    }
  }

  const getPriorityBadgeVariant = (priority: string) => {
    switch (priority) {
      case 'critical': return 'destructive'
      case 'high': return 'destructive'
      case 'medium': return 'secondary'
      case 'low': return 'outline'
      default: return 'outline'
    }
  }

  const filteredAuditTrail = auditTrail.filter(entry => {
    if (auditFilter.framework && entry.framework.toLowerCase() !== auditFilter.framework.toLowerCase()) {
      return false
    }
    if (auditFilter.startDate && entry.timestamp < auditFilter.startDate) {
      return false
    }
    if (auditFilter.endDate && entry.timestamp > auditFilter.endDate) {
      return false
    }
    return true
  })

  const filteredAlerts = alerts.filter(alert => {
    if (alertFilter.status && alert.status !== alertFilter.status) {
      return false
    }
    if (alertFilter.priority && alert.priority !== alertFilter.priority) {
      return false
    }
    if (alertFilter.framework && alert.framework.toLowerCase() !== alertFilter.framework.toLowerCase()) {
      return false
    }
    return true
  })

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Shield className="h-5 w-5" />
            <span>Compliance Dashboard</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center p-8">
            <RefreshCw className="h-6 w-6 animate-spin mr-2" />
            <span>Loading compliance data...</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header with Refresh Button */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Compliance Dashboard</h1>
          <p className="text-muted-foreground">Monitor regulatory compliance across all frameworks</p>
        </div>
        <Button onClick={refreshData} disabled={refreshing} variant="outline">
          <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* Summary Cards */}
      {summary && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Overall Score</CardTitle>
              <Shield className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${getScoreColor(summary.overall_score)}`}>
                {summary.overall_score}%
              </div>
              <div className="flex items-center text-xs text-muted-foreground">
                {summary.trends.score_change.startsWith('+') ? (
                  <TrendingUp className="h-3 w-3 mr-1 text-green-500" />
                ) : (
                  <TrendingDown className="h-3 w-3 mr-1 text-red-500" />
                )}
                {summary.trends.score_change} from last month
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Frameworks</CardTitle>
              <CheckSquare className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{summary.frameworks.compliant}/{summary.frameworks.total}</div>
              <div className="text-xs text-muted-foreground">
                {summary.frameworks.warning} warnings, {summary.frameworks.non_compliant} non-compliant
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
              <Bell className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{summary.alerts.open}</div>
              <div className="text-xs text-muted-foreground">
                {summary.alerts.critical} critical alerts
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">This Month</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{summary.recent_activity.audits_this_month}</div>
              <div className="text-xs text-muted-foreground">
                Audits completed
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Main Dashboard Tabs */}
      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="frameworks">Frameworks</TabsTrigger>
          <TabsTrigger value="audit-trail">Audit Trail</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
          <TabsTrigger value="reports">Reports</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          {/* Framework Status Grid */}
          <Card>
            <CardHeader>
              <CardTitle>Framework Compliance Status</CardTitle>
              <CardDescription>Current status across all regulatory frameworks</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {frameworks.map((framework) => (
                  <div key={framework.name} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(framework.status)}
                        <span className="font-semibold">{framework.name}</span>
                      </div>
                      <div className={`text-lg font-bold ${getScoreColor(framework.score)}`}>
                        {framework.score.toFixed(1)}%
                      </div>
                    </div>
                    <Progress value={framework.score} className="h-2 mb-2" />
                    <div className="flex items-center justify-between text-sm text-muted-foreground">
                      <span>Last audit: {framework.last_audit}</span>
                      <Badge variant={framework.status === 'compliant' ? 'default' : 'secondary'}>
                        {framework.status}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Recent Alerts */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Alerts</CardTitle>
              <CardDescription>Latest compliance alerts requiring attention</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {alerts.slice(0, 5).map((alert) => (
                  <div key={alert.id} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      {getAlertIcon(alert.type)}
                      <div>
                        <div className="font-medium">{alert.title}</div>
                        <div className="text-sm text-muted-foreground">{alert.framework}</div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge variant={getPriorityBadgeVariant(alert.priority)}>
                        {alert.priority}
                      </Badge>
                      {alert.status === 'open' && (
                        <Button size="sm" variant="outline" onClick={() => resolveAlert(alert.id)}>
                          Resolve
                        </Button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Frameworks Tab */}
        <TabsContent value="frameworks" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Compliance Frameworks</CardTitle>
              <CardDescription>Detailed view of all compliance frameworks</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {frameworks.map((framework) => (
                  <div key={framework.name} className="p-6 border rounded-lg">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="text-lg font-semibold">{framework.full_name}</h3>
                        <p className="text-sm text-muted-foreground">
                          Last audit: {framework.last_audit} | Next audit: {framework.next_audit}
                        </p>
                      </div>
                      <div className="text-right">
                        <div className={`text-2xl font-bold ${getScoreColor(framework.score)}`}>
                          {framework.score.toFixed(1)}%
                        </div>
                        <Badge variant={framework.status === 'compliant' ? 'default' : 'secondary'}>
                          {framework.status}
                        </Badge>
                      </div>
                    </div>
                    
                    <div className="space-y-3">
                      <h4 className="font-medium">Requirements</h4>
                      {framework.requirements.map((req) => (
                        <div key={req.id} className="flex items-center justify-between p-3 bg-muted/50 rounded">
                          <div className="flex items-center space-x-2">
                            {getStatusIcon(req.status)}
                            <span>{req.name}</span>
                          </div>
                          <div className={`font-semibold ${getScoreColor(req.score)}`}>
                            {req.score}%
                          </div>
                        </div>
                      ))}
                    </div>
                    
                    <div className="flex justify-end mt-4">
                      <Button variant="outline" onClick={() => generateReport(framework.name)}>
                        <FileText className="h-4 w-4 mr-2" />
                        Generate Report
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Audit Trail Tab */}
        <TabsContent value="audit-trail" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Audit Trail</CardTitle>
              <CardDescription>Complete audit log of compliance-related activities</CardDescription>
            </CardHeader>
            <CardContent>
              {/* Filters */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <div>
                  <Label htmlFor="framework-filter">Framework</Label>
                  <Select value={auditFilter.framework} onValueChange={(value) => 
                    setAuditFilter(prev => ({ ...prev, framework: value }))
                  }>
                    <SelectTrigger>
                      <SelectValue placeholder="All frameworks" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">All frameworks</SelectItem>
                      {frameworks.map(f => (
                        <SelectItem key={f.name} value={f.name}>{f.name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label htmlFor="start-date">Start Date</Label>
                  <Input
                    type="date"
                    value={auditFilter.startDate}
                    onChange={(e) => setAuditFilter(prev => ({ ...prev, startDate: e.target.value }))}
                  />
                </div>
                <div>
                  <Label htmlFor="end-date">End Date</Label>
                  <Input
                    type="date"
                    value={auditFilter.endDate}
                    onChange={(e) => setAuditFilter(prev => ({ ...prev, endDate: e.target.value }))}
                  />
                </div>
                <div className="flex items-end">
                  <Button variant="outline" onClick={() => setAuditFilter({ framework: '', startDate: '', endDate: '', limit: 50 })}>
                    <Filter className="h-4 w-4 mr-2" />
                    Clear Filters
                  </Button>
                </div>
              </div>

              {/* Audit Entries */}
              <ScrollArea className="h-96">
                <div className="space-y-2">
                  {filteredAuditTrail.map((entry) => (
                    <div key={entry.id} className="p-3 border rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-2">
                          <Badge variant="outline">{entry.framework}</Badge>
                          <span className="font-medium">{entry.event_type}</span>
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {new Date(entry.timestamp).toLocaleString()}
                        </div>
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                        <div>
                          <span className="text-muted-foreground">Resource:</span> {entry.resource}
                        </div>
                        <div>
                          <span className="text-muted-foreground">Action:</span> {entry.action}
                        </div>
                        <div>
                          <span className="text-muted-foreground">Result:</span> 
                          <Badge variant={entry.result === 'SUCCESS' ? 'default' : 'destructive'} className="ml-1">
                            {entry.result}
                          </Badge>
                        </div>
                        <div>
                          <span className="text-muted-foreground">User:</span> {entry.user_id || 'System'}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Alerts Tab */}
        <TabsContent value="alerts" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Compliance Alerts</CardTitle>
              <CardDescription>Manage and resolve compliance alerts</CardDescription>
            </CardHeader>
            <CardContent>
              {/* Alert Filters */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div>
                  <Label>Status</Label>
                  <Select value={alertFilter.status} onValueChange={(value) => 
                    setAlertFilter(prev => ({ ...prev, status: value }))
                  }>
                    <SelectTrigger>
                      <SelectValue placeholder="All statuses" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">All statuses</SelectItem>
                      <SelectItem value="open">Open</SelectItem>
                      <SelectItem value="in_progress">In Progress</SelectItem>
                      <SelectItem value="resolved">Resolved</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Priority</Label>
                  <Select value={alertFilter.priority} onValueChange={(value) => 
                    setAlertFilter(prev => ({ ...prev, priority: value }))
                  }>
                    <SelectTrigger>
                      <SelectValue placeholder="All priorities" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">All priorities</SelectItem>
                      <SelectItem value="critical">Critical</SelectItem>
                      <SelectItem value="high">High</SelectItem>
                      <SelectItem value="medium">Medium</SelectItem>
                      <SelectItem value="low">Low</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Framework</Label>
                  <Select value={alertFilter.framework} onValueChange={(value) => 
                    setAlertFilter(prev => ({ ...prev, framework: value }))
                  }>
                    <SelectTrigger>
                      <SelectValue placeholder="All frameworks" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">All frameworks</SelectItem>
                      {frameworks.map(f => (
                        <SelectItem key={f.name} value={f.name}>{f.name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* Alert List */}
              <div className="space-y-4">
                {filteredAlerts.map((alert) => (
                  <div key={alert.id} className="p-4 border rounded-lg">
                    <div className="flex items-start justify-between">
                      <div className="flex items-start space-x-3">
                        {getAlertIcon(alert.type)}
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-1">
                            <h4 className="font-semibold">{alert.title}</h4>
                            <Badge variant="outline">{alert.framework}</Badge>
                            <Badge variant={getPriorityBadgeVariant(alert.priority)}>
                              {alert.priority}
                            </Badge>
                          </div>
                          <p className="text-sm text-muted-foreground mb-2">
                            {alert.description}
                          </p>
                          <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                            <span>Created: {new Date(alert.created_at).toLocaleString()}</span>
                            {alert.due_date && (
                              <span>Due: {alert.due_date}</span>
                            )}
                            {alert.assigned_to && (
                              <span>Assigned: {alert.assigned_to}</span>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge variant={alert.status === 'resolved' ? 'default' : 'secondary'}>
                          {alert.status}
                        </Badge>
                        {alert.status === 'open' && (
                          <Button size="sm" onClick={() => resolveAlert(alert.id)}>
                            Resolve
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Reports Tab */}
        <TabsContent value="reports" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Compliance Reports</CardTitle>
              <CardDescription>Generate and manage compliance reports</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {frameworks.map((framework) => (
                  <div key={framework.name} className="p-4 border rounded-lg">
                    <h3 className="font-semibold mb-2">{framework.name} Report</h3>
                    <p className="text-sm text-muted-foreground mb-4">
                      Generate comprehensive compliance report for {framework.full_name}
                    </p>
                    <div className="space-y-2">
                      <Button 
                        className="w-full" 
                        variant="outline"
                        onClick={() => generateReport(framework.name)}
                      >
                        <FileText className="h-4 w-4 mr-2" />
                        Monthly Report
                      </Button>
                      <Button 
                        className="w-full" 
                        variant="outline"
                        onClick={() => generateReport(framework.name)}
                      >
                        <Download className="h-4 w-4 mr-2" />
                        Audit Summary
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}