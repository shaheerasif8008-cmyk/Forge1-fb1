'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
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
  TrendingUp
} from 'lucide-react'

const complianceFrameworks = {
  gdpr: {
    name: 'GDPR',
    fullName: 'General Data Protection Regulation',
    status: 'compliant',
    score: 98.5,
    lastAudit: '2024-11-15',
    nextAudit: '2024-12-15',
    requirements: [
      { id: 'data-processing', name: 'Data Processing Records', status: 'compliant', score: 100 },
      { id: 'consent-management', name: 'Consent Management', status: 'compliant', score: 98 },
      { id: 'data-portability', name: 'Data Portability', status: 'compliant', score: 97 },
      { id: 'right-to-erasure', name: 'Right to Erasure', status: 'compliant', score: 99 },
      { id: 'privacy-by-design', name: 'Privacy by Design', status: 'warning', score: 95 }
    ]
  },
  hipaa: {
    name: 'HIPAA',
    fullName: 'Health Insurance Portability and Accountability Act',
    status: 'compliant',
    score: 96.8,
    lastAudit: '2024-11-10',
    nextAudit: '2024-12-10',
    requirements: [
      { id: 'access-controls', name: 'Access Controls', status: 'compliant', score: 98 },
      { id: 'audit-logs', name: 'Audit Logs', status: 'compliant', score: 97 },
      { id: 'encryption', name: 'Data Encryption', status: 'compliant', score: 99 },
      { id: 'breach-notification', name: 'Breach Notification', status: 'compliant', score: 95 },
      { id: 'business-associate', name: 'Business Associate Agreements', status: 'warning', score: 94 }
    ]
  },
  sox: {
    name: 'SOX',
    fullName: 'Sarbanes-Oxley Act',
    status: 'compliant',
    score: 97.2,
    lastAudit: '2024-11-08',
    nextAudit: '2024-12-08',
    requirements: [
      { id: 'financial-controls', name: 'Financial Controls', status: 'compliant', score: 98 },
      { id: 'audit-trail', name: 'Audit Trail', status: 'compliant', score: 97 },
      { id: 'segregation-duties', name: 'Segregation of Duties', status: 'compliant', score: 96 },
      { id: 'change-management', name: 'Change Management', status: 'compliant', score: 98 },
      { id: 'documentation', name: 'Documentation Requirements', status: 'compliant', score: 97 }
    ]
  },
  pci: {
    name: 'PCI DSS',
    fullName: 'Payment Card Industry Data Security Standard',
    status: 'warning',
    score: 92.1,
    lastAudit: '2024-11-12',
    nextAudit: '2024-12-12',
    requirements: [
      { id: 'network-security', name: 'Network Security', status: 'compliant', score: 95 },
      { id: 'cardholder-data', name: 'Cardholder Data Protection', status: 'warning', score: 88 },
      { id: 'vulnerability-management', name: 'Vulnerability Management', status: 'compliant', score: 94 },
      { id: 'access-control', name: 'Access Control Measures', status: 'compliant', score: 93 },
      { id: 'monitoring', name: 'Network Monitoring', status: 'warning', score: 90 }
    ]
  }
}

const auditHistory = [
  {
    id: 1,
    framework: 'GDPR',
    date: '2024-11-15',
    auditor: 'ComplianceAudit Pro',
    score: 98.5,
    status: 'passed',
    findings: 2,
    recommendations: 3
  },
  {
    id: 2,
    framework: 'HIPAA',
    date: '2024-11-10',
    auditor: 'HealthCare Compliance Inc',
    score: 96.8,
    status: 'passed',
    findings: 3,
    recommendations: 5
  },
  {
    id: 3,
    framework: 'SOX',
    date: '2024-11-08',
    auditor: 'Financial Audit Services',
    score: 97.2,
    status: 'passed',
    findings: 1,
    recommendations: 2
  },
  {
    id: 4,
    framework: 'PCI DSS',
    date: '2024-11-12',
    auditor: 'Payment Security Auditors',
    score: 92.1,
    status: 'conditional',
    findings: 5,
    recommendations: 8
  }
]

const complianceAlerts = [
  {
    id: 1,
    type: 'warning',
    framework: 'PCI DSS',
    title: 'Cardholder Data Encryption Review Required',
    description: 'Annual review of cardholder data encryption methods due within 7 days',
    dueDate: '2024-12-01',
    priority: 'high'
  },
  {
    id: 2,
    type: 'info',
    framework: 'GDPR',
    title: 'Privacy Impact Assessment Scheduled',
    description: 'Quarterly privacy impact assessment scheduled for next week',
    dueDate: '2024-12-05',
    priority: 'medium'
  },
  {
    id: 3,
    type: 'success',
    framework: 'HIPAA',
    title: 'Security Training Completed',
    description: 'All staff completed mandatory HIPAA security training',
    dueDate: '2024-11-20',
    priority: 'low'
  }
]

export function ComplianceMonitorWidget() {
  const [selectedFramework, setSelectedFramework] = useState('gdpr')

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'compliant': return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'warning': return <AlertTriangle className="h-4 w-4 text-yellow-500" />
      case 'non-compliant': return <XCircle className="h-4 w-4 text-red-500" />
      default: return <Activity className="h-4 w-4 text-gray-500" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'compliant': return 'text-green-600'
      case 'warning': return 'text-yellow-600'
      case 'non-compliant': return 'text-red-600'
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
      case 'success': return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'info': return <Activity className="h-4 w-4 text-blue-500" />
      default: return <Activity className="h-4 w-4 text-gray-500" />
    }
  }

  const overallComplianceScore = Object.values(complianceFrameworks)
    .reduce((sum, framework) => sum + framework.score, 0) / Object.keys(complianceFrameworks).length

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Shield className="h-5 w-5" />
          <span>Compliance Monitor</span>
        </CardTitle>
        <CardDescription>
          Monitor regulatory compliance across all frameworks
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="frameworks">Frameworks</TabsTrigger>
            <TabsTrigger value="audits">Audits</TabsTrigger>
            <TabsTrigger value="alerts">Alerts</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            {/* Overall Compliance Score */}
            <div className="text-center p-6 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg">
              <div className="text-4xl font-bold text-green-600 mb-2">
                {overallComplianceScore.toFixed(1)}%
              </div>
              <div className="text-lg font-semibold mb-1">Overall Compliance Score</div>
              <div className="text-sm text-muted-foreground">
                Across all regulatory frameworks
              </div>
              <div className="flex items-center justify-center mt-3">
                <TrendingUp className="h-4 w-4 text-green-500 mr-1" />
                <span className="text-sm text-green-600">+2.3% vs last month</span>
              </div>
            </div>

            {/* Framework Status Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(complianceFrameworks).map(([key, framework]) => (
                <div key={key} className="p-4 border rounded-lg text-center">
                  <div className="flex items-center justify-center mb-2">
                    {getStatusIcon(framework.status)}
                  </div>
                  <div className="font-semibold">{framework.name}</div>
                  <div className={`text-2xl font-bold ${getScoreColor(framework.score)}`}>
                    {framework.score.toFixed(1)}%
                  </div>
                  <Badge 
                    variant={framework.status === 'compliant' ? 'default' : 'secondary'}
                    className="mt-2"
                  >
                    {framework.status}
                  </Badge>
                </div>
              ))}
            </div>

            {/* Recent Activity */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Recent Compliance Activity</h3>
              <div className="space-y-3">
                {auditHistory.slice(0, 3).map((audit) => (
                  <div key={audit.id} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 rounded-full ${
                        audit.status === 'passed' ? 'bg-green-100' : 'bg-yellow-100'
                      }`}>
                        {audit.status === 'passed' ? 
                          <CheckCircle className="h-4 w-4 text-green-600" /> :
                          <AlertTriangle className="h-4 w-4 text-yellow-600" />
                        }
                      </div>
                      <div>
                        <div className="font-medium">{audit.framework} Audit</div>
                        <div className="text-sm text-muted-foreground">
                          {audit.auditor} • {audit.date}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`font-bold ${getScoreColor(audit.score)}`}>
                        {audit.score}%
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {audit.findings} findings
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </TabsContent>

          {/* Frameworks Tab */}
          <TabsContent value="frameworks" className="space-y-6">
            <div className="flex space-x-4">
              <div className="w-64 space-y-2">
                {Object.entries(complianceFrameworks).map(([key, framework]) => (
                  <button
                    key={key}
                    onClick={() => setSelectedFramework(key)}
                    className={`w-full p-3 text-left border rounded-lg transition-colors ${
                      selectedFramework === key ? 'border-primary bg-primary/5' : 'hover:bg-muted/50'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium">{framework.name}</div>
                        <div className="text-sm text-muted-foreground">
                          {framework.fullName}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`font-bold ${getScoreColor(framework.score)}`}>
                          {framework.score.toFixed(1)}%
                        </div>
                        {getStatusIcon(framework.status)}
                      </div>
                    </div>
                  </button>
                ))}
              </div>

              <div className="flex-1">
                {selectedFramework && (
                  <div className="space-y-4">
                    <div className="p-4 border rounded-lg">
                      <h3 className="text-lg font-semibold mb-4">
                        {complianceFrameworks[selectedFramework as keyof typeof complianceFrameworks].fullName}
                      </h3>
                      
                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <div>
                          <div className="text-sm text-muted-foreground">Overall Score</div>
                          <div className={`text-2xl font-bold ${getScoreColor(complianceFrameworks[selectedFramework as keyof typeof complianceFrameworks].score)}`}>
                            {complianceFrameworks[selectedFramework as keyof typeof complianceFrameworks].score.toFixed(1)}%
                          </div>
                        </div>
                        <div>
                          <div className="text-sm text-muted-foreground">Status</div>
                          <Badge variant={complianceFrameworks[selectedFramework as keyof typeof complianceFrameworks].status === 'compliant' ? 'default' : 'secondary'}>
                            {complianceFrameworks[selectedFramework as keyof typeof complianceFrameworks].status}
                          </Badge>
                        </div>
                        <div>
                          <div className="text-sm text-muted-foreground">Last Audit</div>
                          <div className="font-medium">
                            {complianceFrameworks[selectedFramework as keyof typeof complianceFrameworks].lastAudit}
                          </div>
                        </div>
                        <div>
                          <div className="text-sm text-muted-foreground">Next Audit</div>
                          <div className="font-medium">
                            {complianceFrameworks[selectedFramework as keyof typeof complianceFrameworks].nextAudit}
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-3">
                      <h4 className="font-semibold">Requirements Compliance</h4>
                      {complianceFrameworks[selectedFramework as keyof typeof complianceFrameworks].requirements.map((req) => (
                        <div key={req.id} className="p-3 border rounded-lg">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center space-x-2">
                              {getStatusIcon(req.status)}
                              <span className="font-medium">{req.name}</span>
                            </div>
                            <div className={`font-bold ${getScoreColor(req.score)}`}>
                              {req.score}%
                            </div>
                          </div>
                          <Progress value={req.score} className="h-2" />
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </TabsContent>

          {/* Audits Tab */}
          <TabsContent value="audits" className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Audit History</h3>
              <Button variant="outline" size="sm">
                <Download className="h-4 w-4 mr-2" />
                Export Report
              </Button>
            </div>

            <div className="space-y-4">
              {auditHistory.map((audit) => (
                <div key={audit.id} className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 rounded-full ${
                        audit.status === 'passed' ? 'bg-green-100' : 
                        audit.status === 'conditional' ? 'bg-yellow-100' : 'bg-red-100'
                      }`}>
                        {audit.status === 'passed' ? 
                          <CheckCircle className="h-5 w-5 text-green-600" /> :
                          audit.status === 'conditional' ?
                          <AlertTriangle className="h-5 w-5 text-yellow-600" /> :
                          <XCircle className="h-5 w-5 text-red-600" />
                        }
                      </div>
                      <div>
                        <h4 className="font-semibold">{audit.framework} Compliance Audit</h4>
                        <div className="text-sm text-muted-foreground">
                          Conducted by {audit.auditor}
                        </div>
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <div className={`text-2xl font-bold ${getScoreColor(audit.score)}`}>
                        {audit.score}%
                      </div>
                      <Badge variant={audit.status === 'passed' ? 'default' : 'secondary'}>
                        {audit.status}
                      </Badge>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <div className="text-muted-foreground">Audit Date</div>
                      <div className="font-medium">{audit.date}</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Findings</div>
                      <div className="font-medium">{audit.findings} issues</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Recommendations</div>
                      <div className="font-medium">{audit.recommendations} items</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>

          {/* Alerts Tab */}
          <TabsContent value="alerts" className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Compliance Alerts</h3>
              <Badge variant="secondary">{complianceAlerts.length} active</Badge>
            </div>

            <div className="space-y-4">
              {complianceAlerts.map((alert) => (
                <div key={alert.id} className="p-4 border rounded-lg">
                  <div className="flex items-start space-x-3">
                    <div className={`p-2 rounded-full mt-1 ${
                      alert.type === 'warning' ? 'bg-yellow-100' :
                      alert.type === 'success' ? 'bg-green-100' : 'bg-blue-100'
                    }`}>
                      {getAlertIcon(alert.type)}
                    </div>
                    
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <h4 className="font-semibold">{alert.title}</h4>
                        <div className="flex items-center space-x-2">
                          <Badge variant="outline">{alert.framework}</Badge>
                          <Badge variant={
                            alert.priority === 'high' ? 'destructive' :
                            alert.priority === 'medium' ? 'secondary' : 'outline'
                          }>
                            {alert.priority}
                          </Badge>
                        </div>
                      </div>
                      
                      <p className="text-sm text-muted-foreground mt-1">
                        {alert.description}
                      </p>
                      
                      <div className="flex items-center space-x-4 mt-3 text-sm">
                        <div className="flex items-center space-x-1">
                          <Calendar className="h-4 w-4 text-muted-foreground" />
                          <span>Due: {alert.dueDate}</span>
                        </div>
                        
                        <Button variant="outline" size="sm">
                          <Eye className="h-3 w-3 mr-1" />
                          View Details
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}