'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Progress } from '@/components/ui/progress'
import { 
  Bot, 
  Search, 
  Filter, 
  MoreVertical, 
  Play, 
  Pause, 
  Settings, 
  TrendingUp,
  TrendingDown,
  Activity,
  Clock,
  CheckCircle,
  AlertTriangle,
  Users,
  Zap
} from 'lucide-react'

interface AIEmployeeStatusGridProps {
  timeRange: string
}

const aiEmployees = [
  {
    id: 'emp-001',
    name: 'Sarah - Customer Support',
    type: 'Customer Service',
    status: 'active',
    performance: 98.2,
    tasksCompleted: 1247,
    currentTask: 'Handling customer inquiry #CS-4521',
    uptime: 99.8,
    responseTime: 1.2,
    accuracy: 97.8,
    lastActive: '2 minutes ago',
    trend: 'up'
  },
  {
    id: 'emp-002',
    name: 'Alex - Data Analyst',
    type: 'Data Analysis',
    status: 'active',
    performance: 96.7,
    tasksCompleted: 892,
    currentTask: 'Processing quarterly sales report',
    uptime: 99.9,
    responseTime: 45.3,
    accuracy: 99.1,
    lastActive: '5 minutes ago',
    trend: 'up'
  },
  {
    id: 'emp-003',
    name: 'Maya - Content Writer',
    type: 'Content Writing',
    status: 'active',
    performance: 94.5,
    tasksCompleted: 634,
    currentTask: 'Writing blog post about AI trends',
    uptime: 98.7,
    responseTime: 180.5,
    accuracy: 95.2,
    lastActive: '1 minute ago',
    trend: 'stable'
  },
  {
    id: 'emp-004',
    name: 'Jordan - Sales Assistant',
    type: 'Sales Support',
    status: 'maintenance',
    performance: 92.1,
    tasksCompleted: 445,
    currentTask: 'Scheduled maintenance',
    uptime: 97.3,
    responseTime: 8.7,
    accuracy: 93.4,
    lastActive: '30 minutes ago',
    trend: 'down'
  },
  {
    id: 'emp-005',
    name: 'Riley - Research Assistant',
    type: 'Research',
    status: 'active',
    performance: 97.9,
    tasksCompleted: 278,
    currentTask: 'Researching market trends for Q4',
    uptime: 99.5,
    responseTime: 120.8,
    accuracy: 98.7,
    lastActive: '8 minutes ago',
    trend: 'up'
  },
  {
    id: 'emp-006',
    name: 'Casey - Project Manager',
    type: 'Project Management',
    status: 'active',
    performance: 95.8,
    tasksCompleted: 356,
    currentTask: 'Coordinating sprint planning',
    uptime: 99.2,
    responseTime: 15.4,
    accuracy: 96.1,
    lastActive: '3 minutes ago',
    trend: 'up'
  }
]

const statusColors = {
  active: 'bg-green-500',
  maintenance: 'bg-yellow-500',
  offline: 'bg-red-500',
  idle: 'bg-gray-500'
}

const statusLabels = {
  active: 'Active',
  maintenance: 'Maintenance',
  offline: 'Offline',
  idle: 'Idle'
}

export function AIEmployeeStatusGrid({ timeRange }: AIEmployeeStatusGridProps) {
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [typeFilter, setTypeFilter] = useState('all')

  const filteredEmployees = aiEmployees.filter(employee => {
    const matchesSearch = employee.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         employee.type.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesStatus = statusFilter === 'all' || employee.status === statusFilter
    const matchesType = typeFilter === 'all' || employee.type === typeFilter
    
    return matchesSearch && matchesStatus && matchesType
  })

  const getPerformanceColor = (performance: number) => {
    if (performance >= 95) return 'text-green-600'
    if (performance >= 90) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <TrendingUp className="h-4 w-4 text-green-500" />
      case 'down': return <TrendingDown className="h-4 w-4 text-red-500" />
      default: return <Activity className="h-4 w-4 text-muted-foreground" />
    }
  }

  const formatResponseTime = (time: number) => {
    if (time < 60) return `${time.toFixed(1)}s`
    if (time < 3600) return `${(time / 60).toFixed(1)}m`
    return `${(time / 3600).toFixed(1)}h`
  }

  const employeeTypes = [...new Set(aiEmployees.map(emp => emp.type))]

  return (
    <div className="space-y-6">
      {/* Filters and Search */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Bot className="h-5 w-5" />
            <span>AI Employee Status</span>
            <Badge variant="secondary">{filteredEmployees.length} employees</Badge>
          </CardTitle>
          <CardDescription>
            Monitor and manage your AI employee workforce
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search employees..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
            
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="maintenance">Maintenance</SelectItem>
                <SelectItem value="offline">Offline</SelectItem>
                <SelectItem value="idle">Idle</SelectItem>
              </SelectContent>
            </Select>
            
            <Select value={typeFilter} onValueChange={setTypeFilter}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="Employee Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                {employeeTypes.map(type => (
                  <SelectItem key={type} value={type}>{type}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Employee Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredEmployees.map((employee) => (
          <Card key={employee.id} className="hover:shadow-lg transition-shadow">
            <CardHeader className="pb-3">
              <div className="flex items-start justify-between">
                <div className="flex items-center space-x-3">
                  <div className="relative">
                    <div className="w-10 h-10 bg-gradient-forge rounded-full flex items-center justify-center">
                      <Bot className="h-5 w-5 text-white" />
                    </div>
                    <div className={`absolute -bottom-1 -right-1 w-4 h-4 ${statusColors[employee.status as keyof typeof statusColors]} rounded-full border-2 border-white`} />
                  </div>
                  <div>
                    <h3 className="font-semibold">{employee.name}</h3>
                    <p className="text-sm text-muted-foreground">{employee.type}</p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  {getTrendIcon(employee.trend)}
                  <Button variant="ghost" size="sm">
                    <MoreVertical className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardHeader>
            
            <CardContent className="space-y-4">
              {/* Status and Performance */}
              <div className="flex items-center justify-between">
                <Badge variant={employee.status === 'active' ? 'default' : 'secondary'}>
                  {statusLabels[employee.status as keyof typeof statusLabels]}
                </Badge>
                <div className="text-right">
                  <div className={`text-lg font-bold ${getPerformanceColor(employee.performance)}`}>
                    {employee.performance}%
                  </div>
                  <div className="text-xs text-muted-foreground">Performance</div>
                </div>
              </div>
              
              {/* Current Task */}
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <Activity className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">Current Task</span>
                </div>
                <p className="text-sm text-muted-foreground pl-6">
                  {employee.currentTask}
                </p>
              </div>
              
              {/* Key Metrics */}
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="flex items-center space-x-1">
                    <CheckCircle className="h-3 w-3 text-green-500" />
                    <span className="text-muted-foreground">Tasks</span>
                  </div>
                  <div className="font-medium">{employee.tasksCompleted.toLocaleString()}</div>
                </div>
                
                <div>
                  <div className="flex items-center space-x-1">
                    <Clock className="h-3 w-3 text-blue-500" />
                    <span className="text-muted-foreground">Response</span>
                  </div>
                  <div className="font-medium">{formatResponseTime(employee.responseTime)}</div>
                </div>
                
                <div>
                  <div className="flex items-center space-x-1">
                    <Zap className="h-3 w-3 text-yellow-500" />
                    <span className="text-muted-foreground">Uptime</span>
                  </div>
                  <div className="font-medium">{employee.uptime}%</div>
                </div>
                
                <div>
                  <div className="flex items-center space-x-1">
                    <TrendingUp className="h-3 w-3 text-purple-500" />
                    <span className="text-muted-foreground">Accuracy</span>
                  </div>
                  <div className="font-medium">{employee.accuracy}%</div>
                </div>
              </div>
              
              {/* Performance Progress */}
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Performance Score</span>
                  <span>{employee.performance}%</span>
                </div>
                <Progress value={employee.performance} className="h-2" />
              </div>
              
              {/* Actions */}
              <div className="flex items-center justify-between pt-2">
                <div className="text-xs text-muted-foreground">
                  Last active: {employee.lastActive}
                </div>
                
                <div className="flex space-x-2">
                  {employee.status === 'active' ? (
                    <Button variant="outline" size="sm">
                      <Pause className="h-3 w-3 mr-1" />
                      Pause
                    </Button>
                  ) : (
                    <Button variant="outline" size="sm">
                      <Play className="h-3 w-3 mr-1" />
                      Start
                    </Button>
                  )}
                  
                  <Button variant="outline" size="sm">
                    <Settings className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
      
      {filteredEmployees.length === 0 && (
        <Card>
          <CardContent className="text-center py-12">
            <Bot className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
            <h3 className="text-lg font-semibold mb-2">No employees found</h3>
            <p className="text-muted-foreground">
              Try adjusting your search or filter criteria
            </p>
          </CardContent>
        </Card>
      )}
      
      {/* Summary Stats */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Users className="h-5 w-5" />
            <span>Workforce Summary</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {aiEmployees.filter(emp => emp.status === 'active').length}
              </div>
              <div className="text-sm text-muted-foreground">Active Employees</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {aiEmployees.reduce((sum, emp) => sum + emp.tasksCompleted, 0).toLocaleString()}
              </div>
              <div className="text-sm text-muted-foreground">Total Tasks Completed</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {(aiEmployees.reduce((sum, emp) => sum + emp.performance, 0) / aiEmployees.length).toFixed(1)}%
              </div>
              <div className="text-sm text-muted-foreground">Average Performance</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {(aiEmployees.reduce((sum, emp) => sum + emp.uptime, 0) / aiEmployees.length).toFixed(1)}%
              </div>
              <div className="text-sm text-muted-foreground">Average Uptime</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}