'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { 
  Bot, 
  Activity, 
  CheckCircle, 
  Clock, 
  TrendingUp, 
  TrendingDown,
  Zap,
  Target,
  AlertTriangle,
  Pause,
  Play,
  Settings,
  MoreHorizontal,
  Search,
  Filter,
  Eye,
  BarChart3
} from 'lucide-react'

interface EmployeeListProps {
  selectedEmployee: string
  onEmployeeSelect: (employeeId: string) => void
}

interface AIEmployee {
  id: string
  name: string
  role: string
  status: 'active' | 'idle' | 'maintenance' | 'error'
  avatar: string
  tasksCompleted: number
  tasksInProgress: number
  accuracy: number
  speed: number
  uptime: number
  lastActive: string
  specialties: string[]
  performance: {
    trend: 'up' | 'down' | 'stable'
    change: number
  }
  currentTask?: string
  estimatedCompletion?: string
}

const mockEmployees: AIEmployee[] = [
  {
    id: 'sarah-cs',
    name: 'Sarah',
    role: 'Customer Service Specialist',
    status: 'active',
    avatar: '/avatars/sarah.jpg',
    tasksCompleted: 1247,
    tasksInProgress: 3,
    accuracy: 98.5,
    speed: 9.2,
    uptime: 99.8,
    lastActive: '2 minutes ago',
    specialties: ['Customer Support', 'Issue Resolution', 'Live Chat'],
    performance: { trend: 'up', change: 5.2 },
    currentTask: 'Resolving billing inquiry',
    estimatedCompletion: '2 min'
  },
  {
    id: 'alex-da',
    name: 'Alex',
    role: 'Data Analyst',
    status: 'active',
    avatar: '/avatars/alex.jpg',
    tasksCompleted: 892,
    tasksInProgress: 2,
    accuracy: 97.8,
    speed: 8.7,
    uptime: 99.5,
    lastActive: '5 minutes ago',
    specialties: ['Data Analysis', 'Report Generation', 'SQL Queries'],
    performance: { trend: 'up', change: 3.1 },
    currentTask: 'Generating sales report',
    estimatedCompletion: '8 min'
  },
  {
    id: 'maya-cw',
    name: 'Maya',
    role: 'Content Writer',
    status: 'active',
    avatar: '/avatars/maya.jpg',
    tasksCompleted: 654,
    tasksInProgress: 1,
    accuracy: 96.9,
    speed: 7.8,
    uptime: 98.9,
    lastActive: '1 minute ago',
    specialties: ['Content Creation', 'SEO Writing', 'Blog Posts'],
    performance: { trend: 'stable', change: 0.5 },
    currentTask: 'Writing product description',
    estimatedCompletion: '15 min'
  },
  {
    id: 'jordan-sa',
    name: 'Jordan',
    role: 'Sales Assistant',
    status: 'idle',
    avatar: '/avatars/jordan.jpg',
    tasksCompleted: 1156,
    tasksInProgress: 0,
    accuracy: 95.2,
    speed: 8.1,
    uptime: 97.8,
    lastActive: '12 minutes ago',
    specialties: ['Lead Qualification', 'CRM Updates', 'Follow-ups'],
    performance: { trend: 'down', change: -1.2 }
  },
  {
    id: 'sam-pm',
    name: 'Sam',
    role: 'Project Manager',
    status: 'active',
    avatar: '/avatars/sam.jpg',
    tasksCompleted: 423,
    tasksInProgress: 4,
    accuracy: 99.1,
    speed: 6.9,
    uptime: 99.9,
    lastActive: 'Just now',
    specialties: ['Project Planning', 'Task Management', 'Team Coordination'],
    performance: { trend: 'up', change: 7.8 },
    currentTask: 'Updating project timeline',
    estimatedCompletion: '5 min'
  },
  {
    id: 'riley-hr',
    name: 'Riley',
    role: 'HR Assistant',
    status: 'maintenance',
    avatar: '/avatars/riley.jpg',
    tasksCompleted: 789,
    tasksInProgress: 0,
    accuracy: 98.7,
    speed: 7.5,
    uptime: 95.2,
    lastActive: '2 hours ago',
    specialties: ['Recruitment', 'Employee Onboarding', 'Policy Updates'],
    performance: { trend: 'stable', change: 0.8 }
  }
]

export function EmployeeList({ selectedEmployee, onEmployeeSelect }: EmployeeListProps) {
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [sortBy, setSortBy] = useState('performance')

  const getStatusIcon = (status: AIEmployee['status']) => {
    switch (status) {
      case 'active':
        return <Activity className="h-4 w-4 text-green-500" />
      case 'idle':
        return <Pause className="h-4 w-4 text-yellow-500" />
      case 'maintenance':
        return <Settings className="h-4 w-4 text-blue-500" />
      case 'error':
        return <AlertTriangle className="h-4 w-4 text-red-500" />
    }
  }

  const getStatusColor = (status: AIEmployee['status']) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800 border-green-200'
      case 'idle':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      case 'maintenance':
        return 'bg-blue-100 text-blue-800 border-blue-200'
      case 'error':
        return 'bg-red-100 text-red-800 border-red-200'
    }
  }

  const getTrendIcon = (trend: 'up' | 'down' | 'stable') => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="h-4 w-4 text-green-500" />
      case 'down':
        return <TrendingDown className="h-4 w-4 text-red-500" />
      case 'stable':
        return <BarChart3 className="h-4 w-4 text-gray-500" />
    }
  }

  const filteredEmployees = mockEmployees
    .filter(employee => {
      const matchesSearch = employee.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           employee.role.toLowerCase().includes(searchTerm.toLowerCase())
      const matchesStatus = statusFilter === 'all' || employee.status === statusFilter
      return matchesSearch && matchesStatus
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'performance':
          return (b.accuracy * b.speed) - (a.accuracy * a.speed)
        case 'tasks':
          return b.tasksCompleted - a.tasksCompleted
        case 'accuracy':
          return b.accuracy - a.accuracy
        case 'speed':
          return b.speed - a.speed
        default:
          return 0
      }
    })

  return (
    <div className="space-y-6">
      {/* Filters and Search */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Bot className="h-5 w-5 text-blue-500" />
              <span>AI Employee Management</span>
            </div>
            <Badge variant="outline">
              {filteredEmployees.length} employees
            </Badge>
          </CardTitle>
          <CardDescription>
            Monitor and manage your AI workforce
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col md:flex-row gap-4">
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
              <SelectTrigger className="w-48">
                <Filter className="h-4 w-4 mr-2" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="idle">Idle</SelectItem>
                <SelectItem value="maintenance">Maintenance</SelectItem>
                <SelectItem value="error">Error</SelectItem>
              </SelectContent>
            </Select>
            
            <Select value={sortBy} onValueChange={setSortBy}>
              <SelectTrigger className="w-48">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="performance">Performance</SelectItem>
                <SelectItem value="tasks">Tasks Completed</SelectItem>
                <SelectItem value="accuracy">Accuracy</SelectItem>
                <SelectItem value="speed">Speed</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Employee Grid */}
      <div className="grid lg:grid-cols-2 gap-6">
        {filteredEmployees.map((employee, index) => (
          <motion.div
            key={employee.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
          >
            <Card 
              className={`card-hover cursor-pointer transition-all ${
                selectedEmployee === employee.id ? 'ring-2 ring-forge-500' : ''
              }`}
              onClick={() => onEmployeeSelect(employee.id)}
            >
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <Avatar className="h-12 w-12">
                      <AvatarImage src={employee.avatar} alt={employee.name} />
                      <AvatarFallback>
                        {employee.name.split(' ').map(n => n[0]).join('')}
                      </AvatarFallback>
                    </Avatar>
                    <div>
                      <h3 className="font-semibold">{employee.name}</h3>
                      <p className="text-sm text-muted-foreground">{employee.role}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <Badge className={getStatusColor(employee.status)}>
                      {getStatusIcon(employee.status)}
                      <span className="ml-1 capitalize">{employee.status}</span>
                    </Badge>
                    <Button variant="ghost" size="sm">
                      <MoreHorizontal className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="space-y-4">
                {/* Current Task */}
                {employee.currentTask && (
                  <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-blue-900">Current Task</p>
                        <p className="text-sm text-blue-700">{employee.currentTask}</p>
                      </div>
                      {employee.estimatedCompletion && (
                        <Badge variant="outline" className="text-blue-600 border-blue-300">
                          <Clock className="h-3 w-3 mr-1" />
                          {employee.estimatedCompletion}
                        </Badge>
                      )}
                    </div>
                  </div>
                )}
                
                {/* Performance Metrics */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-muted-foreground">Accuracy</span>
                      <div className="flex items-center space-x-1">
                        <span className="text-sm font-medium">{employee.accuracy}%</span>
                        {getTrendIcon(employee.performance.trend)}
                      </div>
                    </div>
                    <Progress value={employee.accuracy} className="h-2" />
                  </div>
                  
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-muted-foreground">Speed</span>
                      <div className="flex items-center space-x-1">
                        <span className="text-sm font-medium">{employee.speed}x</span>
                        <Zap className="h-3 w-3 text-yellow-500" />
                      </div>
                    </div>
                    <Progress value={(employee.speed / 10) * 100} className="h-2" />
                  </div>
                </div>
                
                {/* Task Statistics */}
                <div className="grid grid-cols-3 gap-4 pt-2 border-t">
                  <div className="text-center">
                    <div className="text-lg font-bold text-green-600">
                      {employee.tasksCompleted.toLocaleString()}
                    </div>
                    <div className="text-xs text-muted-foreground">Completed</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-lg font-bold text-blue-600">
                      {employee.tasksInProgress}
                    </div>
                    <div className="text-xs text-muted-foreground">In Progress</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-lg font-bold text-purple-600">
                      {employee.uptime}%
                    </div>
                    <div className="text-xs text-muted-foreground">Uptime</div>
                  </div>
                </div>
                
                {/* Specialties */}
                <div>
                  <p className="text-sm text-muted-foreground mb-2">Specialties</p>
                  <div className="flex flex-wrap gap-1">
                    {employee.specialties.map((specialty, idx) => (
                      <Badge key={idx} variant="secondary" className="text-xs">
                        {specialty}
                      </Badge>
                    ))}
                  </div>
                </div>
                
                {/* Performance Change */}
                <div className="flex items-center justify-between pt-2 border-t">
                  <span className="text-sm text-muted-foreground">
                    Last active: {employee.lastActive}
                  </span>
                  <div className={`flex items-center space-x-1 text-sm ${
                    employee.performance.trend === 'up' ? 'text-green-600' : 
                    employee.performance.trend === 'down' ? 'text-red-600' : 'text-gray-600'
                  }`}>
                    {getTrendIcon(employee.performance.trend)}
                    <span>{Math.abs(employee.performance.change).toFixed(1)}%</span>
                  </div>
                </div>
                
                {/* Action Buttons */}
                <div className="flex space-x-2 pt-2">
                  <Button variant="outline" size="sm" className="flex-1">
                    <Eye className="h-4 w-4 mr-2" />
                    View Details
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm"
                    disabled={employee.status === 'maintenance'}
                  >
                    {employee.status === 'active' ? (
                      <Pause className="h-4 w-4" />
                    ) : (
                      <Play className="h-4 w-4" />
                    )}
                  </Button>
                  <Button variant="outline" size="sm">
                    <Settings className="h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>
      
      {filteredEmployees.length === 0 && (
        <Card>
          <CardContent className="text-center py-12">
            <Bot className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
            <h3 className="text-lg font-semibold mb-2">No employees found</h3>
            <p className="text-muted-foreground">
              Try adjusting your search or filter criteria
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}