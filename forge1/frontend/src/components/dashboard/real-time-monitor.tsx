'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { 
  Activity, 
  Zap, 
  Target, 
  Clock, 
  TrendingUp, 
  AlertTriangle,
  CheckCircle,
  Flame,
  Bot,
  Users
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

interface PerformanceData {
  timestamp: string
  accuracy: number
  speed: number
  throughput: number
  errorRate: number
  responseTime: number
}

interface RealTimeMonitorProps {
  metrics: DashboardMetrics
  performanceData: PerformanceData[]
}

interface LiveActivity {
  id: string
  employee: string
  task: string
  status: 'processing' | 'completed' | 'error'
  timestamp: Date
  duration?: number
}

export function RealTimeMonitor({ metrics, performanceData }: RealTimeMonitorProps) {
  const [liveActivities, setLiveActivities] = useState<LiveActivity[]>([])
  const [currentThroughput, setCurrentThroughput] = useState(0)
  const [systemLoad, setSystemLoad] = useState(0)

  // Simulate live activities
  useEffect(() => {
    const generateActivity = (): LiveActivity => {
      const employees = [
        'Sarah - Customer Service',
        'Alex - Data Analyst', 
        'Maya - Content Writer',
        'Jordan - Sales Assistant',
        'Sam - Project Manager'
      ]
      
      const tasks = [
        'Processing customer inquiry',
        'Analyzing sales data',
        'Writing product description',
        'Qualifying lead',
        'Updating project status',
        'Generating report',
        'Responding to email',
        'Creating content',
        'Processing order',
        'Updating database'
      ]
      
      const statuses: ('processing' | 'completed' | 'error')[] = ['processing', 'completed', 'completed', 'completed', 'error']
      const status = statuses[Math.floor(Math.random() * statuses.length)]
      
      return {
        id: `activity-${Date.now()}-${Math.random()}`,
        employee: employees[Math.floor(Math.random() * employees.length)],
        task: tasks[Math.floor(Math.random() * tasks.length)],
        status,
        timestamp: new Date(),
        duration: status === 'completed' ? Math.floor(Math.random() * 5000) + 500 : undefined
      }
    }

    const interval = setInterval(() => {
      const newActivity = generateActivity()
      
      setLiveActivities(prev => {
        const updated = [newActivity, ...prev.slice(0, 9)] // Keep last 10 activities
        return updated
      })
      
      // Update real-time metrics
      setCurrentThroughput(Math.floor(Math.random() * 20) + 80)
      setSystemLoad(Math.floor(Math.random() * 30) + 20)
    }, 2000 + Math.random() * 3000) // Random interval between 2-5 seconds

    return () => clearInterval(interval)
  }, [])

  const getStatusIcon = (status: LiveActivity['status']) => {
    switch (status) {
      case 'processing':
        return <Activity className="h-4 w-4 text-blue-500 animate-pulse" />
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'error':
        return <AlertTriangle className="h-4 w-4 text-red-500" />
    }
  }

  const getStatusColor = (status: LiveActivity['status']) => {
    switch (status) {
      case 'processing':
        return 'border-blue-200 bg-blue-50'
      case 'completed':
        return 'border-green-200 bg-green-50'
      case 'error':
        return 'border-red-200 bg-red-50'
    }
  }

  const completionRate = (metrics.completedTasks / metrics.totalTasks) * 100
  const activeRate = (metrics.activeEmployees / metrics.totalEmployees) * 100

  return (
    <div className="space-y-6">
      {/* System Status Overview */}
      <div className="grid md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">System Status</p>
                <div className="flex items-center space-x-2 mt-1">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  <span className="font-semibold text-green-600">Operational</span>
                </div>
              </div>
              <Activity className="h-8 w-8 text-green-500" />
            </div>
            <div className="mt-3">
              <div className="flex justify-between text-xs text-muted-foreground mb-1">
                <span>Uptime</span>
                <span>{metrics.uptime}%</span>
              </div>
              <Progress value={metrics.uptime} className="h-2" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Current Throughput</p>
                <p className="text-2xl font-bold">{currentThroughput}/hr</p>
              </div>
              <Zap className="h-8 w-8 text-yellow-500" />
            </div>
            <div className="mt-3">
              <div className="flex justify-between text-xs text-muted-foreground mb-1">
                <span>vs Target (100/hr)</span>
                <span>{currentThroughput}%</span>
              </div>
              <Progress value={currentThroughput} className="h-2" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">System Load</p>
                <p className="text-2xl font-bold">{systemLoad}%</p>
              </div>
              <Target className="h-8 w-8 text-blue-500" />
            </div>
            <div className="mt-3">
              <div className="flex justify-between text-xs text-muted-foreground mb-1">
                <span>Optimal Range</span>
                <span>{'<50%'}</span>
              </div>
              <Progress value={systemLoad} className="h-2" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Performance Indicators */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Flame className="h-5 w-5 text-orange-500" />
            <span>Superhuman Performance Indicators</span>
          </CardTitle>
          <CardDescription>
            Real-time performance metrics showing superhuman capabilities
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-3xl font-bold text-green-600 mb-2">
                {metrics.averageSpeed.toFixed(1)}x
              </div>
              <div className="text-sm text-muted-foreground mb-2">Speed Multiplier</div>
              <Badge className="bg-green-100 text-green-800">
                vs Human Baseline
              </Badge>
            </div>
            
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">
                {metrics.averageAccuracy.toFixed(1)}%
              </div>
              <div className="text-sm text-muted-foreground mb-2">Accuracy Rate</div>
              <Badge className="bg-blue-100 text-blue-800">
                Target: 95%
              </Badge>
            </div>
            
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-600 mb-2">
                {metrics.uptime.toFixed(1)}%
              </div>
              <div className="text-sm text-muted-foreground mb-2">Availability</div>
              <Badge className="bg-purple-100 text-purple-800">
                24/7 Operation
              </Badge>
            </div>
            
            <div className="text-center">
              <div className="text-3xl font-bold text-orange-600 mb-2">
                ${(metrics.totalCostSavings / 1000).toFixed(0)}K
              </div>
              <div className="text-sm text-muted-foreground mb-2">Cost Savings</div>
              <Badge className="bg-orange-100 text-orange-800">
                This Month
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Live Activity Feed */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Bot className="h-5 w-5 text-blue-500" />
              <span>Live AI Employee Activity</span>
            </div>
            <Badge variant="outline" className="animate-pulse">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-2" />
              Live
            </Badge>
          </CardTitle>
          <CardDescription>
            Real-time activity from your AI employees
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            <AnimatePresence>
              {liveActivities.map((activity) => (
                <motion.div
                  key={activity.id}
                  initial={{ opacity: 0, y: -20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 20 }}
                  className={`p-3 rounded-lg border ${getStatusColor(activity.status)}`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      {getStatusIcon(activity.status)}
                      <div>
                        <div className="font-medium text-sm">{activity.employee}</div>
                        <div className="text-sm text-muted-foreground">{activity.task}</div>
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <div className="text-xs text-muted-foreground">
                        {activity.timestamp.toLocaleTimeString()}
                      </div>
                      {activity.duration && (
                        <div className="text-xs text-green-600 font-medium">
                          {(activity.duration / 1000).toFixed(1)}s
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            
            {liveActivities.length === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                <Bot className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Waiting for AI employee activity...</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Performance Summary */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Task Completion</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>Completed Tasks</span>
                  <span>{metrics.completedTasks.toLocaleString()} / {metrics.totalTasks.toLocaleString()}</span>
                </div>
                <Progress value={completionRate} className="h-3" />
                <div className="text-xs text-muted-foreground mt-1">
                  {completionRate.toFixed(1)}% completion rate
                </div>
              </div>
              
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>Active Employees</span>
                  <span>{metrics.activeEmployees} / {metrics.totalEmployees}</span>
                </div>
                <Progress value={activeRate} className="h-3" />
                <div className="text-xs text-muted-foreground mt-1">
                  {activeRate.toFixed(0)}% of AI employees active
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Performance Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Error Rate</span>
                <div className="flex items-center space-x-2">
                  <span className="font-semibold text-green-600">{metrics.errorRate.toFixed(2)}%</span>
                  <CheckCircle className="h-4 w-4 text-green-500" />
                </div>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Customer Satisfaction</span>
                <div className="flex items-center space-x-2">
                  <span className="font-semibold">{metrics.customerSatisfaction.toFixed(1)}%</span>
                  <TrendingUp className="h-4 w-4 text-green-500" />
                </div>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Performance Improvement</span>
                <div className="flex items-center space-x-2">
                  <span className="font-semibold text-blue-600">+{metrics.performanceImprovement.toFixed(1)}%</span>
                  <TrendingUp className="h-4 w-4 text-blue-500" />
                </div>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">ROI</span>
                <div className="flex items-center space-x-2">
                  <span className="font-semibold text-green-600">{metrics.roiPercentage}%</span>
                  <Flame className="h-4 w-4 text-orange-500" />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}