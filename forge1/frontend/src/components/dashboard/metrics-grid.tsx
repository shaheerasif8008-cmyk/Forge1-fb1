'use client'

import { useMemo } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts'
import { 
  TrendingUp, 
  TrendingDown, 
  Target, 
  Zap, 
  Clock, 
  DollarSign,
  Users,
  CheckCircle,
  AlertTriangle,
  Activity
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

interface MetricsGridProps {
  metrics: DashboardMetrics
  performanceData: PerformanceData[]
  timeRange: string
}

export function MetricsGrid({ metrics, performanceData, timeRange }: MetricsGridProps) {
  const chartData = useMemo(() => {
    return performanceData.map(item => ({
      ...item,
      time: new Date(item.timestamp).toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        ...(timeRange === '24h' ? {} : { month: 'short', day: 'numeric' })
      })
    }))
  }, [performanceData, timeRange])

  const taskDistribution = [
    { name: 'Completed', value: metrics.completedTasks, color: '#10b981' },
    { name: 'In Progress', value: metrics.totalTasks - metrics.completedTasks, color: '#f59e0b' }
  ]

  const employeeUtilization = [
    { name: 'Active', value: metrics.activeEmployees, color: '#3b82f6' },
    { name: 'Idle', value: metrics.totalEmployees - metrics.activeEmployees, color: '#6b7280' }
  ]

  const performanceComparison = [
    { metric: 'Speed', ai: metrics.averageSpeed, human: 1, target: 5 },
    { metric: 'Accuracy', ai: metrics.averageAccuracy, human: 85, target: 95 },
    { metric: 'Availability', ai: metrics.uptime, human: 40, target: 99 },
    { metric: 'Error Rate', ai: metrics.errorRate, human: 5.2, target: 1 }
  ]

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-background border rounded-lg shadow-lg p-3">
          <p className="font-medium">{label}</p>
          <div className="space-y-1 mt-2">
            {payload.map((entry: any, index: number) => (
              <div key={index} className="flex items-center justify-between space-x-4">
                <span className="text-sm" style={{ color: entry.color }}>
                  {entry.name}:
                </span>
                <span className="font-medium">
                  {typeof entry.value === 'number' ? entry.value.toFixed(1) : entry.value}
                  {entry.name === 'Speed' && 'x'}
                  {(entry.name === 'Accuracy' || entry.name === 'Error Rate') && '%'}
                  {entry.name === 'Response Time' && 'ms'}
                  {entry.name === 'Throughput' && '/hr'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )
    }
    return null
  }

  return (
    <div className="space-y-6">
      {/* Performance Trends */}
      <div className="grid lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5 text-green-500" />
              <span>Performance Trends</span>
            </CardTitle>
            <CardDescription>
              AI employee performance over time
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                  <XAxis 
                    dataKey="time" 
                    tick={{ fontSize: 12 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis 
                    tick={{ fontSize: 12 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Line
                    type="monotone"
                    dataKey="accuracy"
                    stroke="#10b981"
                    strokeWidth={2}
                    dot={{ fill: '#10b981', strokeWidth: 2, r: 4 }}
                    name="Accuracy"
                  />
                  <Line
                    type="monotone"
                    dataKey="speed"
                    stroke="#f59e0b"
                    strokeWidth={2}
                    dot={{ fill: '#f59e0b', strokeWidth: 2, r: 4 }}
                    name="Speed"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="h-5 w-5 text-blue-500" />
              <span>System Performance</span>
            </CardTitle>
            <CardDescription>
              Throughput and response time metrics
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="throughputGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                  <XAxis 
                    dataKey="time" 
                    tick={{ fontSize: 12 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis 
                    tick={{ fontSize: 12 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Area
                    type="monotone"
                    dataKey="throughput"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    fill="url(#throughputGradient)"
                    name="Throughput"
                  />
                  <Line
                    type="monotone"
                    dataKey="responseTime"
                    stroke="#8b5cf6"
                    strokeWidth={2}
                    dot={{ fill: '#8b5cf6', strokeWidth: 2, r: 4 }}
                    name="Response Time"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Performance Comparison */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Target className="h-5 w-5 text-purple-500" />
            <span>AI vs Human Performance</span>
          </CardTitle>
          <CardDescription>
            Comparative analysis showing superhuman capabilities
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={performanceComparison} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                <XAxis 
                  dataKey="metric" 
                  tick={{ fontSize: 12 }}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis 
                  tick={{ fontSize: 12 }}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip 
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div className="bg-background border rounded-lg shadow-lg p-3">
                          <p className="font-medium">{label}</p>
                          <div className="space-y-1 mt-2">
                            {payload.map((entry: any, index: number) => (
                              <div key={index} className="flex items-center justify-between space-x-4">
                                <span className="text-sm" style={{ color: entry.color }}>
                                  {entry.name}:
                                </span>
                                <span className="font-medium">
                                  {entry.value.toFixed(1)}
                                  {label === 'Speed' && 'x'}
                                  {(label === 'Accuracy' || label === 'Availability' || label === 'Error Rate') && '%'}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )
                    }
                    return null
                  }}
                />
                <Bar dataKey="ai" fill="#10b981" name="AI Employee" radius={[4, 4, 0, 0]} />
                <Bar dataKey="human" fill="#6b7280" name="Human Baseline" radius={[4, 4, 0, 0]} />
                <Bar dataKey="target" fill="#f59e0b" name="Target" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Distribution Charts */}
      <div className="grid lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <CheckCircle className="h-5 w-5 text-green-500" />
              <span>Task Distribution</span>
            </CardTitle>
            <CardDescription>
              Current task completion status
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-8">
              <div className="h-48 w-48">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={taskDistribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {taskDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      formatter={(value: number) => [value.toLocaleString(), 'Tasks']}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              
              <div className="space-y-4">
                {taskDistribution.map((item, index) => (
                  <div key={index} className="flex items-center space-x-3">
                    <div 
                      className="w-4 h-4 rounded-full" 
                      style={{ backgroundColor: item.color }}
                    />
                    <div>
                      <div className="font-medium">{item.name}</div>
                      <div className="text-sm text-muted-foreground">
                        {item.value.toLocaleString()} tasks
                      </div>
                    </div>
                  </div>
                ))}
                
                <div className="pt-4 border-t">
                  <div className="text-sm text-muted-foreground">Completion Rate</div>
                  <div className="text-2xl font-bold text-green-600">
                    {((metrics.completedTasks / metrics.totalTasks) * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Users className="h-5 w-5 text-blue-500" />
              <span>Employee Utilization</span>
            </CardTitle>
            <CardDescription>
              AI employee activity status
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-8">
              <div className="h-48 w-48">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={employeeUtilization}
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {employeeUtilization.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      formatter={(value: number) => [value, 'Employees']}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              
              <div className="space-y-4">
                {employeeUtilization.map((item, index) => (
                  <div key={index} className="flex items-center space-x-3">
                    <div 
                      className="w-4 h-4 rounded-full" 
                      style={{ backgroundColor: item.color }}
                    />
                    <div>
                      <div className="font-medium">{item.name}</div>
                      <div className="text-sm text-muted-foreground">
                        {item.value} employees
                      </div>
                    </div>
                  </div>
                ))}
                
                <div className="pt-4 border-t">
                  <div className="text-sm text-muted-foreground">Utilization Rate</div>
                  <div className="text-2xl font-bold text-blue-600">
                    {((metrics.activeEmployees / metrics.totalEmployees) * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Key Performance Indicators */}
      <div className="grid md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Avg Response Time</p>
                <p className="text-2xl font-bold">0.8s</p>
              </div>
              <Clock className="h-8 w-8 text-blue-500" />
            </div>
            <div className="mt-4">
              <Badge className="bg-green-100 text-green-800">
                85% faster than human
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Cost per Task</p>
                <p className="text-2xl font-bold">$0.05</p>
              </div>
              <DollarSign className="h-8 w-8 text-green-500" />
            </div>
            <div className="mt-4">
              <Badge className="bg-green-100 text-green-800">
                95% cost reduction
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Quality Score</p>
                <p className="text-2xl font-bold">{metrics.averageAccuracy.toFixed(1)}%</p>
              </div>
              <Target className="h-8 w-8 text-purple-500" />
            </div>
            <div className="mt-4">
              <Badge className="bg-purple-100 text-purple-800">
                Exceeds target
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Productivity Gain</p>
                <p className="text-2xl font-bold">{metrics.averageSpeed.toFixed(1)}x</p>
              </div>
              <Zap className="h-8 w-8 text-yellow-500" />
            </div>
            <div className="mt-4">
              <Badge className="bg-yellow-100 text-yellow-800">
                Superhuman level
              </Badge>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}