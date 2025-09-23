'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  LineChart,
  Line,
  Area,
  AreaChart,
  PieChart,
  Pie,
  Cell
} from 'recharts'
import { 
  Target, 
  Zap, 
  Clock, 
  TrendingUp, 
  Users, 
  CheckCircle,
  AlertTriangle,
  Activity
} from 'lucide-react'

interface PerformanceMetricsWidgetProps {
  timeRange: string
}

const performanceData = [
  { name: 'Mon', accuracy: 96.2, speed: 11.8, availability: 99.9, tasks: 1240 },
  { name: 'Tue', accuracy: 97.1, speed: 12.3, availability: 99.8, tasks: 1380 },
  { name: 'Wed', accuracy: 98.2, speed: 13.1, availability: 99.9, tasks: 1520 },
  { name: 'Thu', accuracy: 97.8, speed: 12.7, availability: 99.7, tasks: 1450 },
  { name: 'Fri', accuracy: 98.5, speed: 13.4, availability: 99.9, tasks: 1680 },
  { name: 'Sat', accuracy: 97.9, speed: 12.9, availability: 99.8, tasks: 1320 },
  { name: 'Sun', accuracy: 98.1, speed: 13.2, availability: 99.9, tasks: 1180 }
]

const employeeTypeData = [
  { name: 'Customer Service', value: 35, performance: 98.2, color: '#0ea5e9' },
  { name: 'Data Analysis', value: 25, performance: 97.8, color: '#10b981' },
  { name: 'Content Writing', value: 20, performance: 96.5, color: '#f59e0b' },
  { name: 'Sales Support', value: 15, performance: 95.9, color: '#ef4444' },
  { name: 'Research', value: 5, performance: 98.7, color: '#8b5cf6' }
]

const realTimeMetrics = {
  currentTasks: 847,
  queuedTasks: 23,
  completedToday: 12450,
  averageResponseTime: 1.2,
  peakPerformanceHour: '2:00 PM',
  topPerformer: 'Customer Service Agent #7'
}

export function PerformanceMetricsWidget({ timeRange }: PerformanceMetricsWidgetProps) {
  const [activeMetric, setActiveMetric] = useState('accuracy')

  const getMetricColor = (value: number, target: number) => {
    if (value >= target * 1.1) return 'text-green-600'
    if (value >= target) return 'text-blue-600'
    if (value >= target * 0.9) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getMetricIcon = (metric: string) => {
    switch (metric) {
      case 'accuracy': return Target
      case 'speed': return Zap
      case 'availability': return Clock
      case 'tasks': return CheckCircle
      default: return Activity
    }
  }

  return (
    <div className="space-y-6">
      {/* Real-time Performance Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Activity className="h-5 w-5" />
            <span>Real-time Performance</span>
          </CardTitle>
          <CardDescription>
            Live performance metrics across all AI employees
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {realTimeMetrics.currentTasks}
              </div>
              <div className="text-sm text-muted-foreground">Active Tasks</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-600">
                {realTimeMetrics.queuedTasks}
              </div>
              <div className="text-sm text-muted-foreground">Queued Tasks</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {realTimeMetrics.completedToday.toLocaleString()}
              </div>
              <div className="text-sm text-muted-foreground">Completed Today</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {realTimeMetrics.averageResponseTime}s
              </div>
              <div className="text-sm text-muted-foreground">Avg Response</div>
            </div>
            <div className="text-center">
              <div className="text-sm font-medium">Peak Hour</div>
              <div className="text-sm text-muted-foreground">
                {realTimeMetrics.peakPerformanceHour}
              </div>
            </div>
            <div className="text-center">
              <div className="text-sm font-medium">Top Performer</div>
              <div className="text-sm text-muted-foreground">
                {realTimeMetrics.topPerformer}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Performance Trends */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Trends</CardTitle>
          <CardDescription>
            Track key performance indicators over time
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs value={activeMetric} onValueChange={setActiveMetric} className="space-y-4">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="accuracy">Accuracy</TabsTrigger>
              <TabsTrigger value="speed">Speed</TabsTrigger>
              <TabsTrigger value="availability">Availability</TabsTrigger>
              <TabsTrigger value="tasks">Tasks</TabsTrigger>
            </TabsList>

            <TabsContent value="accuracy" className="space-y-4">
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis domain={[95, 100]} />
                    <Tooltip 
                      formatter={(value) => [`${value}%`, 'Accuracy']}
                      labelFormatter={(label) => `Day: ${label}`}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="accuracy" 
                      stroke="#0ea5e9" 
                      fill="#0ea5e9" 
                      fillOpacity={0.3}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>

            <TabsContent value="speed" className="space-y-4">
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip 
                      formatter={(value) => [`${value}x`, 'Speed Multiplier']}
                      labelFormatter={(label) => `Day: ${label}`}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="speed" 
                      stroke="#10b981" 
                      strokeWidth={3}
                      dot={{ fill: '#10b981', strokeWidth: 2, r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>

            <TabsContent value="availability" className="space-y-4">
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis domain={[99.5, 100]} />
                    <Tooltip 
                      formatter={(value) => [`${value}%`, 'Availability']}
                      labelFormatter={(label) => `Day: ${label}`}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="availability" 
                      stroke="#f59e0b" 
                      fill="#f59e0b" 
                      fillOpacity={0.3}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>

            <TabsContent value="tasks" className="space-y-4">
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip 
                      formatter={(value) => [value, 'Tasks Completed']}
                      labelFormatter={(label) => `Day: ${label}`}
                    />
                    <Bar dataKey="tasks" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Employee Type Performance */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Performance by Employee Type</CardTitle>
            <CardDescription>
              Compare performance across different AI employee categories
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={employeeTypeData}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}%`}
                  >
                    {employeeTypeData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Employee Type Rankings</CardTitle>
            <CardDescription>
              Performance rankings by employee category
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {employeeTypeData
                .sort((a, b) => b.performance - a.performance)
                .map((type, index) => (
                  <div key={type.name} className="flex items-center space-x-4">
                    <div className="flex items-center justify-center w-8 h-8 rounded-full bg-muted text-sm font-medium">
                      {index + 1}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{type.name}</span>
                        <span className="text-sm font-medium">
                          {type.performance}%
                        </span>
                      </div>
                      <Progress 
                        value={type.performance} 
                        className="mt-2"
                        style={{ 
                          '--progress-background': type.color 
                        } as React.CSSProperties}
                      />
                    </div>
                  </div>
                ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Performance Targets */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Target className="h-5 w-5" />
            <span>Performance Targets</span>
          </CardTitle>
          <CardDescription>
            Track progress against superhuman performance goals
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { name: 'Accuracy', current: 97.8, target: 95, unit: '%' },
              { name: 'Speed', current: 12.4, target: 5, unit: 'x' },
              { name: 'Availability', current: 99.94, target: 99.9, unit: '%' },
              { name: 'Customer Satisfaction', current: 94.2, target: 90, unit: '%' }
            ].map((metric) => {
              const progress = Math.min((metric.current / metric.target) * 100, 100)
              const isExceeding = metric.current > metric.target
              
              return (
                <div key={metric.name} className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{metric.name}</span>
                    <Badge variant={isExceeding ? "default" : "secondary"}>
                      {isExceeding ? 'Exceeding' : 'On Track'}
                    </Badge>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex items-baseline space-x-2">
                      <span className="text-2xl font-bold">
                        {metric.current}{metric.unit}
                      </span>
                      <span className="text-sm text-muted-foreground">
                        / {metric.target}{metric.unit}
                      </span>
                    </div>
                    
                    <Progress value={progress} className="h-2" />
                    
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">
                        {progress.toFixed(1)}% of target
                      </span>
                      {isExceeding && (
                        <div className="flex items-center text-green-600">
                          <TrendingUp className="h-3 w-3 mr-1" />
                          <span>+{((metric.current / metric.target - 1) * 100).toFixed(1)}%</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}