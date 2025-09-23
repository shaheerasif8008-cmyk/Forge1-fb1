'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
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
  Bar
} from 'recharts'
import { TrendingUp, BarChart3, Activity } from 'lucide-react'

interface PerformanceTrendsChartProps {
  timeRange: string
}

const performanceData = [
  { time: '00:00', accuracy: 96.2, speed: 11.8, tasks: 45, errors: 2 },
  { time: '02:00', accuracy: 97.1, speed: 12.3, tasks: 38, errors: 1 },
  { time: '04:00', accuracy: 98.2, speed: 13.1, tasks: 42, errors: 1 },
  { time: '06:00', accuracy: 97.8, speed: 12.7, tasks: 58, errors: 2 },
  { time: '08:00', accuracy: 98.5, speed: 13.4, tasks: 125, errors: 1 },
  { time: '10:00', accuracy: 97.9, speed: 12.9, tasks: 187, errors: 3 },
  { time: '12:00', accuracy: 98.1, speed: 13.2, tasks: 234, errors: 2 },
  { time: '14:00', accuracy: 98.7, speed: 14.1, tasks: 298, errors: 1 },
  { time: '16:00', accuracy: 97.6, speed: 12.8, tasks: 267, errors: 4 },
  { time: '18:00', accuracy: 98.3, speed: 13.5, tasks: 198, errors: 2 },
  { time: '20:00', accuracy: 97.4, speed: 12.6, tasks: 145, errors: 3 },
  { time: '22:00', accuracy: 98.0, speed: 13.0, tasks: 89, errors: 1 }
]

export function PerformanceTrendsChart({ timeRange }: PerformanceTrendsChartProps) {
  const formatTooltip = (value: any, name: string) => {
    if (name === 'accuracy') return [`${value}%`, 'Accuracy']
    if (name === 'speed') return [`${value}x`, 'Speed Multiplier']
    if (name === 'tasks') return [value, 'Tasks Completed']
    if (name === 'errors') return [value, 'Errors']
    return [value, name]
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <BarChart3 className="h-5 w-5" />
          <span>Performance Trends</span>
        </CardTitle>
        <CardDescription>
          Real-time performance metrics over the last {timeRange}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="accuracy" className="space-y-4">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="accuracy">Accuracy</TabsTrigger>
            <TabsTrigger value="speed">Speed</TabsTrigger>
            <TabsTrigger value="throughput">Throughput</TabsTrigger>
            <TabsTrigger value="errors">Errors</TabsTrigger>
          </TabsList>

          <TabsContent value="accuracy" className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <Badge variant="default" className="bg-green-600">
                  98.1% Current
                </Badge>
                <div className="flex items-center space-x-1 text-green-600">
                  <TrendingUp className="h-4 w-4" />
                  <span className="text-sm">+2.3% vs yesterday</span>
                </div>
              </div>
            </div>
            
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis domain={[95, 100]} />
                  <Tooltip formatter={formatTooltip} />
                  <Area 
                    type="monotone" 
                    dataKey="accuracy" 
                    stroke="#10b981" 
                    fill="#10b981" 
                    fillOpacity={0.3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>

          <TabsContent value="speed" className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <Badge variant="default" className="bg-blue-600">
                  13.2x Current
                </Badge>
                <div className="flex items-center space-x-1 text-blue-600">
                  <TrendingUp className="h-4 w-4" />
                  <span className="text-sm">+8.7% vs yesterday</span>
                </div>
              </div>
            </div>
            
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip formatter={formatTooltip} />
                  <Line 
                    type="monotone" 
                    dataKey="speed" 
                    stroke="#3b82f6" 
                    strokeWidth={3}
                    dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>

          <TabsContent value="throughput" className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <Badge variant="default" className="bg-purple-600">
                  1,847 Tasks/Hour
                </Badge>
                <div className="flex items-center space-x-1 text-purple-600">
                  <TrendingUp className="h-4 w-4" />
                  <span className="text-sm">+23.4% vs yesterday</span>
                </div>
              </div>
            </div>
            
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip formatter={formatTooltip} />
                  <Bar dataKey="tasks" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>

          <TabsContent value="errors" className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <Badge variant="secondary" className="bg-red-100 text-red-700">
                  2.2% Error Rate
                </Badge>
                <div className="flex items-center space-x-1 text-green-600">
                  <TrendingUp className="h-4 w-4 rotate-180" />
                  <span className="text-sm">-15.3% vs yesterday</span>
                </div>
              </div>
            </div>
            
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip formatter={formatTooltip} />
                  <Area 
                    type="monotone" 
                    dataKey="errors" 
                    stroke="#ef4444" 
                    fill="#ef4444" 
                    fillOpacity={0.3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}