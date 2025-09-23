'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { 
  Activity, 
  CheckCircle, 
  AlertTriangle, 
  Clock, 
  Bot, 
  Zap,
  Users,
  TrendingUp,
  Play,
  Pause
} from 'lucide-react'

const activityTypes = {
  task_completed: { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-50' },
  task_started: { icon: Play, color: 'text-blue-500', bg: 'bg-blue-50' },
  error_occurred: { icon: AlertTriangle, color: 'text-red-500', bg: 'bg-red-50' },
  performance_milestone: { icon: TrendingUp, color: 'text-purple-500', bg: 'bg-purple-50' },
  employee_deployed: { icon: Bot, color: 'text-green-500', bg: 'bg-green-50' },
  employee_paused: { icon: Pause, color: 'text-yellow-500', bg: 'bg-yellow-50' },
  high_performance: { icon: Zap, color: 'text-orange-500', bg: 'bg-orange-50' }
}

const initialActivities = [
  {
    id: 1,
    type: 'task_completed',
    title: 'Customer inquiry resolved',
    description: 'Sarah - Customer Support completed ticket #CS-4521 in 1.2 seconds',
    timestamp: new Date(Date.now() - 30000),
    employee: 'Sarah - Customer Support',
    metadata: { taskId: 'CS-4521', duration: 1.2, accuracy: 98.5 }
  },
  {
    id: 2,
    type: 'high_performance',
    title: 'Superhuman performance detected',
    description: 'Alex - Data Analyst achieved 15.2x speed improvement on quarterly report',
    timestamp: new Date(Date.now() - 120000),
    employee: 'Alex - Data Analyst',
    metadata: { speedMultiplier: 15.2, task: 'quarterly_report' }
  },
  {
    id: 3,
    type: 'task_started',
    title: 'New content creation task',
    description: 'Maya - Content Writer started blog post about AI trends',
    timestamp: new Date(Date.now() - 180000),
    employee: 'Maya - Content Writer',
    metadata: { taskType: 'blog_post', estimatedDuration: 45 }
  },
  {
    id: 4,
    type: 'performance_milestone',
    title: '1000 tasks milestone reached',
    description: 'Riley - Research Assistant completed 1000th task this month',
    timestamp: new Date(Date.now() - 300000),
    employee: 'Riley - Research Assistant',
    metadata: { milestone: 1000, period: 'month' }
  },
  {
    id: 5,
    type: 'task_completed',
    title: 'Sales lead qualified',
    description: 'Jordan - Sales Assistant qualified lead #SL-8934 with 94% confidence',
    timestamp: new Date(Date.now() - 420000),
    employee: 'Jordan - Sales Assistant',
    metadata: { leadId: 'SL-8934', confidence: 94 }
  }
]

export function RealTimeActivityFeed() {
  const [activities, setActivities] = useState(initialActivities)
  const [isLive, setIsLive] = useState(true)

  // Simulate real-time activity updates
  useEffect(() => {
    if (!isLive) return

    const interval = setInterval(() => {
      const newActivity = generateRandomActivity()
      setActivities(prev => [newActivity, ...prev.slice(0, 19)]) // Keep last 20 activities
    }, 5000 + Math.random() * 10000) // Random interval between 5-15 seconds

    return () => clearInterval(interval)
  }, [isLive])

  const generateRandomActivity = () => {
    const employees = [
      'Sarah - Customer Support',
      'Alex - Data Analyst', 
      'Maya - Content Writer',
      'Jordan - Sales Assistant',
      'Riley - Research Assistant',
      'Casey - Project Manager'
    ]

    const activityTemplates = [
      {
        type: 'task_completed',
        title: 'Task completed successfully',
        description: (emp: string) => `${emp} completed task #${Math.random().toString(36).substr(2, 6).toUpperCase()} in ${(Math.random() * 5 + 0.5).toFixed(1)} seconds`,
        metadata: () => ({ 
          taskId: Math.random().toString(36).substr(2, 6).toUpperCase(),
          duration: Math.random() * 5 + 0.5,
          accuracy: 95 + Math.random() * 5
        })
      },
      {
        type: 'high_performance',
        title: 'Exceptional performance detected',
        description: (emp: string) => `${emp} achieved ${(Math.random() * 10 + 8).toFixed(1)}x speed improvement`,
        metadata: () => ({ speedMultiplier: Math.random() * 10 + 8 })
      },
      {
        type: 'task_started',
        title: 'New task initiated',
        description: (emp: string) => `${emp} started processing new request`,
        metadata: () => ({ estimatedDuration: Math.random() * 60 + 10 })
      }
    ]

    const employee = employees[Math.floor(Math.random() * employees.length)]
    const template = activityTemplates[Math.floor(Math.random() * activityTemplates.length)]

    return {
      id: Date.now() + Math.random(),
      type: template.type,
      title: template.title,
      description: template.description(employee),
      timestamp: new Date(),
      employee,
      metadata: template.metadata()
    }
  }

  const formatTimeAgo = (timestamp: Date) => {
    const now = new Date()
    const diffInSeconds = Math.floor((now.getTime() - timestamp.getTime()) / 1000)
    
    if (diffInSeconds < 60) return `${diffInSeconds}s ago`
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`
    return `${Math.floor(diffInSeconds / 86400)}d ago`
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Activity className="h-5 w-5" />
            <span>Live Activity Feed</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
            <Badge variant={isLive ? 'default' : 'secondary'}>
              {isLive ? 'Live' : 'Paused'}
            </Badge>
          </div>
        </CardTitle>
        <CardDescription>
          Real-time updates from your AI employee workforce
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-96">
          <div className="space-y-4">
            {activities.map((activity) => {
              const activityType = activityTypes[activity.type as keyof typeof activityTypes]
              const Icon = activityType.icon
              
              return (
                <div key={activity.id} className="flex items-start space-x-3 p-3 rounded-lg hover:bg-muted/50 transition-colors">
                  <div className={`p-2 rounded-full ${activityType.bg}`}>
                    <Icon className={`h-4 w-4 ${activityType.color}`} />
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <h4 className="font-medium text-sm">{activity.title}</h4>
                      <span className="text-xs text-muted-foreground">
                        {formatTimeAgo(activity.timestamp)}
                      </span>
                    </div>
                    
                    <p className="text-sm text-muted-foreground mt-1">
                      {activity.description}
                    </p>
                    
                    <div className="flex items-center space-x-4 mt-2">
                      <Badge variant="outline" className="text-xs">
                        {activity.employee}
                      </Badge>
                      
                      {activity.metadata.accuracy && (
                        <span className="text-xs text-green-600">
                          {activity.metadata.accuracy.toFixed(1)}% accuracy
                        </span>
                      )}
                      
                      {activity.metadata.speedMultiplier && (
                        <span className="text-xs text-blue-600">
                          {activity.metadata.speedMultiplier.toFixed(1)}x speed
                        </span>
                      )}
                      
                      {activity.metadata.duration && (
                        <span className="text-xs text-purple-600">
                          {activity.metadata.duration.toFixed(1)}s
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </ScrollArea>
        
        <div className="flex items-center justify-between pt-4 border-t">
          <div className="text-sm text-muted-foreground">
            Showing last {activities.length} activities
          </div>
          
          <button
            onClick={() => setIsLive(!isLive)}
            className="text-sm text-blue-600 hover:text-blue-800"
          >
            {isLive ? 'Pause feed' : 'Resume feed'}
          </button>
        </div>
      </CardContent>
    </Card>
  )
}