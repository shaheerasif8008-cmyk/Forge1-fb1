'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { DragDropContext, Droppable, Draggable, DropResult } from 'react-beautiful-dnd'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { 
  Play, 
  Pause, 
  Square, 
  Plus, 
  Trash2, 
  Copy, 
  Settings, 
  Zap, 
  GitBranch,
  RotateCcw,
  Clock,
  Database,
  Globe,
  Mail,
  FileText,
  MessageSquare,
  Bot,
  Workflow,
  ArrowRight,
  ArrowDown,
  CheckCircle,
  AlertCircle,
  Save,
  Download,
  Upload,
  Eye,
  Code
} from 'lucide-react'
import { EmployeeConfig, Workflow, WorkflowStep } from '@/types/employee'
import { useToast } from '@/hooks/use-toast'

interface WorkflowDesignerProps {
  employeeConfig: EmployeeConfig
  onConfigUpdate: (updates: Partial<EmployeeConfig>) => void
}

const stepTypes = {
  trigger: {
    name: 'Triggers',
    icon: Zap,
    color: 'text-yellow-500',
    bgColor: 'bg-yellow-50',
    borderColor: 'border-yellow-200',
    steps: [
      {
        id: 'webhook-trigger',
        name: 'Webhook Trigger',
        description: 'Triggered by incoming HTTP requests',
        icon: Globe,
        config: {
          method: 'POST',
          path: '/webhook',
          authentication: 'none'
        }
      },
      {
        id: 'schedule-trigger',
        name: 'Schedule Trigger',
        description: 'Triggered on a time schedule',
        icon: Clock,
        config: {
          schedule: '0 9 * * *',
          timezone: 'UTC'
        }
      },
      {
        id: 'email-trigger',
        name: 'Email Trigger',
        description: 'Triggered by incoming emails',
        icon: Mail,
        config: {
          folder: 'inbox',
          filter: ''
        }
      },
      {
        id: 'database-trigger',
        name: 'Database Trigger',
        description: 'Triggered by database changes',
        icon: Database,
        config: {
          table: '',
          operation: 'insert'
        }
      }
    ]
  },
  action: {
    name: 'Actions',
    icon: Bot,
    color: 'text-blue-500',
    bgColor: 'bg-blue-50',
    borderColor: 'border-blue-200',
    steps: [
      {
        id: 'ai-response',
        name: 'AI Response',
        description: 'Generate AI-powered response',
        icon: Bot,
        config: {
          prompt: '',
          model: 'gpt-4',
          temperature: 0.7,
          max_tokens: 1000
        }
      },
      {
        id: 'send-email',
        name: 'Send Email',
        description: 'Send email notification',
        icon: Mail,
        config: {
          to: '',
          subject: '',
          template: ''
        }
      },
      {
        id: 'http-request',
        name: 'HTTP Request',
        description: 'Make HTTP API call',
        icon: Globe,
        config: {
          url: '',
          method: 'POST',
          headers: {},
          body: ''
        }
      },
      {
        id: 'database-operation',
        name: 'Database Operation',
        description: 'Perform database operation',
        icon: Database,
        config: {
          operation: 'insert',
          table: '',
          data: {}
        }
      },
      {
        id: 'file-operation',
        name: 'File Operation',
        description: 'Read, write, or process files',
        icon: FileText,
        config: {
          operation: 'read',
          path: '',
          format: 'text'
        }
      }
    ]
  },
  condition: {
    name: 'Conditions',
    icon: GitBranch,
    color: 'text-purple-500',
    bgColor: 'bg-purple-50',
    borderColor: 'border-purple-200',
    steps: [
      {
        id: 'if-condition',
        name: 'If Condition',
        description: 'Conditional branching logic',
        icon: GitBranch,
        config: {
          condition: '',
          operator: 'equals',
          value: ''
        }
      },
      {
        id: 'switch-condition',
        name: 'Switch Condition',
        description: 'Multiple condition branches',
        icon: GitBranch,
        config: {
          variable: '',
          cases: []
        }
      },
      {
        id: 'filter',
        name: 'Filter',
        description: 'Filter data based on criteria',
        icon: GitBranch,
        config: {
          criteria: '',
          operator: 'contains'
        }
      }
    ]
  },
  loop: {
    name: 'Loops',
    icon: RotateCcw,
    color: 'text-green-500',
    bgColor: 'bg-green-50',
    borderColor: 'border-green-200',
    steps: [
      {
        id: 'for-each',
        name: 'For Each',
        description: 'Iterate over a collection',
        icon: RotateCcw,
        config: {
          collection: '',
          variable: 'item'
        }
      },
      {
        id: 'while-loop',
        name: 'While Loop',
        description: 'Loop while condition is true',
        icon: RotateCcw,
        config: {
          condition: '',
          maxIterations: 100
        }
      },
      {
        id: 'retry',
        name: 'Retry',
        description: 'Retry operation on failure',
        icon: RotateCcw,
        config: {
          maxRetries: 3,
          delay: 1000,
          backoff: 'exponential'
        }
      }
    ]
  }
}

export function WorkflowDesigner({ employeeConfig, onConfigUpdate }: WorkflowDesignerProps) {
  const { toast } = useToast()
  const [workflows, setWorkflows] = useState<Workflow[]>(employeeConfig.workflows || [])
  const [activeWorkflow, setActiveWorkflow] = useState<Workflow | null>(workflows[0] || null)
  const [selectedStep, setSelectedStep] = useState<WorkflowStep | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [executionResults, setExecutionResults] = useState<Record<string, any>>({})
  const canvasRef = useRef<HTMLDivElement>(null)

  // Update parent config when workflows change
  useEffect(() => {
    onConfigUpdate({ workflows })
  }, [workflows, onConfigUpdate])

  const createNewWorkflow = useCallback(() => {
    const newWorkflow: Workflow = {
      id: `workflow-${Date.now()}`,
      name: 'New Workflow',
      description: 'Describe what this workflow does',
      steps: [],
      triggers: [],
      enabled: true
    }
    
    setWorkflows(prev => [...prev, newWorkflow])
    setActiveWorkflow(newWorkflow)
  }, [])

  const updateWorkflow = useCallback((updates: Partial<Workflow>) => {
    if (!activeWorkflow) return
    
    const updatedWorkflow = { ...activeWorkflow, ...updates }
    setActiveWorkflow(updatedWorkflow)
    
    setWorkflows(prev => 
      prev.map(w => w.id === activeWorkflow.id ? updatedWorkflow : w)
    )
  }, [activeWorkflow])

  const addStep = useCallback((stepTemplate: any, position: { x: number; y: number }) => {
    if (!activeWorkflow) return

    const newStep: WorkflowStep = {
      id: `step-${Date.now()}`,
      name: stepTemplate.name,
      type: stepTemplate.type || 'action',
      config: { ...stepTemplate.config },
      position,
      connections: []
    }

    updateWorkflow({
      steps: [...activeWorkflow.steps, newStep]
    })
  }, [activeWorkflow, updateWorkflow])

  const updateStep = useCallback((stepId: string, updates: Partial<WorkflowStep>) => {
    if (!activeWorkflow) return

    const updatedSteps = activeWorkflow.steps.map(step =>
      step.id === stepId ? { ...step, ...updates } : step
    )

    updateWorkflow({ steps: updatedSteps })
  }, [activeWorkflow, updateWorkflow])

  const deleteStep = useCallback((stepId: string) => {
    if (!activeWorkflow) return

    const updatedSteps = activeWorkflow.steps.filter(step => step.id !== stepId)
    updateWorkflow({ steps: updatedSteps })
    
    if (selectedStep?.id === stepId) {
      setSelectedStep(null)
    }
  }, [activeWorkflow, updateWorkflow, selectedStep])

  const connectSteps = useCallback((fromId: string, toId: string) => {
    if (!activeWorkflow) return

    const updatedSteps = activeWorkflow.steps.map(step => {
      if (step.id === fromId) {
        return {
          ...step,
          connections: [...step.connections.filter(id => id !== toId), toId]
        }
      }
      return step
    })

    updateWorkflow({ steps: updatedSteps })
  }, [activeWorkflow, updateWorkflow])

  const executeWorkflow = useCallback(async () => {
    if (!activeWorkflow) return

    setIsRunning(true)
    setExecutionResults({})

    try {
      // Simulate workflow execution
      const results: Record<string, any> = {}
      
      for (const step of activeWorkflow.steps) {
        await new Promise(resolve => setTimeout(resolve, 500)) // Simulate processing time
        
        results[step.id] = {
          status: Math.random() > 0.1 ? 'success' : 'error',
          output: `Mock output for ${step.name}`,
          executionTime: Math.random() * 1000,
          timestamp: new Date().toISOString()
        }
        
        setExecutionResults({ ...results })
      }

      toast({
        title: "Workflow Executed",
        description: `${activeWorkflow.name} completed successfully`,
      })

    } catch (error) {
      toast({
        title: "Execution Failed",
        description: "Workflow execution encountered an error",
        variant: "destructive"
      })
    } finally {
      setIsRunning(false)
    }
  }, [activeWorkflow, toast])

  const exportWorkflow = useCallback(() => {
    if (!activeWorkflow) return

    const dataStr = JSON.stringify(activeWorkflow, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    
    const link = document.createElement('a')
    link.href = url
    link.download = `${activeWorkflow.name.replace(/\s+/g, '-').toLowerCase()}.json`
    link.click()
    
    URL.revokeObjectURL(url)
  }, [activeWorkflow])

  const getStepIcon = (step: WorkflowStep) => {
    const stepType = Object.values(stepTypes)
      .flatMap(type => type.steps)
      .find(s => s.id === step.type || s.name === step.name)
    
    return stepType?.icon || Bot
  }

  const getStepColor = (step: WorkflowStep) => {
    if (step.type === 'trigger') return 'border-yellow-300 bg-yellow-50'
    if (step.type === 'condition') return 'border-purple-300 bg-purple-50'
    if (step.type === 'loop') return 'border-green-300 bg-green-50'
    return 'border-blue-300 bg-blue-50'
  }

  return (
    <div className="space-y-6">
      {/* Workflow Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Workflow className="h-5 w-5" />
                <CardTitle>Workflow Designer</CardTitle>
              </div>
              
              {workflows.length > 0 && (
                <Select
                  value={activeWorkflow?.id || ''}
                  onValueChange={(value) => {
                    const workflow = workflows.find(w => w.id === value)
                    setActiveWorkflow(workflow || null)
                  }}
                >
                  <SelectTrigger className="w-64">
                    <SelectValue placeholder="Select workflow" />
                  </SelectTrigger>
                  <SelectContent>
                    {workflows.map((workflow) => (
                      <SelectItem key={workflow.id} value={workflow.id}>
                        {workflow.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </div>

            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm" onClick={createNewWorkflow}>
                <Plus className="h-4 w-4 mr-2" />
                New Workflow
              </Button>
              
              {activeWorkflow && (
                <>
                  <Button variant="outline" size="sm" onClick={exportWorkflow}>
                    <Download className="h-4 w-4 mr-2" />
                    Export
                  </Button>
                  
                  <Button
                    onClick={executeWorkflow}
                    disabled={isRunning || activeWorkflow.steps.length === 0}
                    className="btn-glow"
                    size="sm"
                  >
                    {isRunning ? (
                      <>
                        <Pause className="h-4 w-4 mr-2" />
                        Running...
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4 mr-2" />
                        Test Run
                      </>
                    )}
                  </Button>
                </>
              )}
            </div>
          </div>
          
          {activeWorkflow && (
            <CardDescription>
              Design and configure automated workflows for your AI employee
            </CardDescription>
          )}
        </CardHeader>
        
        {activeWorkflow && (
          <CardContent>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="workflow-name">Workflow Name</Label>
                <Input
                  id="workflow-name"
                  value={activeWorkflow.name}
                  onChange={(e) => updateWorkflow({ name: e.target.value })}
                  placeholder="Enter workflow name"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="workflow-description">Description</Label>
                <Input
                  id="workflow-description"
                  value={activeWorkflow.description}
                  onChange={(e) => updateWorkflow({ description: e.target.value })}
                  placeholder="Describe what this workflow does"
                />
              </div>
            </div>
            
            <div className="flex items-center space-x-2 mt-4">
              <Switch
                id="workflow-enabled"
                checked={activeWorkflow.enabled}
                onCheckedChange={(enabled) => updateWorkflow({ enabled })}
              />
              <Label htmlFor="workflow-enabled">Enable this workflow</Label>
            </div>
          </CardContent>
        )}
      </Card>

      {activeWorkflow ? (
        <div className="grid lg:grid-cols-4 gap-6">
          {/* Step Library */}
          <Card className="lg:col-span-1">
            <CardHeader>
              <CardTitle className="text-sm">Step Library</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="trigger" className="space-y-4">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="trigger" className="text-xs">Triggers</TabsTrigger>
                  <TabsTrigger value="action" className="text-xs">Actions</TabsTrigger>
                </TabsList>
                
                {Object.entries(stepTypes).map(([key, type]) => (
                  <TabsContent key={key} value={key} className="space-y-2">
                    <ScrollArea className="h-64">
                      {type.steps.map((step) => (
                        <motion.div
                          key={step.id}
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          className={`p-3 border rounded-lg cursor-pointer mb-2 ${type.bgColor} ${type.borderColor} hover:shadow-sm transition-all`}
                          onClick={() => addStep({ ...step, type: key }, { x: 100, y: 100 })}
                        >
                          <div className="flex items-center space-x-2">
                            <step.icon className={`h-4 w-4 ${type.color}`} />
                            <div>
                              <div className="font-medium text-sm">{step.name}</div>
                              <div className="text-xs text-muted-foreground">
                                {step.description}
                              </div>
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </ScrollArea>
                  </TabsContent>
                ))}
              </Tabs>
            </CardContent>
          </Card>

          {/* Workflow Canvas */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="text-sm flex items-center justify-between">
                <span>Workflow Canvas</span>
                <Badge variant="secondary">
                  {activeWorkflow.steps.length} steps
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div
                ref={canvasRef}
                className="relative h-96 border-2 border-dashed border-muted rounded-lg overflow-auto bg-muted/20"
              >
                {activeWorkflow.steps.length === 0 ? (
                  <div className="flex items-center justify-center h-full text-muted-foreground">
                    <div className="text-center">
                      <Workflow className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>Drag steps from the library to start building your workflow</p>
                    </div>
                  </div>
                ) : (
                  <div className="relative p-4">
                    {activeWorkflow.steps.map((step, index) => {
                      const StepIcon = getStepIcon(step)
                      const executionResult = executionResults[step.id]
                      
                      return (
                        <motion.div
                          key={step.id}
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          className={`absolute w-48 p-3 border-2 rounded-lg cursor-pointer ${getStepColor(step)} ${
                            selectedStep?.id === step.id ? 'ring-2 ring-primary' : ''
                          }`}
                          style={{
                            left: step.position.x,
                            top: step.position.y + index * 80
                          }}
                          onClick={() => setSelectedStep(step)}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                              <StepIcon className="h-4 w-4" />
                              <span className="font-medium text-sm">{step.name}</span>
                            </div>
                            
                            <div className="flex items-center space-x-1">
                              {executionResult && (
                                <div className={`w-2 h-2 rounded-full ${
                                  executionResult.status === 'success' ? 'bg-green-500' : 'bg-red-500'
                                }`} />
                              )}
                              
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  deleteStep(step.id)
                                }}
                                className="h-6 w-6 p-0"
                              >
                                <Trash2 className="h-3 w-3" />
                              </Button>
                            </div>
                          </div>
                          
                          {executionResult && (
                            <div className="mt-2 text-xs">
                              <div className={`font-medium ${
                                executionResult.status === 'success' ? 'text-green-600' : 'text-red-600'
                              }`}>
                                {executionResult.status}
                              </div>
                              <div className="text-muted-foreground">
                                {executionResult.executionTime?.toFixed(0)}ms
                              </div>
                            </div>
                          )}
                          
                          {/* Connection points */}
                          <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 w-4 h-4 bg-white border-2 border-gray-300 rounded-full" />
                          <div className="absolute -top-2 left-1/2 transform -translate-x-1/2 w-4 h-4 bg-white border-2 border-gray-300 rounded-full" />
                        </motion.div>
                      )
                    })}
                    
                    {/* Connection lines */}
                    <svg className="absolute inset-0 pointer-events-none">
                      {activeWorkflow.steps.map((step) =>
                        step.connections.map((connectionId) => {
                          const targetStep = activeWorkflow.steps.find(s => s.id === connectionId)
                          if (!targetStep) return null
                          
                          const startX = step.position.x + 96 // Half width of step
                          const startY = step.position.y + 60 // Bottom of step
                          const endX = targetStep.position.x + 96
                          const endY = targetStep.position.y
                          
                          return (
                            <line
                              key={`${step.id}-${connectionId}`}
                              x1={startX}
                              y1={startY}
                              x2={endX}
                              y2={endY}
                              stroke="#6b7280"
                              strokeWidth="2"
                              markerEnd="url(#arrowhead)"
                            />
                          )
                        })
                      )}
                      
                      <defs>
                        <marker
                          id="arrowhead"
                          markerWidth="10"
                          markerHeight="7"
                          refX="9"
                          refY="3.5"
                          orient="auto"
                        >
                          <polygon
                            points="0 0, 10 3.5, 0 7"
                            fill="#6b7280"
                          />
                        </marker>
                      </defs>
                    </svg>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Step Configuration */}
          <Card className="lg:col-span-1">
            <CardHeader>
              <CardTitle className="text-sm">Step Configuration</CardTitle>
            </CardHeader>
            <CardContent>
              {selectedStep ? (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="step-name">Step Name</Label>
                    <Input
                      id="step-name"
                      value={selectedStep.name}
                      onChange={(e) => updateStep(selectedStep.id, { name: e.target.value })}
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label>Step Type</Label>
                    <Badge variant="outline">{selectedStep.type}</Badge>
                  </div>
                  
                  <Separator />
                  
                  <div className="space-y-3">
                    <Label className="text-sm font-medium">Configuration</Label>
                    
                    {Object.entries(selectedStep.config).map(([key, value]) => (
                      <div key={key} className="space-y-1">
                        <Label htmlFor={`config-${key}`} className="text-xs">
                          {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                        </Label>
                        
                        {typeof value === 'boolean' ? (
                          <Switch
                            id={`config-${key}`}
                            checked={value}
                            onCheckedChange={(checked) => 
                              updateStep(selectedStep.id, {
                                config: { ...selectedStep.config, [key]: checked }
                              })
                            }
                          />
                        ) : typeof value === 'string' && value.length > 50 ? (
                          <Textarea
                            id={`config-${key}`}
                            value={value}
                            onChange={(e) =>
                              updateStep(selectedStep.id, {
                                config: { ...selectedStep.config, [key]: e.target.value }
                              })
                            }
                            rows={3}
                          />
                        ) : (
                          <Input
                            id={`config-${key}`}
                            value={String(value)}
                            onChange={(e) =>
                              updateStep(selectedStep.id, {
                                config: { ...selectedStep.config, [key]: e.target.value }
                              })
                            }
                          />
                        )}
                      </div>
                    ))}
                  </div>
                  
                  {executionResults[selectedStep.id] && (
                    <>
                      <Separator />
                      <div className="space-y-2">
                        <Label className="text-sm font-medium">Execution Result</Label>
                        <div className="p-2 bg-muted rounded text-xs">
                          <pre>{JSON.stringify(executionResults[selectedStep.id], null, 2)}</pre>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              ) : (
                <div className="text-center text-muted-foreground py-8">
                  <Settings className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">Select a step to configure its properties</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      ) : (
        <Card>
          <CardContent className="text-center py-12">
            <Workflow className="h-16 w-16 mx-auto mb-4 text-muted-foreground opacity-50" />
            <h3 className="text-lg font-semibold mb-2">No Workflows Yet</h3>
            <p className="text-muted-foreground mb-4">
              Create your first workflow to automate your AI employee's tasks
            </p>
            <Button onClick={createNewWorkflow}>
              <Plus className="h-4 w-4 mr-2" />
              Create First Workflow
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  )
}