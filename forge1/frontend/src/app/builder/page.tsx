'use client'

import { useState, useCallback } from 'react'
import { DragDropContext, Droppable, Draggable, DropResult } from 'react-beautiful-dnd'
import { motion, AnimatePresence } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Separator } from '@/components/ui/separator'
import { ScrollArea } from '@/components/ui/scroll-area'
import { 
  Bot, 
  Brain, 
  Zap, 
  Shield, 
  Users, 
  MessageSquare, 
  FileText, 
  BarChart3,
  Settings,
  Play,
  Save,
  Download,
  Upload,
  Plus,
  Trash2,
  Copy,
  Eye,
  Sparkles,
  Target,
  Clock,
  Database,
  Globe,
  Lock,
  Palette
} from 'lucide-react'
import { useToast } from '@/hooks/use-toast'
import { EmployeeTemplate, EmployeeConfig, Skill, Personality } from '@/types/employee'
import { TemplateLibrary } from '@/components/builder/template-library'
import { SkillsEditor } from '@/components/builder/skills-editor'
import { PersonalityEditor } from '@/components/builder/personality-editor'
import { WorkflowDesigner } from '@/components/builder/workflow-designer'
import { PerformanceValidator } from '@/components/builder/performance-validator'

const employeeTypes = [
  {
    id: 'customer-service',
    name: 'Customer Service Agent',
    description: 'Handle customer inquiries, complaints, and support requests',
    icon: MessageSquare,
    color: 'text-blue-500',
    skills: ['communication', 'problem-solving', 'empathy', 'product-knowledge'],
    personality: { friendliness: 90, patience: 95, assertiveness: 60 }
  },
  {
    id: 'data-analyst',
    name: 'Data Analyst',
    description: 'Analyze data, generate insights, and create reports',
    icon: BarChart3,
    color: 'text-green-500',
    skills: ['data-analysis', 'statistics', 'visualization', 'sql'],
    personality: { analytical: 95, attention_to_detail: 90, creativity: 70 }
  },
  {
    id: 'content-writer',
    name: 'Content Writer',
    description: 'Create engaging content, articles, and marketing materials',
    icon: FileText,
    color: 'text-purple-500',
    skills: ['writing', 'creativity', 'research', 'seo'],
    personality: { creativity: 95, communication: 85, adaptability: 80 }
  },
  {
    id: 'sales-assistant',
    name: 'Sales Assistant',
    description: 'Support sales processes, lead qualification, and follow-ups',
    icon: Target,
    color: 'text-orange-500',
    skills: ['sales', 'persuasion', 'crm', 'lead-qualification'],
    personality: { persuasiveness: 90, confidence: 85, persistence: 80 }
  },
  {
    id: 'project-manager',
    name: 'Project Manager',
    description: 'Coordinate projects, manage timelines, and track progress',
    icon: Users,
    color: 'text-red-500',
    skills: ['project-management', 'coordination', 'planning', 'communication'],
    personality: { leadership: 90, organization: 95, communication: 85 }
  },
  {
    id: 'research-assistant',
    name: 'Research Assistant',
    description: 'Conduct research, gather information, and synthesize findings',
    icon: Brain,
    color: 'text-indigo-500',
    skills: ['research', 'analysis', 'synthesis', 'fact-checking'],
    personality: { curiosity: 95, analytical: 90, thoroughness: 85 }
  }
]

export default function EmployeeBuilderPage() {
  const { toast } = useToast()
  const [selectedTemplate, setSelectedTemplate] = useState<EmployeeTemplate | null>(null)
  const [employeeConfig, setEmployeeConfig] = useState<EmployeeConfig>({
    id: '',
    name: '',
    description: '',
    type: '',
    skills: [],
    personality: {},
    capabilities: [],
    integrations: [],
    performance_targets: {
      accuracy: 95,
      speed: 80,
      availability: 99
    },
    security_level: 'standard',
    deployment_config: {
      environment: 'production',
      scaling: 'auto',
      monitoring: true
    }
  })
  const [activeTab, setActiveTab] = useState('template')
  const [isBuilding, setIsBuilding] = useState(false)

  const handleTemplateSelect = useCallback((template: EmployeeTemplate) => {
    setSelectedTemplate(template)
    setEmployeeConfig({
      ...employeeConfig,
      type: template.id,
      name: template.name,
      description: template.description,
      skills: template.skills || [],
      personality: template.personality || {},
      capabilities: template.capabilities || []
    })
    setActiveTab('configuration')
  }, [employeeConfig])

  const handleConfigUpdate = useCallback((updates: Partial<EmployeeConfig>) => {
    setEmployeeConfig(prev => ({ ...prev, ...updates }))
  }, [])

  const handleSaveEmployee = async () => {
    try {
      setIsBuilding(true)
      
      // Validate configuration
      if (!employeeConfig.name || !employeeConfig.type) {
        toast({
          title: "Validation Error",
          description: "Please provide a name and select an employee type.",
          variant: "destructive"
        })
        return
      }

      // Save employee configuration
      const response = await fetch('/api/v1/ai-employees', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(employeeConfig)
      })

      if (response.ok) {
        toast({
          title: "Success",
          description: "AI Employee created successfully!",
        })
        // Reset form or redirect
      } else {
        throw new Error('Failed to create employee')
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to create AI Employee. Please try again.",
        variant: "destructive"
      })
    } finally {
      setIsBuilding(false)
    }
  }

  const handleDeployEmployee = async () => {
    try {
      setIsBuilding(true)
      
      const response = await fetch(`/api/v1/ai-employees/${employeeConfig.id}/deploy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(employeeConfig.deployment_config)
      })

      if (response.ok) {
        toast({
          title: "Deployed",
          description: "AI Employee deployed successfully!",
        })
      } else {
        throw new Error('Deployment failed')
      }
    } catch (error) {
      toast({
        title: "Deployment Error",
        description: "Failed to deploy AI Employee. Please try again.",
        variant: "destructive"
      })
    } finally {
      setIsBuilding(false)
    }
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container-forge flex h-16 items-center justify-between">
          <div className="flex items-center space-x-4">
            <Bot className="h-8 w-8 text-forge-500" />
            <div>
              <h1 className="text-xl font-bold">AI Employee Builder</h1>
              <p className="text-sm text-muted-foreground">Create superhuman AI employees</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <Button variant="outline" size="sm">
              <Upload className="h-4 w-4 mr-2" />
              Import
            </Button>
            <Button variant="outline" size="sm">
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
            <Button 
              onClick={handleSaveEmployee}
              disabled={isBuilding}
              size="sm"
            >
              <Save className="h-4 w-4 mr-2" />
              Save
            </Button>
            <Button 
              onClick={handleDeployEmployee}
              disabled={isBuilding || !employeeConfig.id}
              className="btn-glow"
              size="sm"
            >
              <Play className="h-4 w-4 mr-2" />
              Deploy
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container-forge py-6">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="template" className="flex items-center space-x-2">
              <Palette className="h-4 w-4" />
              <span>Template</span>
            </TabsTrigger>
            <TabsTrigger value="configuration" className="flex items-center space-x-2">
              <Settings className="h-4 w-4" />
              <span>Configuration</span>
            </TabsTrigger>
            <TabsTrigger value="skills" className="flex items-center space-x-2">
              <Brain className="h-4 w-4" />
              <span>Skills</span>
            </TabsTrigger>
            <TabsTrigger value="workflow" className="flex items-center space-x-2">
              <Zap className="h-4 w-4" />
              <span>Workflow</span>
            </TabsTrigger>
            <TabsTrigger value="validation" className="flex items-center space-x-2">
              <Shield className="h-4 w-4" />
              <span>Validation</span>
            </TabsTrigger>
          </TabsList>

          {/* Template Selection */}
          <TabsContent value="template" className="space-y-6">
            <div className="space-y-4">
              <div>
                <h2 className="text-2xl font-bold">Choose Employee Type</h2>
                <p className="text-muted-foreground">
                  Select a pre-configured template or start from scratch
                </p>
              </div>

              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                {employeeTypes.map((type) => (
                  <motion.div
                    key={type.id}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <Card 
                      className={`cursor-pointer transition-all duration-200 hover:shadow-lg ${
                        selectedTemplate?.id === type.id ? 'ring-2 ring-primary' : ''
                      }`}
                      onClick={() => handleTemplateSelect(type as any)}
                    >
                      <CardHeader>
                        <div className="flex items-center space-x-3">
                          <div className={`p-2 rounded-lg bg-muted ${type.color}`}>
                            <type.icon className="h-6 w-6" />
                          </div>
                          <div>
                            <CardTitle className="text-lg">{type.name}</CardTitle>
                            <CardDescription>{type.description}</CardDescription>
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-3">
                          <div>
                            <Label className="text-sm font-medium">Key Skills</Label>
                            <div className="flex flex-wrap gap-1 mt-1">
                              {type.skills.slice(0, 3).map((skill) => (
                                <Badge key={skill} variant="secondary" className="text-xs">
                                  {skill}
                                </Badge>
                              ))}
                              {type.skills.length > 3 && (
                                <Badge variant="outline" className="text-xs">
                                  +{type.skills.length - 3} more
                                </Badge>
                              )}
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                ))}
              </div>

              <Card className="border-dashed">
                <CardContent className="flex flex-col items-center justify-center py-12">
                  <Plus className="h-12 w-12 text-muted-foreground mb-4" />
                  <h3 className="text-lg font-semibold mb-2">Custom Employee</h3>
                  <p className="text-muted-foreground text-center mb-4">
                    Start from scratch and build a completely custom AI employee
                  </p>
                  <Button 
                    variant="outline"
                    onClick={() => {
                      setSelectedTemplate(null)
                      setEmployeeConfig({
                        ...employeeConfig,
                        type: 'custom',
                        name: '',
                        description: ''
                      })
                      setActiveTab('configuration')
                    }}
                  >
                    Start Custom Build
                  </Button>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Configuration */}
          <TabsContent value="configuration" className="space-y-6">
            <div className="grid lg:grid-cols-3 gap-6">
              {/* Basic Information */}
              <div className="lg:col-span-2 space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Bot className="h-5 w-5" />
                      <span>Basic Information</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="name">Employee Name</Label>
                        <Input
                          id="name"
                          value={employeeConfig.name}
                          onChange={(e) => handleConfigUpdate({ name: e.target.value })}
                          placeholder="e.g., Sarah - Customer Support"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="type">Employee Type</Label>
                        <Select
                          value={employeeConfig.type}
                          onValueChange={(value) => handleConfigUpdate({ type: value })}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Select type" />
                          </SelectTrigger>
                          <SelectContent>
                            {employeeTypes.map((type) => (
                              <SelectItem key={type.id} value={type.id}>
                                {type.name}
                              </SelectItem>
                            ))}
                            <SelectItem value="custom">Custom</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="description">Description</Label>
                      <Textarea
                        id="description"
                        value={employeeConfig.description}
                        onChange={(e) => handleConfigUpdate({ description: e.target.value })}
                        placeholder="Describe what this AI employee will do..."
                        rows={3}
                      />
                    </div>
                  </CardContent>
                </Card>

                {/* Performance Targets */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Target className="h-5 w-5" />
                      <span>Performance Targets</span>
                    </CardTitle>
                    <CardDescription>
                      Set superhuman performance expectations
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <Label>Accuracy Target</Label>
                          <span className="text-sm text-muted-foreground">
                            {employeeConfig.performance_targets.accuracy}%
                          </span>
                        </div>
                        <Slider
                          value={[employeeConfig.performance_targets.accuracy]}
                          onValueChange={([value]) => 
                            handleConfigUpdate({
                              performance_targets: {
                                ...employeeConfig.performance_targets,
                                accuracy: value
                              }
                            })
                          }
                          max={100}
                          min={50}
                          step={1}
                          className="w-full"
                        />
                      </div>

                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <Label>Speed Multiplier</Label>
                          <span className="text-sm text-muted-foreground">
                            {employeeConfig.performance_targets.speed}x
                          </span>
                        </div>
                        <Slider
                          value={[employeeConfig.performance_targets.speed]}
                          onValueChange={([value]) => 
                            handleConfigUpdate({
                              performance_targets: {
                                ...employeeConfig.performance_targets,
                                speed: value
                              }
                            })
                          }
                          max={50}
                          min={1}
                          step={1}
                          className="w-full"
                        />
                      </div>

                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <Label>Availability</Label>
                          <span className="text-sm text-muted-foreground">
                            {employeeConfig.performance_targets.availability}%
                          </span>
                        </div>
                        <Slider
                          value={[employeeConfig.performance_targets.availability]}
                          onValueChange={([value]) => 
                            handleConfigUpdate({
                              performance_targets: {
                                ...employeeConfig.performance_targets,
                                availability: value
                              }
                            })
                          }
                          max={100}
                          min={90}
                          step={0.1}
                          className="w-full"
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Security & Compliance */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Lock className="h-5 w-5" />
                      <span>Security & Compliance</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label>Security Level</Label>
                      <Select
                        value={employeeConfig.security_level}
                        onValueChange={(value) => handleConfigUpdate({ security_level: value })}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="basic">Basic</SelectItem>
                          <SelectItem value="standard">Standard</SelectItem>
                          <SelectItem value="high">High Security</SelectItem>
                          <SelectItem value="enterprise">Enterprise</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div className="flex items-center space-x-2">
                        <Switch id="gdpr" />
                        <Label htmlFor="gdpr">GDPR Compliant</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Switch id="hipaa" />
                        <Label htmlFor="hipaa">HIPAA Compliant</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Switch id="sox" />
                        <Label htmlFor="sox">SOX Compliant</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Switch id="pci" />
                        <Label htmlFor="pci">PCI DSS</Label>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Preview Panel */}
              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Eye className="h-5 w-5" />
                      <span>Preview</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="text-center">
                        <div className="w-16 h-16 bg-gradient-forge rounded-full flex items-center justify-center mx-auto mb-3">
                          <Bot className="h-8 w-8 text-white" />
                        </div>
                        <h3 className="font-semibold">
                          {employeeConfig.name || 'Unnamed Employee'}
                        </h3>
                        <p className="text-sm text-muted-foreground">
                          {employeeConfig.type || 'No type selected'}
                        </p>
                      </div>

                      <Separator />

                      <div className="space-y-3">
                        <div>
                          <Label className="text-xs font-medium text-muted-foreground">
                            PERFORMANCE TARGETS
                          </Label>
                          <div className="mt-1 space-y-2">
                            <div className="flex justify-between text-sm">
                              <span>Accuracy</span>
                              <span className="font-medium">
                                {employeeConfig.performance_targets.accuracy}%
                              </span>
                            </div>
                            <div className="flex justify-between text-sm">
                              <span>Speed</span>
                              <span className="font-medium">
                                {employeeConfig.performance_targets.speed}x
                              </span>
                            </div>
                            <div className="flex justify-between text-sm">
                              <span>Availability</span>
                              <span className="font-medium">
                                {employeeConfig.performance_targets.availability}%
                              </span>
                            </div>
                          </div>
                        </div>

                        <div>
                          <Label className="text-xs font-medium text-muted-foreground">
                            SECURITY LEVEL
                          </Label>
                          <Badge variant="outline" className="mt-1">
                            {employeeConfig.security_level}
                          </Badge>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Sparkles className="h-5 w-5" />
                      <span>AI Suggestions</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3 text-sm">
                      <div className="p-3 bg-muted rounded-lg">
                        <p className="font-medium mb-1">💡 Performance Tip</p>
                        <p className="text-muted-foreground">
                          Consider adding "multilingual" capability for 15% better customer satisfaction.
                        </p>
                      </div>
                      <div className="p-3 bg-muted rounded-lg">
                        <p className="font-medium mb-1">🔒 Security Recommendation</p>
                        <p className="text-muted-foreground">
                          Enable GDPR compliance for European customer data handling.
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Skills & Personality */}
          <TabsContent value="skills" className="space-y-6">
            <div className="grid lg:grid-cols-2 gap-6">
              <SkillsEditor
                skills={employeeConfig.skills}
                onSkillsChange={(skills) => handleConfigUpdate({ skills })}
              />
              <PersonalityEditor
                personality={employeeConfig.personality}
                onPersonalityChange={(personality) => handleConfigUpdate({ personality })}
              />
            </div>
          </TabsContent>

          {/* Workflow Designer */}
          <TabsContent value="workflow" className="space-y-6">
            <WorkflowDesigner
              employeeConfig={employeeConfig}
              onConfigUpdate={handleConfigUpdate}
            />
          </TabsContent>

          {/* Performance Validation */}
          <TabsContent value="validation" className="space-y-6">
            <PerformanceValidator
              employeeConfig={employeeConfig}
              onValidationComplete={(results) => {
                toast({
                  title: "Validation Complete",
                  description: `Performance score: ${results.overall_score}%`,
                })
              }}
            />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}