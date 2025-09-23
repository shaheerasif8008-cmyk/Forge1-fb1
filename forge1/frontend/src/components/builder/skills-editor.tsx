'use client'

import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { 
  Plus, 
  Trash2, 
  Brain, 
  MessageSquare, 
  BarChart3, 
  Code, 
  Globe, 
  Shield,
  Search,
  Star,
  TrendingUp
} from 'lucide-react'
import { Skill } from '@/types/employee'

interface SkillsEditorProps {
  skills: Skill[]
  onSkillsChange: (skills: Skill[]) => void
}

const skillCategories = {
  communication: {
    name: 'Communication',
    icon: MessageSquare,
    color: 'text-blue-500',
    skills: [
      { id: 'verbal-communication', name: 'Verbal Communication', description: 'Clear and effective spoken communication' },
      { id: 'written-communication', name: 'Written Communication', description: 'Professional writing and documentation' },
      { id: 'active-listening', name: 'Active Listening', description: 'Understanding and responding to customer needs' },
      { id: 'multilingual', name: 'Multilingual Support', description: 'Communication in multiple languages' },
      { id: 'empathy', name: 'Empathy', description: 'Understanding and relating to emotions' },
      { id: 'conflict-resolution', name: 'Conflict Resolution', description: 'Resolving disputes and disagreements' }
    ]
  },
  analytical: {
    name: 'Analytical',
    icon: BarChart3,
    color: 'text-green-500',
    skills: [
      { id: 'data-analysis', name: 'Data Analysis', description: 'Analyzing and interpreting data patterns' },
      { id: 'statistical-analysis', name: 'Statistical Analysis', description: 'Advanced statistical methods and modeling' },
      { id: 'pattern-recognition', name: 'Pattern Recognition', description: 'Identifying trends and anomalies' },
      { id: 'forecasting', name: 'Forecasting', description: 'Predicting future trends and outcomes' },
      { id: 'research', name: 'Research', description: 'Gathering and synthesizing information' },
      { id: 'critical-thinking', name: 'Critical Thinking', description: 'Logical reasoning and problem-solving' }
    ]
  },
  technical: {
    name: 'Technical',
    icon: Code,
    color: 'text-purple-500',
    skills: [
      { id: 'programming', name: 'Programming', description: 'Software development and coding' },
      { id: 'database-management', name: 'Database Management', description: 'SQL and database operations' },
      { id: 'api-integration', name: 'API Integration', description: 'Working with REST and GraphQL APIs' },
      { id: 'automation', name: 'Automation', description: 'Process automation and scripting' },
      { id: 'cybersecurity', name: 'Cybersecurity', description: 'Security protocols and threat detection' },
      { id: 'cloud-computing', name: 'Cloud Computing', description: 'Cloud platforms and services' }
    ]
  },
  creative: {
    name: 'Creative',
    icon: Brain,
    color: 'text-orange-500',
    skills: [
      { id: 'content-creation', name: 'Content Creation', description: 'Creating engaging written content' },
      { id: 'design-thinking', name: 'Design Thinking', description: 'User-centered problem solving' },
      { id: 'storytelling', name: 'Storytelling', description: 'Crafting compelling narratives' },
      { id: 'innovation', name: 'Innovation', description: 'Generating creative solutions' },
      { id: 'visual-design', name: 'Visual Design', description: 'Creating visual content and layouts' },
      { id: 'brand-strategy', name: 'Brand Strategy', description: 'Developing brand positioning and messaging' }
    ]
  },
  business: {
    name: 'Business',
    icon: TrendingUp,
    color: 'text-red-500',
    skills: [
      { id: 'project-management', name: 'Project Management', description: 'Planning and executing projects' },
      { id: 'sales', name: 'Sales', description: 'Lead generation and conversion' },
      { id: 'marketing', name: 'Marketing', description: 'Promotional strategies and campaigns' },
      { id: 'customer-service', name: 'Customer Service', description: 'Supporting and satisfying customers' },
      { id: 'negotiation', name: 'Negotiation', description: 'Reaching mutually beneficial agreements' },
      { id: 'strategic-planning', name: 'Strategic Planning', description: 'Long-term business planning' }
    ]
  },
  specialized: {
    name: 'Specialized',
    icon: Star,
    color: 'text-indigo-500',
    skills: [
      { id: 'legal-knowledge', name: 'Legal Knowledge', description: 'Understanding of legal principles' },
      { id: 'financial-analysis', name: 'Financial Analysis', description: 'Financial modeling and analysis' },
      { id: 'healthcare-knowledge', name: 'Healthcare Knowledge', description: 'Medical and healthcare expertise' },
      { id: 'compliance', name: 'Compliance', description: 'Regulatory compliance and auditing' },
      { id: 'quality-assurance', name: 'Quality Assurance', description: 'Testing and quality control' },
      { id: 'risk-management', name: 'Risk Management', description: 'Identifying and mitigating risks' }
    ]
  }
}

export function SkillsEditor({ skills, onSkillsChange }: SkillsEditorProps) {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [customSkillName, setCustomSkillName] = useState('')
  const [customSkillDescription, setCustomSkillDescription] = useState('')

  const addSkill = useCallback((skillTemplate: any, proficiency: number = 80) => {
    const newSkill: Skill = {
      id: skillTemplate.id,
      name: skillTemplate.name,
      description: skillTemplate.description,
      proficiency,
      category: Object.keys(skillCategories).find(cat => 
        skillCategories[cat as keyof typeof skillCategories].skills.some(s => s.id === skillTemplate.id)
      ) || 'general',
      required: false
    }

    const existingIndex = skills.findIndex(s => s.id === skillTemplate.id)
    if (existingIndex >= 0) {
      const updatedSkills = [...skills]
      updatedSkills[existingIndex] = newSkill
      onSkillsChange(updatedSkills)
    } else {
      onSkillsChange([...skills, newSkill])
    }
  }, [skills, onSkillsChange])

  const removeSkill = useCallback((skillId: string) => {
    onSkillsChange(skills.filter(s => s.id !== skillId))
  }, [skills, onSkillsChange])

  const updateSkillProficiency = useCallback((skillId: string, proficiency: number) => {
    const updatedSkills = skills.map(skill => 
      skill.id === skillId ? { ...skill, proficiency } : skill
    )
    onSkillsChange(updatedSkills)
  }, [skills, onSkillsChange])

  const addCustomSkill = useCallback(() => {
    if (!customSkillName.trim()) return

    const customSkill: Skill = {
      id: `custom-${Date.now()}`,
      name: customSkillName,
      description: customSkillDescription || `Custom skill: ${customSkillName}`,
      proficiency: 80,
      category: 'custom',
      required: false
    }

    onSkillsChange([...skills, customSkill])
    setCustomSkillName('')
    setCustomSkillDescription('')
  }, [customSkillName, customSkillDescription, skills, onSkillsChange])

  const filteredCategories = Object.entries(skillCategories).filter(([key, category]) => {
    if (selectedCategory !== 'all' && key !== selectedCategory) return false
    if (!searchTerm) return true
    
    return category.skills.some(skill => 
      skill.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      skill.description.toLowerCase().includes(searchTerm.toLowerCase())
    )
  })

  const getSkillProficiency = (skillId: string) => {
    const skill = skills.find(s => s.id === skillId)
    return skill?.proficiency || 0
  }

  const isSkillAdded = (skillId: string) => {
    return skills.some(s => s.id === skillId)
  }

  return (
    <div className="space-y-6">
      {/* Current Skills */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-5 w-5" />
            <span>Current Skills</span>
            <Badge variant="secondary">{skills.length}</Badge>
          </CardTitle>
          <CardDescription>
            Configure your AI employee's capabilities and proficiency levels
          </CardDescription>
        </CardHeader>
        <CardContent>
          {skills.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No skills added yet. Browse the skill library below to get started.</p>
            </div>
          ) : (
            <div className="space-y-4">
              {skills.map((skill) => (
                <motion.div
                  key={skill.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="p-4 border rounded-lg space-y-3"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <h4 className="font-medium">{skill.name}</h4>
                        <Badge variant="outline" className="text-xs">
                          {skill.category}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">
                        {skill.description}
                      </p>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeSkill(skill.id)}
                      className="text-destructive hover:text-destructive"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <Label>Proficiency Level</Label>
                      <span className="font-medium">{skill.proficiency}%</span>
                    </div>
                    <Slider
                      value={[skill.proficiency]}
                      onValueChange={([value]) => updateSkillProficiency(skill.id, value)}
                      max={100}
                      min={0}
                      step={5}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Beginner</span>
                      <span>Expert</span>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Skill Library */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Search className="h-5 w-5" />
            <span>Skill Library</span>
          </CardTitle>
          <CardDescription>
            Browse and add skills from our comprehensive library
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Search and Filter */}
          <div className="flex space-x-4">
            <div className="flex-1">
              <Input
                placeholder="Search skills..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full"
              />
            </div>
            <Select value={selectedCategory} onValueChange={setSelectedCategory}>
              <SelectTrigger className="w-48">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Categories</SelectItem>
                {Object.entries(skillCategories).map(([key, category]) => (
                  <SelectItem key={key} value={key}>
                    {category.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Skill Categories */}
          <ScrollArea className="h-96">
            <div className="space-y-6">
              {filteredCategories.map(([categoryKey, category]) => (
                <div key={categoryKey} className="space-y-3">
                  <div className="flex items-center space-x-2">
                    <category.icon className={`h-5 w-5 ${category.color}`} />
                    <h3 className="font-semibold">{category.name}</h3>
                  </div>
                  
                  <div className="grid gap-2">
                    {category.skills
                      .filter(skill => 
                        !searchTerm || 
                        skill.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                        skill.description.toLowerCase().includes(searchTerm.toLowerCase())
                      )
                      .map((skill) => (
                        <div
                          key={skill.id}
                          className="flex items-center justify-between p-3 border rounded-lg hover:bg-muted/50 transition-colors"
                        >
                          <div className="flex-1">
                            <div className="flex items-center space-x-2">
                              <h4 className="font-medium text-sm">{skill.name}</h4>
                              {isSkillAdded(skill.id) && (
                                <Badge variant="secondary" className="text-xs">
                                  {getSkillProficiency(skill.id)}%
                                </Badge>
                              )}
                            </div>
                            <p className="text-xs text-muted-foreground mt-1">
                              {skill.description}
                            </p>
                          </div>
                          
                          <Button
                            variant={isSkillAdded(skill.id) ? "secondary" : "outline"}
                            size="sm"
                            onClick={() => addSkill(skill)}
                            disabled={isSkillAdded(skill.id)}
                          >
                            {isSkillAdded(skill.id) ? (
                              "Added"
                            ) : (
                              <>
                                <Plus className="h-3 w-3 mr-1" />
                                Add
                              </>
                            )}
                          </Button>
                        </div>
                      ))}
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>

          <Separator />

          {/* Custom Skill */}
          <div className="space-y-4">
            <h3 className="font-semibold">Add Custom Skill</h3>
            <div className="grid gap-4">
              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="custom-skill-name">Skill Name</Label>
                  <Input
                    id="custom-skill-name"
                    value={customSkillName}
                    onChange={(e) => setCustomSkillName(e.target.value)}
                    placeholder="e.g., Industry-specific knowledge"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="custom-skill-description">Description</Label>
                  <Input
                    id="custom-skill-description"
                    value={customSkillDescription}
                    onChange={(e) => setCustomSkillDescription(e.target.value)}
                    placeholder="Brief description of the skill"
                  />
                </div>
              </div>
              <Button
                onClick={addCustomSkill}
                disabled={!customSkillName.trim()}
                className="w-fit"
              >
                <Plus className="h-4 w-4 mr-2" />
                Add Custom Skill
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}