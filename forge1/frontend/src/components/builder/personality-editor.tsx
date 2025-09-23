'use client'

import { useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  Heart, 
  Brain, 
  Zap, 
  Shield, 
  Users, 
  Target,
  Smile,
  Clock,
  Lightbulb,
  CheckCircle,
  AlertTriangle,
  RotateCcw
} from 'lucide-react'
import { Personality } from '@/types/employee'

interface PersonalityEditorProps {
  personality: Personality
  onPersonalityChange: (personality: Personality) => void
}

const personalityTraits = {
  core: {
    name: 'Core Traits',
    icon: Heart,
    color: 'text-red-500',
    traits: [
      {
        id: 'empathy',
        name: 'Empathy',
        description: 'Understanding and sharing feelings with others',
        icon: Heart,
        min: 0,
        max: 100,
        default: 70,
        impact: 'Higher empathy improves customer satisfaction but may slow decision-making'
      },
      {
        id: 'assertiveness',
        name: 'Assertiveness',
        description: 'Confidence in expressing opinions and making decisions',
        icon: Target,
        min: 0,
        max: 100,
        default: 60,
        impact: 'Higher assertiveness improves leadership but may reduce collaboration'
      },
      {
        id: 'patience',
        name: 'Patience',
        description: 'Ability to remain calm and persistent',
        icon: Clock,
        min: 0,
        max: 100,
        default: 80,
        impact: 'Higher patience improves customer service but may slow task completion'
      },
      {
        id: 'optimism',
        name: 'Optimism',
        description: 'Positive outlook and solution-focused thinking',
        icon: Smile,
        min: 0,
        max: 100,
        default: 75,
        impact: 'Higher optimism improves team morale and customer interactions'
      }
    ]
  },
  cognitive: {
    name: 'Cognitive Style',
    icon: Brain,
    color: 'text-blue-500',
    traits: [
      {
        id: 'analytical',
        name: 'Analytical Thinking',
        description: 'Systematic approach to problem-solving',
        icon: Brain,
        min: 0,
        max: 100,
        default: 85,
        impact: 'Higher analytical thinking improves accuracy but may slow responses'
      },
      {
        id: 'creativity',
        name: 'Creativity',
        description: 'Innovative and original thinking',
        icon: Lightbulb,
        min: 0,
        max: 100,
        default: 65,
        impact: 'Higher creativity improves problem-solving but may reduce consistency'
      },
      {
        id: 'attention_to_detail',
        name: 'Attention to Detail',
        description: 'Focus on accuracy and thoroughness',
        icon: CheckCircle,
        min: 0,
        max: 100,
        default: 90,
        impact: 'Higher attention to detail improves quality but may slow processing'
      },
      {
        id: 'adaptability',
        name: 'Adaptability',
        description: 'Flexibility in changing situations',
        icon: RotateCcw,
        min: 0,
        max: 100,
        default: 70,
        impact: 'Higher adaptability improves versatility but may reduce specialization'
      }
    ]
  },
  social: {
    name: 'Social Traits',
    icon: Users,
    color: 'text-green-500',
    traits: [
      {
        id: 'communication',
        name: 'Communication Style',
        description: 'Clarity and effectiveness in communication',
        icon: Users,
        min: 0,
        max: 100,
        default: 85,
        impact: 'Higher communication improves understanding and relationships'
      },
      {
        id: 'collaboration',
        name: 'Collaboration',
        description: 'Working effectively with others',
        icon: Users,
        min: 0,
        max: 100,
        default: 75,
        impact: 'Higher collaboration improves teamwork but may slow individual tasks'
      },
      {
        id: 'leadership',
        name: 'Leadership',
        description: 'Ability to guide and influence others',
        icon: Target,
        min: 0,
        max: 100,
        default: 60,
        impact: 'Higher leadership improves team coordination and decision-making'
      },
      {
        id: 'cultural_sensitivity',
        name: 'Cultural Sensitivity',
        description: 'Awareness and respect for cultural differences',
        icon: Heart,
        min: 0,
        max: 100,
        default: 80,
        impact: 'Higher cultural sensitivity improves global customer interactions'
      }
    ]
  },
  work: {
    name: 'Work Style',
    icon: Zap,
    color: 'text-orange-500',
    traits: [
      {
        id: 'proactivity',
        name: 'Proactivity',
        description: 'Taking initiative and anticipating needs',
        icon: Zap,
        min: 0,
        max: 100,
        default: 75,
        impact: 'Higher proactivity improves efficiency and customer satisfaction'
      },
      {
        id: 'persistence',
        name: 'Persistence',
        description: 'Continuing effort despite challenges',
        icon: Target,
        min: 0,
        max: 100,
        default: 80,
        impact: 'Higher persistence improves problem resolution rates'
      },
      {
        id: 'risk_tolerance',
        name: 'Risk Tolerance',
        description: 'Comfort with uncertainty and calculated risks',
        icon: AlertTriangle,
        min: 0,
        max: 100,
        default: 50,
        impact: 'Higher risk tolerance enables innovation but may reduce safety'
      },
      {
        id: 'efficiency_focus',
        name: 'Efficiency Focus',
        description: 'Emphasis on speed and resource optimization',
        icon: Zap,
        min: 0,
        max: 100,
        default: 85,
        impact: 'Higher efficiency focus improves speed but may reduce thoroughness'
      }
    ]
  }
}

const personalityPresets = [
  {
    id: 'customer-service',
    name: 'Customer Service Specialist',
    description: 'Optimized for customer support and satisfaction',
    personality: {
      empathy: 95,
      patience: 90,
      communication: 90,
      assertiveness: 60,
      analytical: 70,
      creativity: 60,
      attention_to_detail: 85,
      adaptability: 80,
      collaboration: 85,
      cultural_sensitivity: 90,
      proactivity: 80,
      persistence: 85,
      optimism: 85,
      leadership: 50,
      risk_tolerance: 40,
      efficiency_focus: 75
    }
  },
  {
    id: 'data-analyst',
    name: 'Data Analyst',
    description: 'Focused on analytical thinking and accuracy',
    personality: {
      analytical: 95,
      attention_to_detail: 95,
      creativity: 70,
      patience: 85,
      empathy: 60,
      communication: 75,
      assertiveness: 70,
      adaptability: 65,
      collaboration: 70,
      cultural_sensitivity: 70,
      proactivity: 80,
      persistence: 90,
      optimism: 70,
      leadership: 60,
      risk_tolerance: 60,
      efficiency_focus: 90
    }
  },
  {
    id: 'creative-writer',
    name: 'Creative Writer',
    description: 'Optimized for creative content and storytelling',
    personality: {
      creativity: 95,
      communication: 90,
      empathy: 85,
      adaptability: 85,
      analytical: 70,
      attention_to_detail: 80,
      patience: 75,
      assertiveness: 65,
      collaboration: 75,
      cultural_sensitivity: 85,
      proactivity: 85,
      persistence: 80,
      optimism: 90,
      leadership: 60,
      risk_tolerance: 70,
      efficiency_focus: 70
    }
  },
  {
    id: 'project-manager',
    name: 'Project Manager',
    description: 'Leadership and coordination focused',
    personality: {
      leadership: 90,
      assertiveness: 85,
      communication: 90,
      collaboration: 90,
      analytical: 80,
      attention_to_detail: 85,
      adaptability: 85,
      patience: 80,
      empathy: 75,
      creativity: 70,
      cultural_sensitivity: 80,
      proactivity: 90,
      persistence: 85,
      optimism: 80,
      risk_tolerance: 65,
      efficiency_focus: 85
    }
  }
]

export function PersonalityEditor({ personality, onPersonalityChange }: PersonalityEditorProps) {
  const [activePreset, setActivePreset] = useState<string | null>(null)

  const updateTrait = useCallback((traitId: string, value: number) => {
    onPersonalityChange({
      ...personality,
      [traitId]: value
    })
  }, [personality, onPersonalityChange])

  const applyPreset = useCallback((preset: typeof personalityPresets[0]) => {
    onPersonalityChange(preset.personality)
    setActivePreset(preset.id)
  }, [onPersonalityChange])

  const resetToDefaults = useCallback(() => {
    const defaultPersonality: Personality = {}
    
    Object.values(personalityTraits).forEach(category => {
      category.traits.forEach(trait => {
        defaultPersonality[trait.id] = trait.default
      })
    })
    
    onPersonalityChange(defaultPersonality)
    setActivePreset(null)
  }, [onPersonalityChange])

  const getTraitValue = (traitId: string, defaultValue: number) => {
    return personality[traitId] ?? defaultValue
  }

  const getPersonalityScore = () => {
    const values = Object.values(personality).filter(v => typeof v === 'number')
    if (values.length === 0) return 0
    return Math.round(values.reduce((sum, val) => sum + val, 0) / values.length)
  }

  const getTraitLevel = (value: number) => {
    if (value >= 90) return { label: 'Very High', color: 'text-green-600' }
    if (value >= 75) return { label: 'High', color: 'text-green-500' }
    if (value >= 60) return { label: 'Moderate', color: 'text-yellow-500' }
    if (value >= 40) return { label: 'Low', color: 'text-orange-500' }
    return { label: 'Very Low', color: 'text-red-500' }
  }

  return (
    <div className="space-y-6">
      {/* Personality Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Heart className="h-5 w-5" />
            <span>Personality Profile</span>
            <Badge variant="secondary">{getPersonalityScore()}% Configured</Badge>
          </CardTitle>
          <CardDescription>
            Define your AI employee's personality traits and behavioral patterns
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2 mb-4">
            {personalityPresets.map((preset) => (
              <Button
                key={preset.id}
                variant={activePreset === preset.id ? "default" : "outline"}
                size="sm"
                onClick={() => applyPreset(preset)}
              >
                {preset.name}
              </Button>
            ))}
            <Button
              variant="outline"
              size="sm"
              onClick={resetToDefaults}
            >
              <RotateCcw className="h-3 w-3 mr-1" />
              Reset
            </Button>
          </div>
          
          {activePreset && (
            <div className="p-3 bg-muted rounded-lg mb-4">
              <p className="text-sm">
                <strong>{personalityPresets.find(p => p.id === activePreset)?.name}:</strong>{' '}
                {personalityPresets.find(p => p.id === activePreset)?.description}
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Personality Traits */}
      <Tabs defaultValue="core" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          {Object.entries(personalityTraits).map(([key, category]) => (
            <TabsTrigger key={key} value={key} className="flex items-center space-x-2">
              <category.icon className="h-4 w-4" />
              <span className="hidden sm:inline">{category.name}</span>
            </TabsTrigger>
          ))}
        </TabsList>

        {Object.entries(personalityTraits).map(([categoryKey, category]) => (
          <TabsContent key={categoryKey} value={categoryKey}>
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <category.icon className={`h-5 w-5 ${category.color}`} />
                  <span>{category.name}</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {category.traits.map((trait) => {
                  const value = getTraitValue(trait.id, trait.default)
                  const level = getTraitLevel(value)
                  
                  return (
                    <motion.div
                      key={trait.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="space-y-3"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <trait.icon className="h-4 w-4 text-muted-foreground" />
                          <div>
                            <Label className="font-medium">{trait.name}</Label>
                            <p className="text-sm text-muted-foreground">
                              {trait.description}
                            </p>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="font-medium">{value}%</div>
                          <div className={`text-xs ${level.color}`}>
                            {level.label}
                          </div>
                        </div>
                      </div>
                      
                      <Slider
                        value={[value]}
                        onValueChange={([newValue]) => updateTrait(trait.id, newValue)}
                        max={trait.max}
                        min={trait.min}
                        step={5}
                        className="w-full"
                      />
                      
                      <div className="text-xs text-muted-foreground bg-muted p-2 rounded">
                        <strong>Impact:</strong> {trait.impact}
                      </div>
                    </motion.div>
                  )
                })}
              </CardContent>
            </Card>
          </TabsContent>
        ))}
      </Tabs>

      {/* Personality Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-5 w-5" />
            <span>Personality Summary</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="font-medium">Strengths</h4>
              <div className="space-y-2">
                {Object.entries(personalityTraits).map(([categoryKey, category]) => {
                  const highTraits = category.traits.filter(trait => 
                    getTraitValue(trait.id, trait.default) >= 80
                  )
                  
                  return highTraits.map(trait => (
                    <div key={trait.id} className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span className="text-sm">{trait.name}</span>
                      <Badge variant="secondary" className="text-xs">
                        {getTraitValue(trait.id, trait.default)}%
                      </Badge>
                    </div>
                  ))
                })}
              </div>
            </div>
            
            <div className="space-y-4">
              <h4 className="font-medium">Areas for Development</h4>
              <div className="space-y-2">
                {Object.entries(personalityTraits).map(([categoryKey, category]) => {
                  const lowTraits = category.traits.filter(trait => 
                    getTraitValue(trait.id, trait.default) < 60
                  )
                  
                  return lowTraits.map(trait => (
                    <div key={trait.id} className="flex items-center space-x-2">
                      <AlertTriangle className="h-4 w-4 text-orange-500" />
                      <span className="text-sm">{trait.name}</span>
                      <Badge variant="outline" className="text-xs">
                        {getTraitValue(trait.id, trait.default)}%
                      </Badge>
                    </div>
                  ))
                })}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}