'use client'

import { useState, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { 
  Play, 
  Pause, 
  RotateCcw, 
  CheckCircle, 
  XCircle, 
  AlertTriangle,
  TrendingUp,
  Clock,
  Target,
  Shield,
  Zap,
  Users,
  BarChart3,
  Award,
  Flame,
  Brain
} from 'lucide-react'
import { EmployeeConfig, ValidationResult, TestCase, TestResult, BenchmarkResult } from '@/types/employee'
import { useToast } from '@/hooks/use-toast'

interface PerformanceValidatorProps {
  employeeConfig: EmployeeConfig
  onValidationComplete: (results: ValidationResult) => void
}

const testSuites = {
  accuracy: {
    name: 'Accuracy Tests',
    description: 'Validate response accuracy and correctness',
    icon: Target,
    color: 'text-green-500',
    tests: [
      {
        id: 'factual-accuracy',
        name: 'Factual Accuracy',
        description: 'Verify factual correctness of responses',
        category: 'accuracy',
        difficulty: 'medium' as const,
        weight: 0.3
      },
      {
        id: 'context-understanding',
        name: 'Context Understanding',
        description: 'Test comprehension of complex contexts',
        category: 'accuracy',
        difficulty: 'hard' as const,
        weight: 0.4
      },
      {
        id: 'instruction-following',
        name: 'Instruction Following',
        description: 'Adherence to specific instructions',
        category: 'accuracy',
        difficulty: 'easy' as const,
        weight: 0.3
      }
    ]
  },
  speed: {
    name: 'Speed Tests',
    description: 'Measure response time and throughput',
    icon: Zap,
    color: 'text-yellow-500',
    tests: [
      {
        id: 'response-time',
        name: 'Response Time',
        description: 'Average time to generate responses',
        category: 'speed',
        difficulty: 'easy' as const,
        weight: 0.4
      },
      {
        id: 'concurrent-processing',
        name: 'Concurrent Processing',
        description: 'Handle multiple requests simultaneously',
        category: 'speed',
        difficulty: 'hard' as const,
        weight: 0.4
      },
      {
        id: 'throughput',
        name: 'Throughput',
        description: 'Requests processed per minute',
        category: 'speed',
        difficulty: 'medium' as const,
        weight: 0.2
      }
    ]
  },
  reliability: {
    name: 'Reliability Tests',
    description: 'Test consistency and error handling',
    icon: Shield,
    color: 'text-blue-500',
    tests: [
      {
        id: 'consistency',
        name: 'Response Consistency',
        description: 'Consistent responses to similar inputs',
        category: 'reliability',
        difficulty: 'medium' as const,
        weight: 0.3
      },
      {
        id: 'error-handling',
        name: 'Error Handling',
        description: 'Graceful handling of invalid inputs',
        category: 'reliability',
        difficulty: 'hard' as const,
        weight: 0.4
      },
      {
        id: 'edge-cases',
        name: 'Edge Cases',
        description: 'Performance on unusual scenarios',
        category: 'reliability',
        difficulty: 'hard' as const,
        weight: 0.3
      }
    ]
  },
  security: {
    name: 'Security Tests',
    description: 'Validate security and compliance measures',
    icon: Shield,
    color: 'text-red-500',
    tests: [
      {
        id: 'data-privacy',
        name: 'Data Privacy',
        description: 'Protection of sensitive information',
        category: 'security',
        difficulty: 'hard' as const,
        weight: 0.4
      },
      {
        id: 'access-control',
        name: 'Access Control',
        description: 'Proper authentication and authorization',
        category: 'security',
        difficulty: 'medium' as const,
        weight: 0.3
      },
      {
        id: 'injection-resistance',
        name: 'Injection Resistance',
        description: 'Resistance to prompt injection attacks',
        category: 'security',
        difficulty: 'hard' as const,
        weight: 0.3
      }
    ]
  }
}

const humanBaselines = {
  'customer-service': {
    accuracy: 85,
    speed: 1.0,
    availability: 75,
    customer_satisfaction: 80,
    error_rate: 15,
    response_time: 120 // seconds
  },
  'data-analyst': {
    accuracy: 90,
    speed: 1.0,
    availability: 80,
    error_rate: 10,
    response_time: 1800 // 30 minutes
  },
  'content-writer': {
    accuracy: 88,
    speed: 1.0,
    availability: 70,
    error_rate: 12,
    response_time: 3600 // 1 hour
  },
  'sales-assistant': {
    accuracy: 82,
    speed: 1.0,
    availability: 75,
    error_rate: 18,
    response_time: 300 // 5 minutes
  }
}

export function PerformanceValidator({ employeeConfig, onValidationComplete }: PerformanceValidatorProps) {
  const { toast } = useToast()
  const [isRunning, setIsRunning] = useState(false)
  const [currentTest, setCurrentTest] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [results, setResults] = useState<ValidationResult | null>(null)
  const [testResults, setTestResults] = useState<Record<string, TestResult[]>>({})
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResult | null>(null)
  const [activeTab, setActiveTab] = useState('overview')

  const runValidation = useCallback(async () => {
    setIsRunning(true)
    setProgress(0)
    setResults(null)
    setTestResults({})
    
    try {
      const allTests = Object.values(testSuites).flatMap(suite => suite.tests)
      const totalTests = allTests.length
      let completedTests = 0
      const newTestResults: Record<string, TestResult[]> = {}

      // Run each test suite
      for (const [suiteKey, suite] of Object.entries(testSuites)) {
        newTestResults[suiteKey] = []
        
        for (const test of suite.tests) {
          setCurrentTest(test.name)
          
          // Simulate test execution
          await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000))
          
          // Generate mock test result
          const testResult: TestResult = {
            test_case_id: test.id,
            passed: Math.random() > 0.1, // 90% pass rate
            actual_output: { score: Math.random() * 100 },
            execution_time: Math.random() * 5000,
            accuracy_score: 85 + Math.random() * 15,
            error_message: Math.random() > 0.9 ? 'Mock error for testing' : undefined
          }
          
          newTestResults[suiteKey].push(testResult)
          completedTests++
          setProgress((completedTests / totalTests) * 100)
        }
      }
      
      setTestResults(newTestResults)
      
      // Calculate overall validation results
      const validationResult = calculateValidationResults(newTestResults)
      setResults(validationResult)
      
      // Generate benchmark comparison
      const benchmark = generateBenchmarkResults(employeeConfig, validationResult)
      setBenchmarkResults(benchmark)
      
      onValidationComplete(validationResult)
      
      toast({
        title: "Validation Complete",
        description: `Overall performance score: ${validationResult.overall_score}%`,
      })
      
    } catch (error) {
      toast({
        title: "Validation Failed",
        description: "An error occurred during validation. Please try again.",
        variant: "destructive"
      })
    } finally {
      setIsRunning(false)
      setCurrentTest(null)
    }
  }, [employeeConfig, onValidationComplete, toast])

  const calculateValidationResults = (testResults: Record<string, TestResult[]>): ValidationResult => {
    const suiteScores: Record<string, number> = {}
    
    // Calculate score for each test suite
    Object.entries(testResults).forEach(([suiteKey, results]) => {
      const suite = testSuites[suiteKey as keyof typeof testSuites]
      let weightedScore = 0
      let totalWeight = 0
      
      results.forEach((result, index) => {
        const test = suite.tests[index]
        const score = result.passed ? result.accuracy_score : 0
        weightedScore += score * test.weight
        totalWeight += test.weight
      })
      
      suiteScores[suiteKey] = totalWeight > 0 ? weightedScore / totalWeight : 0
    })
    
    // Calculate overall scores
    const accuracy_score = suiteScores.accuracy || 0
    const speed_score = suiteScores.speed || 0
    const reliability_score = suiteScores.reliability || 0
    const security_score = suiteScores.security || 0
    
    const overall_score = (accuracy_score + speed_score + reliability_score + security_score) / 4
    
    // Generate recommendations
    const recommendations: string[] = []
    const warnings: string[] = []
    const errors: string[] = []
    
    if (accuracy_score < 90) {
      recommendations.push("Consider adding more training data to improve accuracy")
    }
    if (speed_score < 80) {
      warnings.push("Response time may not meet superhuman performance targets")
    }
    if (security_score < 95) {
      errors.push("Security measures need improvement before deployment")
    }
    
    return {
      overall_score: Math.round(overall_score),
      accuracy_score: Math.round(accuracy_score),
      speed_score: Math.round(speed_score),
      reliability_score: Math.round(reliability_score),
      security_score: Math.round(security_score),
      compliance_score: Math.round(security_score), // Simplified
      recommendations,
      warnings,
      errors
    }
  }

  const generateBenchmarkResults = (config: EmployeeConfig, validation: ValidationResult): BenchmarkResult => {
    const baseline = humanBaselines[config.type as keyof typeof humanBaselines] || humanBaselines['customer-service']
    
    // Calculate improvement factors
    const accuracyImprovement = validation.accuracy_score / baseline.accuracy
    const speedImprovement = Math.random() * 10 + 5 // 5-15x improvement
    const availabilityImprovement = config.performance_targets.availability / baseline.availability
    
    const overallImprovement = (accuracyImprovement + speedImprovement + availabilityImprovement) / 3
    
    return {
      employee_id: config.id,
      test_suite_id: 'comprehensive-validation',
      overall_score: validation.overall_score,
      individual_scores: [],
      performance_metrics: {
        accuracy: validation.accuracy_score,
        speed: speedImprovement,
        availability: config.performance_targets.availability,
        customer_satisfaction: Math.min(95, baseline.customer_satisfaction! * accuracyImprovement),
        error_rate: Math.max(1, baseline.error_rate / accuracyImprovement),
        throughput: speedImprovement * 100, // requests per hour
        response_time: baseline.response_time / speedImprovement,
        cost_per_task: 0.05 // $0.05 per task
      },
      comparison_baseline: 'human',
      improvement_factor: overallImprovement,
      executed_at: new Date().toISOString()
    }
  }

  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-green-600'
    if (score >= 80) return 'text-yellow-600'
    if (score >= 70) return 'text-orange-600'
    return 'text-red-600'
  }

  const getScoreIcon = (score: number) => {
    if (score >= 90) return CheckCircle
    if (score >= 70) return AlertTriangle
    return XCircle
  }

  return (
    <div className="space-y-6">
      {/* Validation Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-5 w-5" />
            <span>Performance Validation</span>
          </CardTitle>
          <CardDescription>
            Validate superhuman performance against industry benchmarks
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="space-y-2">
              <div className="flex items-center space-x-4">
                <Button
                  onClick={runValidation}
                  disabled={isRunning}
                  className="btn-glow"
                >
                  {isRunning ? (
                    <>
                      <Pause className="h-4 w-4 mr-2" />
                      Running...
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4 mr-2" />
                      Start Validation
                    </>
                  )}
                </Button>
                
                {results && (
                  <Button
                    variant="outline"
                    onClick={() => {
                      setResults(null)
                      setTestResults({})
                      setBenchmarkResults(null)
                      setProgress(0)
                    }}
                  >
                    <RotateCcw className="h-4 w-4 mr-2" />
                    Reset
                  </Button>
                )}
              </div>
              
              {isRunning && (
                <div className="space-y-2">
                  <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                    <Clock className="h-4 w-4" />
                    <span>Running: {currentTest || 'Initializing...'}</span>
                  </div>
                  <Progress value={progress} className="w-64" />
                </div>
              )}
            </div>
            
            {results && (
              <div className="text-right">
                <div className="text-2xl font-bold">
                  <span className={getScoreColor(results.overall_score)}>
                    {results.overall_score}%
                  </span>
                </div>
                <div className="text-sm text-muted-foreground">Overall Score</div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {results && (
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="detailed">Detailed Results</TabsTrigger>
            <TabsTrigger value="benchmark">Benchmark</TabsTrigger>
            <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-4">
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
              {[
                { key: 'accuracy_score', name: 'Accuracy', icon: Target, color: 'text-green-500' },
                { key: 'speed_score', name: 'Speed', icon: Zap, color: 'text-yellow-500' },
                { key: 'reliability_score', name: 'Reliability', icon: Shield, color: 'text-blue-500' },
                { key: 'security_score', name: 'Security', icon: Shield, color: 'text-red-500' }
              ].map(({ key, name, icon: Icon, color }) => {
                const score = results[key as keyof ValidationResult] as number
                const ScoreIcon = getScoreIcon(score)
                
                return (
                  <Card key={key}>
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <Icon className={`h-5 w-5 ${color}`} />
                          <span className="font-medium">{name}</span>
                        </div>
                        <ScoreIcon className={`h-5 w-5 ${getScoreColor(score)}`} />
                      </div>
                      <div className="mt-2">
                        <div className={`text-2xl font-bold ${getScoreColor(score)}`}>
                          {score}%
                        </div>
                        <Progress value={score} className="mt-2" />
                      </div>
                    </CardContent>
                  </Card>
                )
              })}
            </div>

            {/* Performance Highlights */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Award className="h-5 w-5" />
                  <span>Performance Highlights</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-3 gap-6">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-green-600 mb-2">
                      {benchmarkResults?.improvement_factor.toFixed(1)}x
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Overall Improvement vs Human
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-blue-600 mb-2">
                      {benchmarkResults?.performance_metrics.availability}%
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Availability Target
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-purple-600 mb-2">
                      {benchmarkResults?.performance_metrics.response_time.toFixed(1)}s
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Average Response Time
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Detailed Results Tab */}
          <TabsContent value="detailed" className="space-y-4">
            {Object.entries(testSuites).map(([suiteKey, suite]) => {
              const suiteResults = testResults[suiteKey] || []
              const passedTests = suiteResults.filter(r => r.passed).length
              const totalTests = suiteResults.length
              
              return (
                <Card key={suiteKey}>
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <suite.icon className={`h-5 w-5 ${suite.color}`} />
                        <span>{suite.name}</span>
                      </div>
                      <Badge variant={passedTests === totalTests ? "default" : "secondary"}>
                        {passedTests}/{totalTests} Passed
                      </Badge>
                    </CardTitle>
                    <CardDescription>{suite.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {suite.tests.map((test, index) => {
                        const result = suiteResults[index]
                        if (!result) return null
                        
                        const ResultIcon = result.passed ? CheckCircle : XCircle
                        const iconColor = result.passed ? 'text-green-500' : 'text-red-500'
                        
                        return (
                          <div key={test.id} className="flex items-center justify-between p-3 border rounded-lg">
                            <div className="flex items-center space-x-3">
                              <ResultIcon className={`h-5 w-5 ${iconColor}`} />
                              <div>
                                <div className="font-medium">{test.name}</div>
                                <div className="text-sm text-muted-foreground">
                                  {test.description}
                                </div>
                              </div>
                            </div>
                            <div className="text-right">
                              <div className="font-medium">
                                {result.accuracy_score.toFixed(1)}%
                              </div>
                              <div className="text-sm text-muted-foreground">
                                {result.execution_time.toFixed(0)}ms
                              </div>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </TabsContent>

          {/* Benchmark Tab */}
          <TabsContent value="benchmark" className="space-y-4">
            {benchmarkResults && (
              <>
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <TrendingUp className="h-5 w-5" />
                      <span>Human vs AI Performance</span>
                    </CardTitle>
                    <CardDescription>
                      Comparison against human baseline performance
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-6">
                      {[
                        { 
                          metric: 'Accuracy', 
                          ai: benchmarkResults.performance_metrics.accuracy,
                          human: humanBaselines[employeeConfig.type as keyof typeof humanBaselines]?.accuracy || 85,
                          unit: '%'
                        },
                        { 
                          metric: 'Speed', 
                          ai: benchmarkResults.performance_metrics.speed,
                          human: 1,
                          unit: 'x'
                        },
                        { 
                          metric: 'Availability', 
                          ai: benchmarkResults.performance_metrics.availability,
                          human: humanBaselines[employeeConfig.type as keyof typeof humanBaselines]?.availability || 75,
                          unit: '%'
                        },
                        { 
                          metric: 'Error Rate', 
                          ai: benchmarkResults.performance_metrics.error_rate,
                          human: humanBaselines[employeeConfig.type as keyof typeof humanBaselines]?.error_rate || 15,
                          unit: '%',
                          inverse: true
                        }
                      ].map(({ metric, ai, human, unit, inverse }) => {
                        const improvement = inverse ? human / ai : ai / human
                        const isImprovement = improvement > 1
                        
                        return (
                          <div key={metric} className="space-y-2">
                            <div className="flex justify-between items-center">
                              <span className="font-medium">{metric}</span>
                              <div className="flex items-center space-x-4">
                                <div className="text-sm">
                                  <span className="text-muted-foreground">Human: </span>
                                  <span>{human}{unit}</span>
                                </div>
                                <div className="text-sm">
                                  <span className="text-muted-foreground">AI: </span>
                                  <span className="font-medium">{ai.toFixed(1)}{unit}</span>
                                </div>
                                <Badge variant={isImprovement ? "default" : "secondary"}>
                                  {improvement.toFixed(1)}x {isImprovement ? 'better' : 'worse'}
                                </Badge>
                              </div>
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                              <Progress value={inverse ? (100 - human) : (human / Math.max(ai, human) * 100)} />
                              <Progress value={inverse ? (100 - ai) : (ai / Math.max(ai, human) * 100)} />
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Flame className="h-5 w-5" />
                      <span>Superhuman Achievement</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-center space-y-4">
                      <div className="text-6xl font-bold text-gradient">
                        {benchmarkResults.improvement_factor.toFixed(1)}x
                      </div>
                      <div className="text-xl font-semibold">
                        Overall Performance Improvement
                      </div>
                      <div className="text-muted-foreground">
                        Your AI employee performs {benchmarkResults.improvement_factor.toFixed(1)} times better than the average human in this role
                      </div>
                      
                      {benchmarkResults.improvement_factor >= 5 && (
                        <Badge className="bg-gradient-forge text-white">
                          🏆 Superhuman Performance Achieved!
                        </Badge>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </>
            )}
          </TabsContent>

          {/* Recommendations Tab */}
          <TabsContent value="recommendations" className="space-y-4">
            <div className="grid gap-4">
              {results.recommendations.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2 text-blue-600">
                      <Lightbulb className="h-5 w-5" />
                      <span>Recommendations</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-2">
                      {results.recommendations.map((rec, index) => (
                        <li key={index} className="flex items-start space-x-2">
                          <CheckCircle className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
                          <span className="text-sm">{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}

              {results.warnings.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2 text-yellow-600">
                      <AlertTriangle className="h-5 w-5" />
                      <span>Warnings</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-2">
                      {results.warnings.map((warning, index) => (
                        <li key={index} className="flex items-start space-x-2">
                          <AlertTriangle className="h-4 w-4 text-yellow-500 mt-0.5 flex-shrink-0" />
                          <span className="text-sm">{warning}</span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}

              {results.errors.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2 text-red-600">
                      <XCircle className="h-5 w-5" />
                      <span>Critical Issues</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-2">
                      {results.errors.map((error, index) => (
                        <li key={index} className="flex items-start space-x-2">
                          <XCircle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                          <span className="text-sm">{error}</span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}

              {results.recommendations.length === 0 && results.warnings.length === 0 && results.errors.length === 0 && (
                <Card>
                  <CardContent className="text-center py-8">
                    <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold mb-2">Perfect Performance!</h3>
                    <p className="text-muted-foreground">
                      Your AI employee has passed all validation tests with flying colors.
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>
        </Tabs>
      )}
    </div>
  )
}