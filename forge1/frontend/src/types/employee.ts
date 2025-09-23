export interface Skill {
  id: string
  name: string
  description: string
  proficiency: number // 0-100
  category: string
  required: boolean
}

export interface Personality {
  [key: string]: number // All personality traits are 0-100 values
}

export interface PerformanceTargets {
  accuracy: number // 0-100
  speed: number // 1-50x multiplier
  availability: number // 0-100
}

export interface DeploymentConfig {
  environment: 'development' | 'staging' | 'production'
  scaling: 'manual' | 'auto'
  monitoring: boolean
  region?: string
  resources?: {
    cpu: string
    memory: string
    storage: string
  }
}

export interface Integration {
  id: string
  name: string
  type: 'api' | 'webhook' | 'database' | 'file' | 'email'
  config: Record<string, any>
  enabled: boolean
}

export interface Capability {
  id: string
  name: string
  description: string
  type: 'input' | 'output' | 'processing'
  config: Record<string, any>
}

export interface WorkflowStep {
  id: string
  name: string
  type: 'trigger' | 'action' | 'condition' | 'loop'
  config: Record<string, any>
  position: { x: number; y: number }
  connections: string[] // IDs of connected steps
}

export interface Workflow {
  id: string
  name: string
  description: string
  steps: WorkflowStep[]
  triggers: string[] // IDs of trigger steps
  enabled: boolean
}

export interface EmployeeConfig {
  id: string
  name: string
  description: string
  type: string
  skills: Skill[]
  personality: Personality
  capabilities: Capability[]
  integrations: Integration[]
  workflows?: Workflow[]
  performance_targets: PerformanceTargets
  security_level: 'basic' | 'standard' | 'high' | 'enterprise'
  deployment_config: DeploymentConfig
  created_at?: string
  updated_at?: string
  status?: 'draft' | 'testing' | 'deployed' | 'archived'
}

export interface EmployeeTemplate {
  id: string
  name: string
  description: string
  category: string
  icon?: string
  skills?: Skill[]
  personality?: Personality
  capabilities?: Capability[]
  integrations?: Integration[]
  performance_targets?: PerformanceTargets
  tags: string[]
  popularity: number
  created_by: string
  created_at: string
}

export interface PerformanceMetrics {
  accuracy: number
  speed: number
  availability: number
  customer_satisfaction?: number
  error_rate: number
  throughput: number
  response_time: number
  cost_per_task: number
}

export interface ValidationResult {
  overall_score: number
  accuracy_score: number
  speed_score: number
  reliability_score: number
  security_score: number
  compliance_score: number
  recommendations: string[]
  warnings: string[]
  errors: string[]
}

export interface TestCase {
  id: string
  name: string
  description: string
  input: Record<string, any>
  expected_output: Record<string, any>
  category: string
  difficulty: 'easy' | 'medium' | 'hard'
  tags: string[]
}

export interface TestResult {
  test_case_id: string
  passed: boolean
  actual_output: Record<string, any>
  execution_time: number
  accuracy_score: number
  error_message?: string
}

export interface BenchmarkResult {
  employee_id: string
  test_suite_id: string
  overall_score: number
  individual_scores: TestResult[]
  performance_metrics: PerformanceMetrics
  comparison_baseline: 'human' | 'previous_version' | 'industry_standard'
  improvement_factor: number // e.g., 5.2 for 5.2x improvement
  executed_at: string
}