'use client'

import { useState, useMemo } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
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
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts'
import { 
  DollarSign, 
  TrendingUp, 
  Calculator, 
  Target,
  Clock,
  Users,
  Zap,
  Award,
  Download,
  RefreshCw,
  Info
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

interface ROICalculatorProps {
  metrics: DashboardMetrics
  performanceData: PerformanceData[]
}

interface ROICalculation {
  initialInvestment: number
  monthlyCosts: number
  humanEquivalentCost: number
  monthlySavings: number
  annualSavings: number
  paybackPeriod: number
  roi: number
  netPresentValue: number
}

export function ROICalculator({ metrics, performanceData }: ROICalculatorProps) {
  const [timeframe, setTimeframe] = useState('12')
  const [humanSalary, setHumanSalary] = useState(65000)
  const [aiCostPerEmployee, setAiCostPerEmployee] = useState(2400)
  const [initialSetupCost, setInitialSetupCost] = useState(50000)
  const [discountRate, setDiscountRate] = useState(8)

  const roiCalculation = useMemo((): ROICalculation => {
    const monthlyHumanCost = (humanSalary * metrics.totalEmployees) / 12
    const monthlyAICost = (aiCostPerEmployee * metrics.totalEmployees) / 12
    const monthlySavings = monthlyHumanCost - monthlyAICost
    const annualSavings = monthlySavings * 12
    const paybackPeriod = initialSetupCost / monthlySavings
    const totalSavings = annualSavings * parseInt(timeframe)
    const roi = ((totalSavings - initialSetupCost) / initialSetupCost) * 100
    
    // NPV calculation
    let npv = -initialSetupCost
    for (let year = 1; year <= parseInt(timeframe); year++) {
      npv += annualSavings / Math.pow(1 + discountRate / 100, year)
    }

    return {
      initialInvestment: initialSetupCost,
      monthlyCosts: monthlyAICost,
      humanEquivalentCost: monthlyHumanCost,
      monthlySavings,
      annualSavings,
      paybackPeriod,
      roi,
      netPresentValue: npv
    }
  }, [humanSalary, aiCostPerEmployee, initialSetupCost, discountRate, timeframe, metrics.totalEmployees])

  // Generate projection data
  const projectionData = useMemo(() => {
    const data = []
    const months = parseInt(timeframe)
    
    for (let month = 0; month <= months; month++) {
      const cumulativeSavings = (roiCalculation.monthlySavings * month) - roiCalculation.initialInvestment
      const humanCost = roiCalculation.humanEquivalentCost * month
      const aiCost = roiCalculation.initialInvestment + (roiCalculation.monthlyCosts * month)
      
      data.push({
        month,
        cumulativeSavings,
        humanCost,
        aiCost,
        netSavings: Math.max(0, cumulativeSavings)
      })
    }
    
    return data
  }, [roiCalculation, timeframe])

  const costBreakdown = [
    { name: 'AI Implementation', value: roiCalculation.monthlyCosts * 12, color: '#3b82f6' },
    { name: 'Human Equivalent', value: roiCalculation.humanEquivalentCost * 12, color: '#6b7280' }
  ]

  const savingsBreakdown = [
    { name: 'Labor Cost Savings', value: roiCalculation.annualSavings * 0.7, color: '#10b981' },
    { name: 'Efficiency Gains', value: roiCalculation.annualSavings * 0.2, color: '#f59e0b' },
    { name: 'Error Reduction', value: roiCalculation.annualSavings * 0.1, color: '#8b5cf6' }
  ]

  const performanceImpact = [
    { metric: 'Speed Improvement', value: metrics.averageSpeed, unit: 'x', color: '#f59e0b' },
    { metric: 'Accuracy Rate', value: metrics.averageAccuracy, unit: '%', color: '#10b981' },
    { metric: 'Uptime', value: metrics.uptime, unit: '%', color: '#3b82f6' },
    { metric: 'Error Reduction', value: 100 - metrics.errorRate, unit: '%', color: '#8b5cf6' }
  ]

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-background border rounded-lg shadow-lg p-3">
          <p className="font-medium">Month {label}</p>
          <div className="space-y-1 mt-2">
            {payload.map((entry: any, index: number) => (
              <div key={index} className="flex items-center justify-between space-x-4">
                <span className="text-sm" style={{ color: entry.color }}>
                  {entry.name}:
                </span>
                <span className="font-medium">
                  ${Math.round(entry.value).toLocaleString()}
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
      {/* ROI Summary Cards */}
      <div className="grid md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total ROI</p>
                <p className="text-2xl font-bold text-green-600">
                  {roiCalculation.roi.toFixed(0)}%
                </p>
              </div>
              <TrendingUp className="h-8 w-8 text-green-500" />
            </div>
            <div className="mt-2">
              <Badge className="bg-green-100 text-green-800">
                {parseInt(timeframe)} year projection
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Annual Savings</p>
                <p className="text-2xl font-bold text-blue-600">
                  ${(roiCalculation.annualSavings / 1000).toFixed(0)}K
                </p>
              </div>
              <DollarSign className="h-8 w-8 text-blue-500" />
            </div>
            <div className="mt-2">
              <Badge className="bg-blue-100 text-blue-800">
                vs Human workforce
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Payback Period</p>
                <p className="text-2xl font-bold text-purple-600">
                  {roiCalculation.paybackPeriod.toFixed(1)}
                </p>
              </div>
              <Clock className="h-8 w-8 text-purple-500" />
            </div>
            <div className="mt-2">
              <Badge className="bg-purple-100 text-purple-800">
                months
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Net Present Value</p>
                <p className="text-2xl font-bold text-orange-600">
                  ${(roiCalculation.netPresentValue / 1000).toFixed(0)}K
                </p>
              </div>
              <Calculator className="h-8 w-8 text-orange-500" />
            </div>
            <div className="mt-2">
              <Badge className="bg-orange-100 text-orange-800">
                {discountRate}% discount rate
              </Badge>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="projection" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="projection">Cost Projection</TabsTrigger>
          <TabsTrigger value="breakdown">Cost Breakdown</TabsTrigger>
          <TabsTrigger value="calculator">ROI Calculator</TabsTrigger>
          <TabsTrigger value="impact">Business Impact</TabsTrigger>
        </TabsList>

        {/* Cost Projection Tab */}
        <TabsContent value="projection">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <TrendingUp className="h-5 w-5 text-green-500" />
                <span>Cost Savings Projection</span>
              </CardTitle>
              <CardDescription>
                Cumulative savings over time compared to human workforce
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80 mb-6">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={projectionData}>
                    <defs>
                      <linearGradient id="savingsGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                    <XAxis 
                      dataKey="month" 
                      tick={{ fontSize: 12 }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <YAxis 
                      tick={{ fontSize: 12 }}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Area
                      type="monotone"
                      dataKey="cumulativeSavings"
                      stroke="#10b981"
                      strokeWidth={2}
                      fill="url(#savingsGradient)"
                      name="Cumulative Savings"
                    />
                    <Line
                      type="monotone"
                      dataKey="humanCost"
                      stroke="#6b7280"
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      name="Human Cost"
                    />
                    <Line
                      type="monotone"
                      dataKey="aiCost"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      name="AI Cost"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              
              <div className="grid md:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">
                    ${(roiCalculation.annualSavings * parseInt(timeframe) / 1000).toFixed(0)}K
                  </div>
                  <div className="text-sm text-muted-foreground">Total Savings</div>
                </div>
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">
                    {roiCalculation.paybackPeriod.toFixed(1)}
                  </div>
                  <div className="text-sm text-muted-foreground">Months to Break Even</div>
                </div>
                <div className="text-center p-4 bg-purple-50 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600">
                    {((roiCalculation.monthlySavings / roiCalculation.humanEquivalentCost) * 100).toFixed(0)}%
                  </div>
                  <div className="text-sm text-muted-foreground">Monthly Cost Reduction</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Cost Breakdown Tab */}
        <TabsContent value="breakdown">
          <div className="grid lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Annual Cost Comparison</CardTitle>
                <CardDescription>AI implementation vs human workforce</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={costBreakdown}>
                      <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                      <XAxis 
                        dataKey="name" 
                        tick={{ fontSize: 12 }}
                        tickLine={false}
                        axisLine={false}
                      />
                      <YAxis 
                        tick={{ fontSize: 12 }}
                        tickLine={false}
                        axisLine={false}
                        tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`}
                      />
                      <Tooltip 
                        formatter={(value: number) => [`$${value.toLocaleString()}`, 'Annual Cost']}
                      />
                      <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Savings Sources</CardTitle>
                <CardDescription>Where the cost savings come from</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex items-center space-x-8">
                  <div className="h-48 w-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={savingsBreakdown}
                          cx="50%"
                          cy="50%"
                          innerRadius={40}
                          outerRadius={80}
                          paddingAngle={5}
                          dataKey="value"
                        >
                          {savingsBreakdown.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip 
                          formatter={(value: number) => [`$${(value / 1000).toFixed(0)}K`, 'Annual Savings']}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  
                  <div className="space-y-3">
                    {savingsBreakdown.map((item, index) => (
                      <div key={index} className="flex items-center space-x-3">
                        <div 
                          className="w-4 h-4 rounded-full" 
                          style={{ backgroundColor: item.color }}
                        />
                        <div>
                          <div className="font-medium text-sm">{item.name}</div>
                          <div className="text-sm text-muted-foreground">
                            ${(item.value / 1000).toFixed(0)}K annually
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* ROI Calculator Tab */}
        <TabsContent value="calculator">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Calculator className="h-5 w-5 text-blue-500" />
                <span>ROI Calculator</span>
              </CardTitle>
              <CardDescription>
                Adjust parameters to see how they affect your ROI
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid lg:grid-cols-2 gap-8">
                <div className="space-y-6">
                  <div className="space-y-4">
                    <div>
                      <Label htmlFor="humanSalary">Average Human Salary (Annual)</Label>
                      <Input
                        id="humanSalary"
                        type="number"
                        value={humanSalary}
                        onChange={(e) => setHumanSalary(Number(e.target.value))}
                        className="mt-1"
                      />
                    </div>
                    
                    <div>
                      <Label htmlFor="aiCost">AI Employee Cost (Annual)</Label>
                      <Input
                        id="aiCost"
                        type="number"
                        value={aiCostPerEmployee}
                        onChange={(e) => setAiCostPerEmployee(Number(e.target.value))}
                        className="mt-1"
                      />
                    </div>
                    
                    <div>
                      <Label htmlFor="setupCost">Initial Setup Cost</Label>
                      <Input
                        id="setupCost"
                        type="number"
                        value={initialSetupCost}
                        onChange={(e) => setInitialSetupCost(Number(e.target.value))}
                        className="mt-1"
                      />
                    </div>
                    
                    <div>
                      <Label htmlFor="discountRate">Discount Rate (%)</Label>
                      <Input
                        id="discountRate"
                        type="number"
                        value={discountRate}
                        onChange={(e) => setDiscountRate(Number(e.target.value))}
                        className="mt-1"
                      />
                    </div>
                    
                    <div>
                      <Label htmlFor="timeframe">Projection Timeframe</Label>
                      <Select value={timeframe} onValueChange={setTimeframe}>
                        <SelectTrigger className="mt-1">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="1">1 Year</SelectItem>
                          <SelectItem value="3">3 Years</SelectItem>
                          <SelectItem value="5">5 Years</SelectItem>
                          <SelectItem value="10">10 Years</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <h4 className="font-semibold mb-3">Calculated Results</h4>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Monthly Savings:</span>
                        <span className="font-medium">${roiCalculation.monthlySavings.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Annual Savings:</span>
                        <span className="font-medium">${roiCalculation.annualSavings.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Payback Period:</span>
                        <span className="font-medium">{roiCalculation.paybackPeriod.toFixed(1)} months</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Total ROI:</span>
                        <span className="font-medium text-green-600">{roiCalculation.roi.toFixed(0)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Net Present Value:</span>
                        <span className="font-medium">${roiCalculation.netPresentValue.toLocaleString()}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex space-x-2">
                    <Button className="flex-1">
                      <Download className="h-4 w-4 mr-2" />
                      Export Report
                    </Button>
                    <Button variant="outline">
                      <RefreshCw className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Business Impact Tab */}
        <TabsContent value="impact">
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Award className="h-5 w-5 text-orange-500" />
                  <span>Business Impact Metrics</span>
                </CardTitle>
                <CardDescription>
                  Quantified benefits beyond direct cost savings
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-4 gap-4">
                  {performanceImpact.map((item, index) => (
                    <div key={index} className="text-center p-4 rounded-lg border">
                      <div className="text-2xl font-bold mb-2" style={{ color: item.color }}>
                        {item.value.toFixed(1)}{item.unit}
                      </div>
                      <div className="text-sm text-muted-foreground">{item.metric}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
            
            <div className="grid lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Productivity Gains</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Tasks per hour increase</span>
                    <span className="font-semibold text-green-600">+{((metrics.averageSpeed - 1) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Error rate reduction</span>
                    <span className="font-semibold text-green-600">-{(100 - metrics.errorRate).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">24/7 availability</span>
                    <span className="font-semibold text-blue-600">{metrics.uptime}% uptime</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Customer satisfaction</span>
                    <span className="font-semibold text-purple-600">{metrics.customerSatisfaction}%</span>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle>Strategic Benefits</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-start space-x-3">
                    <Info className="h-5 w-5 text-blue-500 mt-0.5" />
                    <div>
                      <div className="font-medium">Scalability</div>
                      <div className="text-sm text-muted-foreground">
                        Instant scaling without recruitment delays
                      </div>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <Info className="h-5 w-5 text-green-500 mt-0.5" />
                    <div>
                      <div className="font-medium">Consistency</div>
                      <div className="text-sm text-muted-foreground">
                        Uniform quality across all tasks
                      </div>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <Info className="h-5 w-5 text-purple-500 mt-0.5" />
                    <div>
                      <div className="font-medium">Innovation</div>
                      <div className="text-sm text-muted-foreground">
                        Free human resources for strategic work
                      </div>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <Info className="h-5 w-5 text-orange-500 mt-0.5" />
                    <div>
                      <div className="font-medium">Competitive Advantage</div>
                      <div className="text-sm text-muted-foreground">
                        Faster response times and lower costs
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}