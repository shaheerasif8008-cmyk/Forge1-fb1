'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
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
  PieChart,
  Pie,
  Cell,
  Area,
  AreaChart
} from 'recharts'
import { 
  DollarSign, 
  TrendingUp, 
  Calculator, 
  Download, 
  Calendar,
  Users,
  Clock,
  Target,
  Zap,
  FileText,
  BarChart3
} from 'lucide-react'

const roiData = {
  summary: {
    totalInvestment: 485000,
    totalSavings: 2847392,
    netROI: 2362392,
    roiPercentage: 487.2,
    paybackPeriod: 2.3, // months
    breakEvenDate: '2024-03-15'
  },
  monthlyData: [
    { month: 'Jan', investment: 120000, savings: 45000, netROI: -75000, cumulative: -75000 },
    { month: 'Feb', investment: 85000, savings: 125000, netROI: 40000, cumulative: -35000 },
    { month: 'Mar', investment: 65000, savings: 185000, netROI: 120000, cumulative: 85000 },
    { month: 'Apr', investment: 45000, savings: 245000, netROI: 200000, cumulative: 285000 },
    { month: 'May', investment: 35000, savings: 285000, netROI: 250000, cumulative: 535000 },
    { month: 'Jun', investment: 25000, savings: 325000, netROI: 300000, cumulative: 835000 },
    { month: 'Jul', investment: 20000, savings: 365000, netROI: 345000, cumulative: 1180000 },
    { month: 'Aug', investment: 15000, savings: 385000, netROI: 370000, cumulative: 1550000 },
    { month: 'Sep', investment: 15000, savings: 425000, netROI: 410000, cumulative: 1960000 },
    { month: 'Oct', investment: 15000, savings: 445000, netROI: 430000, cumulative: 2390000 },
    { month: 'Nov', investment: 15000, savings: 465000, netROI: 450000, cumulative: 2840000 },
    { month: 'Dec', investment: 15000, savings: 485000, netROI: 470000, cumulative: 3310000 }
  ],
  costBreakdown: [
    { category: 'Human Salaries', amount: 1850000, color: '#ef4444' },
    { category: 'Benefits & Overhead', amount: 555000, color: '#f97316' },
    { category: 'Training & Onboarding', amount: 185000, color: '#eab308' },
    { category: 'Office Space & Equipment', amount: 125000, color: '#22c55e' },
    { category: 'Management Overhead', amount: 92000, color: '#3b82f6' },
    { category: 'Recruitment Costs', amount: 40392, color: '#8b5cf6' }
  ],
  aiCosts: [
    { category: 'Platform License', amount: 180000, color: '#0ea5e9' },
    { category: 'Compute Resources', amount: 125000, color: '#10b981' },
    { category: 'Implementation', amount: 85000, color: '#f59e0b' },
    { category: 'Training & Setup', amount: 45000, color: '#ef4444' },
    { category: 'Maintenance', amount: 35000, color: '#8b5cf6' },
    { category: 'Support', amount: 15000, color: '#6b7280' }
  ],
  employeeComparison: [
    {
      role: 'Customer Service Agent',
      humanCost: 65000,
      aiCost: 8500,
      savings: 56500,
      efficiency: 12.4,
      tasks: 15420
    },
    {
      role: 'Data Analyst',
      humanCost: 85000,
      aiCost: 12000,
      savings: 73000,
      efficiency: 8.7,
      tasks: 8920
    },
    {
      role: 'Content Writer',
      humanCost: 55000,
      aiCost: 7500,
      savings: 47500,
      efficiency: 6.2,
      tasks: 6340
    },
    {
      role: 'Sales Assistant',
      humanCost: 48000,
      aiCost: 6800,
      savings: 41200,
      efficiency: 9.1,
      tasks: 4450
    }
  ]
}

export function ROICalculatorWidget() {
  const [selectedPeriod, setSelectedPeriod] = useState('12m')
  const [customCalculation, setCustomCalculation] = useState({
    humanEmployees: 25,
    avgSalary: 65000,
    aiEmployees: 47,
    aiCostPerEmployee: 8500
  })

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount)
  }

  const formatPercentage = (value: number) => {
    return `${value.toFixed(1)}%`
  }

  const calculateCustomROI = () => {
    const humanCost = customCalculation.humanEmployees * customCalculation.avgSalary
    const aiCost = customCalculation.aiEmployees * customCalculation.aiCostPerEmployee
    const savings = humanCost - aiCost
    const roi = ((savings / aiCost) * 100)
    
    return {
      humanCost,
      aiCost,
      savings,
      roi
    }
  }

  const customROI = calculateCustomROI()

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Calculator className="h-5 w-5" />
          <span>ROI Analysis</span>
        </CardTitle>
        <CardDescription>
          Comprehensive return on investment analysis and cost-benefit calculations
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="breakdown">Cost Breakdown</TabsTrigger>
            <TabsTrigger value="comparison">Comparison</TabsTrigger>
            <TabsTrigger value="calculator">Calculator</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            {/* Key ROI Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-green-50 rounded-lg">
                <div className="text-2xl font-bold text-green-600">
                  {formatPercentage(roiData.summary.roiPercentage)}
                </div>
                <div className="text-sm text-muted-foreground">Total ROI</div>
              </div>
              
              <div className="text-center p-4 bg-blue-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">
                  {formatCurrency(roiData.summary.netROI)}
                </div>
                <div className="text-sm text-muted-foreground">Net Savings</div>
              </div>
              
              <div className="text-center p-4 bg-purple-50 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">
                  {roiData.summary.paybackPeriod}
                </div>
                <div className="text-sm text-muted-foreground">Payback (Months)</div>
              </div>
              
              <div className="text-center p-4 bg-orange-50 rounded-lg">
                <div className="text-2xl font-bold text-orange-600">
                  Mar 2024
                </div>
                <div className="text-sm text-muted-foreground">Break-even Date</div>
              </div>
            </div>

            {/* ROI Trend Chart */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">ROI Trend Analysis</h3>
                <Select value={selectedPeriod} onValueChange={setSelectedPeriod}>
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="6m">6 Months</SelectItem>
                    <SelectItem value="12m">12 Months</SelectItem>
                    <SelectItem value="24m">24 Months</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={roiData.monthlyData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis tickFormatter={(value) => formatCurrency(value)} />
                    <Tooltip 
                      formatter={(value, name) => [formatCurrency(value as number), name]}
                      labelFormatter={(label) => `Month: ${label}`}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="cumulative" 
                      stroke="#10b981" 
                      fill="#10b981" 
                      fillOpacity={0.3}
                      name="Cumulative ROI"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="netROI" 
                      stroke="#0ea5e9" 
                      strokeWidth={2}
                      name="Monthly ROI"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Investment vs Savings */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Monthly Investment vs Savings</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={roiData.monthlyData.slice(-6)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis tickFormatter={(value) => formatCurrency(value)} />
                      <Tooltip formatter={(value) => formatCurrency(value as number)} />
                      <Bar dataKey="investment" fill="#ef4444" name="Investment" />
                      <Bar dataKey="savings" fill="#10b981" name="Savings" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="text-lg font-semibold">ROI Milestones</h3>
                <div className="space-y-3">
                  {[
                    { milestone: 'Break-even Point', date: 'March 2024', status: 'achieved' },
                    { milestone: '200% ROI', date: 'June 2024', status: 'achieved' },
                    { milestone: '400% ROI', date: 'October 2024', status: 'achieved' },
                    { milestone: '500% ROI', date: 'December 2024', status: 'projected' }
                  ].map((item, index) => (
                    <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <div className="font-medium">{item.milestone}</div>
                        <div className="text-sm text-muted-foreground">{item.date}</div>
                      </div>
                      <Badge variant={item.status === 'achieved' ? 'default' : 'secondary'}>
                        {item.status}
                      </Badge>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </TabsContent>

          {/* Cost Breakdown Tab */}
          <TabsContent value="breakdown" className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              {/* Human Employee Costs */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Human Employee Costs (Annual)</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={roiData.costBreakdown}
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        dataKey="amount"
                        label={({ name, value }) => `${name}: ${formatCurrency(value)}`}
                      >
                        {roiData.costBreakdown.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value) => formatCurrency(value as number)} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="space-y-2">
                  {roiData.costBreakdown.map((item, index) => (
                    <div key={index} className="flex items-center justify-between text-sm">
                      <div className="flex items-center space-x-2">
                        <div 
                          className="w-3 h-3 rounded-full" 
                          style={{ backgroundColor: item.color }}
                        />
                        <span>{item.category}</span>
                      </div>
                      <span className="font-medium">{formatCurrency(item.amount)}</span>
                    </div>
                  ))}
                  <div className="border-t pt-2 flex justify-between font-semibold">
                    <span>Total Annual Cost</span>
                    <span>{formatCurrency(roiData.costBreakdown.reduce((sum, item) => sum + item.amount, 0))}</span>
                  </div>
                </div>
              </div>

              {/* AI Employee Costs */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">AI Employee Costs (Annual)</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={roiData.aiCosts}
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        dataKey="amount"
                        label={({ name, value }) => `${name}: ${formatCurrency(value)}`}
                      >
                        {roiData.aiCosts.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value) => formatCurrency(value as number)} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="space-y-2">
                  {roiData.aiCosts.map((item, index) => (
                    <div key={index} className="flex items-center justify-between text-sm">
                      <div className="flex items-center space-x-2">
                        <div 
                          className="w-3 h-3 rounded-full" 
                          style={{ backgroundColor: item.color }}
                        />
                        <span>{item.category}</span>
                      </div>
                      <span className="font-medium">{formatCurrency(item.amount)}</span>
                    </div>
                  ))}
                  <div className="border-t pt-2 flex justify-between font-semibold">
                    <span>Total Annual Cost</span>
                    <span>{formatCurrency(roiData.aiCosts.reduce((sum, item) => sum + item.amount, 0))}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Cost Comparison Summary */}
            <div className="p-6 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
                <div>
                  <div className="text-2xl font-bold text-red-600">
                    {formatCurrency(roiData.costBreakdown.reduce((sum, item) => sum + item.amount, 0))}
                  </div>
                  <div className="text-sm text-muted-foreground">Human Employee Costs</div>
                </div>
                
                <div>
                  <div className="text-2xl font-bold text-blue-600">
                    {formatCurrency(roiData.aiCosts.reduce((sum, item) => sum + item.amount, 0))}
                  </div>
                  <div className="text-sm text-muted-foreground">AI Employee Costs</div>
                </div>
                
                <div>
                  <div className="text-2xl font-bold text-green-600">
                    {formatCurrency(
                      roiData.costBreakdown.reduce((sum, item) => sum + item.amount, 0) - 
                      roiData.aiCosts.reduce((sum, item) => sum + item.amount, 0)
                    )}
                  </div>
                  <div className="text-sm text-muted-foreground">Annual Savings</div>
                </div>
              </div>
            </div>
          </TabsContent>

          {/* Comparison Tab */}
          <TabsContent value="comparison" className="space-y-6">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Role-by-Role Cost Comparison</h3>
              
              <div className="space-y-4">
                {roiData.employeeComparison.map((role, index) => (
                  <div key={index} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-4">
                      <h4 className="font-semibold">{role.role}</h4>
                      <Badge variant="default" className="bg-green-600">
                        {formatCurrency(role.savings)} saved/year
                      </Badge>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <div className="text-muted-foreground">Human Cost</div>
                        <div className="font-medium text-red-600">{formatCurrency(role.humanCost)}</div>
                      </div>
                      
                      <div>
                        <div className="text-muted-foreground">AI Cost</div>
                        <div className="font-medium text-blue-600">{formatCurrency(role.aiCost)}</div>
                      </div>
                      
                      <div>
                        <div className="text-muted-foreground">Efficiency</div>
                        <div className="font-medium text-green-600">{role.efficiency}x faster</div>
                      </div>
                      
                      <div>
                        <div className="text-muted-foreground">Tasks/Year</div>
                        <div className="font-medium">{role.tasks.toLocaleString()}</div>
                      </div>
                    </div>
                    
                    <div className="mt-3">
                      <div className="flex justify-between text-xs mb-1">
                        <span>Cost Comparison</span>
                        <span>{((1 - role.aiCost / role.humanCost) * 100).toFixed(1)}% savings</span>
                      </div>
                      <div className="flex h-2 bg-gray-200 rounded">
                        <div 
                          className="bg-red-500 rounded-l"
                          style={{ width: `${(role.humanCost / (role.humanCost + role.aiCost)) * 100}%` }}
                        />
                        <div 
                          className="bg-blue-500 rounded-r"
                          style={{ width: `${(role.aiCost / (role.humanCost + role.aiCost)) * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </TabsContent>

          {/* Calculator Tab */}
          <TabsContent value="calculator" className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Custom ROI Calculator</h3>
                
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="human-employees">Human Employees</Label>
                      <Input
                        id="human-employees"
                        type="number"
                        value={customCalculation.humanEmployees}
                        onChange={(e) => setCustomCalculation(prev => ({
                          ...prev,
                          humanEmployees: parseInt(e.target.value) || 0
                        }))}
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="avg-salary">Average Salary</Label>
                      <Input
                        id="avg-salary"
                        type="number"
                        value={customCalculation.avgSalary}
                        onChange={(e) => setCustomCalculation(prev => ({
                          ...prev,
                          avgSalary: parseInt(e.target.value) || 0
                        }))}
                      />
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="ai-employees">AI Employees</Label>
                      <Input
                        id="ai-employees"
                        type="number"
                        value={customCalculation.aiEmployees}
                        onChange={(e) => setCustomCalculation(prev => ({
                          ...prev,
                          aiEmployees: parseInt(e.target.value) || 0
                        }))}
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="ai-cost">AI Cost per Employee</Label>
                      <Input
                        id="ai-cost"
                        type="number"
                        value={customCalculation.aiCostPerEmployee}
                        onChange={(e) => setCustomCalculation(prev => ({
                          ...prev,
                          aiCostPerEmployee: parseInt(e.target.value) || 0
                        }))}
                      />
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Calculated Results</h3>
                
                <div className="space-y-4">
                  <div className="p-4 bg-red-50 rounded-lg">
                    <div className="text-sm text-muted-foreground">Human Employee Costs</div>
                    <div className="text-2xl font-bold text-red-600">
                      {formatCurrency(customROI.humanCost)}
                    </div>
                  </div>
                  
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <div className="text-sm text-muted-foreground">AI Employee Costs</div>
                    <div className="text-2xl font-bold text-blue-600">
                      {formatCurrency(customROI.aiCost)}
                    </div>
                  </div>
                  
                  <div className="p-4 bg-green-50 rounded-lg">
                    <div className="text-sm text-muted-foreground">Annual Savings</div>
                    <div className="text-2xl font-bold text-green-600">
                      {formatCurrency(customROI.savings)}
                    </div>
                  </div>
                  
                  <div className="p-4 bg-purple-50 rounded-lg">
                    <div className="text-sm text-muted-foreground">ROI Percentage</div>
                    <div className="text-2xl font-bold text-purple-600">
                      {formatPercentage(customROI.roi)}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
        
        {/* Export Button */}
        <div className="flex justify-end pt-4 border-t">
          <Button>
            <Download className="h-4 w-4 mr-2" />
            Export ROI Report
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}