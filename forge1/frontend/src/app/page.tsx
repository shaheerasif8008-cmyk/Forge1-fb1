'use client'

import { useState, useEffect } from 'react'
import { Bot, Zap, Shield, BarChart3, Users, Workflow, ArrowRight } from 'lucide-react'

export default function HomePage() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-center">Loading...</div>
    </div>
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="border-b bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex h-16 items-center justify-between">
          <div className="flex items-center space-x-2">
            <Bot className="h-8 w-8 text-blue-500" />
            <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Forge 1</span>
          </div>
          
          <div className="flex items-center space-x-4">
            <button className="px-4 py-2 text-gray-600 hover:text-gray-900">
              Sign In
            </button>
            <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
              Get Started
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto text-center">
          <div className="space-y-8">
            <div className="space-y-4">
              <div className="inline-flex items-center px-4 py-2 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                🚀 Now Available for Enterprise
              </div>
              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-gray-900">
                Build{' '}
                <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Superhuman</span>
                <br />
                AI Employees
              </h1>
              <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                Create specialized AI employees that deliver 5x-50x performance improvements. 
                Enterprise-grade security, real-time analytics, and seamless integration.
              </p>
            </div>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button className="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium flex items-center justify-center">
                Start Building AI Employees
                <ArrowRight className="ml-2 h-4 w-4" />
              </button>
              <button className="px-8 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 font-medium">
                Watch Demo
              </button>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-8 pt-12">
              <div className="text-center">
                <div className="flex justify-center mb-2">
                  <BarChart3 className="h-8 w-8 text-blue-500" />
                </div>
                <div className="text-2xl font-bold text-gray-900">5x-50x</div>
                <div className="text-sm text-gray-600">Performance Improvement</div>
              </div>
              <div className="text-center">
                <div className="flex justify-center mb-2">
                  <Shield className="h-8 w-8 text-green-500" />
                </div>
                <div className="text-2xl font-bold text-gray-900">99.9%+</div>
                <div className="text-sm text-gray-600">Accuracy Rate</div>
              </div>
              <div className="text-center">
                <div className="flex justify-center mb-2">
                  <Users className="h-8 w-8 text-purple-500" />
                </div>
                <div className="text-2xl font-bold text-gray-900">500+</div>
                <div className="text-sm text-gray-600">Enterprise Clients</div>
              </div>
              <div className="text-center">
                <div className="flex justify-center mb-2">
                  <Bot className="h-8 w-8 text-blue-500" />
                </div>
                <div className="text-2xl font-bold text-gray-900">10,000+</div>
                <div className="text-sm text-gray-600">AI Employees Created</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-white">
        <div className="max-w-7xl mx-auto">
          <div className="text-center space-y-4 mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-gray-900">
              Everything You Need to Build AI Employees
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              From creation to deployment, monitor and optimize your AI workforce with enterprise-grade tools.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="p-6 bg-gray-50 rounded-xl hover:shadow-lg transition-shadow">
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 rounded-lg bg-white text-blue-500">
                  <Bot className="h-6 w-6" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900">AI Employee Builder</h3>
              </div>
              <p className="text-gray-600">
                Create specialized AI employees with drag-and-drop simplicity
              </p>
            </div>

            <div className="p-6 bg-gray-50 rounded-xl hover:shadow-lg transition-shadow">
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 rounded-lg bg-white text-yellow-500">
                  <Zap className="h-6 w-6" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900">Superhuman Performance</h3>
              </div>
              <p className="text-gray-600">
                5x-50x performance improvements over human employees
              </p>
            </div>

            <div className="p-6 bg-gray-50 rounded-xl hover:shadow-lg transition-shadow">
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 rounded-lg bg-white text-green-500">
                  <Shield className="h-6 w-6" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900">Enterprise Security</h3>
              </div>
              <p className="text-gray-600">
                Bank-grade security with GDPR, HIPAA, and SOX compliance
              </p>
            </div>

            <div className="p-6 bg-gray-50 rounded-xl hover:shadow-lg transition-shadow">
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 rounded-lg bg-white text-purple-500">
                  <BarChart3 className="h-6 w-6" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900">Real-time Analytics</h3>
              </div>
              <p className="text-gray-600">
                Monitor performance, ROI, and productivity metrics
              </p>
            </div>

            <div className="p-6 bg-gray-50 rounded-xl hover:shadow-lg transition-shadow">
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 rounded-lg bg-white text-indigo-500">
                  <Users className="h-6 w-6" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900">Multi-Agent Coordination</h3>
              </div>
              <p className="text-gray-600">
                Teams of AI employees working together seamlessly
              </p>
            </div>

            <div className="p-6 bg-gray-50 rounded-xl hover:shadow-lg transition-shadow">
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 rounded-lg bg-white text-orange-500">
                  <Workflow className="h-6 w-6" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900">Automation Integration</h3>
              </div>
              <p className="text-gray-600">
                Connect with Zapier, n8n, and custom workflows
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Status Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gray-50">
        <div className="max-w-7xl mx-auto text-center">
          <div className="space-y-8">
            <h2 className="text-3xl sm:text-4xl font-bold text-gray-900">
              Platform Status: 100% Operational
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div className="bg-white p-6 rounded-lg shadow-sm">
                <div className="text-2xl font-bold text-green-600 mb-2">✅ Backend API</div>
                <div className="text-gray-600">All systems operational</div>
              </div>
              <div className="bg-white p-6 rounded-lg shadow-sm">
                <div className="text-2xl font-bold text-green-600 mb-2">✅ Database</div>
                <div className="text-gray-600">PostgreSQL & Redis running</div>
              </div>
              <div className="bg-white p-6 rounded-lg shadow-sm">
                <div className="text-2xl font-bold text-green-600 mb-2">✅ Monitoring</div>
                <div className="text-gray-600">Prometheus & Grafana active</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t bg-white py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto text-center">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <Bot className="h-6 w-6 text-blue-500" />
            <span className="font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Forge 1</span>
          </div>
          <p className="text-sm text-gray-600">
            © 2024 Forge 1. Building the future of work with superhuman AI employees.
          </p>
        </div>
      </footer>
    </div>
  )
}