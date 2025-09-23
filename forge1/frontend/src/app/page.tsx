'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { 
  ArrowRight, 
  Bot, 
  Zap, 
  Shield, 
  BarChart3, 
  Users, 
  Workflow,
  CheckCircle,
  Star,
  TrendingUp
} from 'lucide-react'
import Link from 'next/link'
import { useAuth } from '@/hooks/use-auth'

const features = [
  {
    icon: Bot,
    title: 'AI Employee Builder',
    description: 'Create specialized AI employees with drag-and-drop simplicity',
    color: 'text-forge-500'
  },
  {
    icon: Zap,
    title: 'Superhuman Performance',
    description: '5x-50x performance improvements over human employees',
    color: 'text-warning-500'
  },
  {
    icon: Shield,
    title: 'Enterprise Security',
    description: 'Bank-grade security with GDPR, HIPAA, and SOX compliance',
    color: 'text-success-500'
  },
  {
    icon: BarChart3,
    title: 'Real-time Analytics',
    description: 'Monitor performance, ROI, and productivity metrics',
    color: 'text-primary'
  },
  {
    icon: Users,
    title: 'Multi-Agent Coordination',
    description: 'Teams of AI employees working together seamlessly',
    color: 'text-purple-500'
  },
  {
    icon: Workflow,
    title: 'Automation Integration',
    description: 'Connect with Zapier, n8n, and custom workflows',
    color: 'text-orange-500'
  }
]

const stats = [
  { label: 'Performance Improvement', value: '5x-50x', icon: TrendingUp },
  { label: 'Accuracy Rate', value: '99.9%+', icon: CheckCircle },
  { label: 'Enterprise Clients', value: '500+', icon: Users },
  { label: 'AI Employees Created', value: '10,000+', icon: Bot }
]

const testimonials = [
  {
    quote: "Forge 1 transformed our customer service. Our AI employees handle 80% of inquiries with 99.5% accuracy.",
    author: "Sarah Chen",
    role: "CTO, TechCorp",
    rating: 5
  },
  {
    quote: "The ROI was immediate. We saw 10x productivity gains in data processing within the first week.",
    author: "Michael Rodriguez",
    role: "Operations Director, DataFlow Inc",
    rating: 5
  },
  {
    quote: "Finally, an AI platform that understands enterprise needs. The security and compliance features are outstanding.",
    author: "Jennifer Park",
    role: "CISO, SecureBank",
    rating: 5
  }
]

export default function HomePage() {
  const { user, isLoading } = useAuth()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return null
  }

  return (
    <div className="flex flex-col min-h-screen">
      {/* Navigation */}
      <nav className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container-forge flex h-16 items-center justify-between">
          <div className="flex items-center space-x-2">
            <Bot className="h-8 w-8 text-forge-500" />
            <span className="text-xl font-bold text-gradient">Forge 1</span>
          </div>
          
          <div className="flex items-center space-x-4">
            {isLoading ? (
              <div className="h-9 w-20 loading-skeleton" />
            ) : user ? (
              <div className="flex items-center space-x-2">
                <span className="text-sm text-muted-foreground">Welcome back, {user.name}</span>
                <Button asChild>
                  <Link href="/dashboard">Dashboard</Link>
                </Button>
              </div>
            ) : (
              <div className="flex items-center space-x-2">
                <Button variant="ghost" asChild>
                  <Link href="/auth/login">Sign In</Link>
                </Button>
                <Button asChild>
                  <Link href="/auth/register">Get Started</Link>
                </Button>
              </div>
            )}
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="section-padding bg-gradient-to-br from-background via-forge-50/20 to-background">
        <div className="container-forge">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center space-y-8"
          >
            <div className="space-y-4">
              <Badge variant="secondary" className="px-4 py-2">
                🚀 Now Available for Enterprise
              </Badge>
              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight">
                Build{' '}
                <span className="text-gradient">Superhuman</span>
                <br />
                AI Employees
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto text-balance">
                Create specialized AI employees that deliver 5x-50x performance improvements. 
                Enterprise-grade security, real-time analytics, and seamless integration.
              </p>
            </div>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button size="lg" className="btn-glow" asChild>
                <Link href="/auth/register">
                  Start Building AI Employees
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
              <Button size="lg" variant="outline" asChild>
                <Link href="/demo">
                  Watch Demo
                </Link>
              </Button>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-8 pt-12">
              {stats.map((stat, index) => (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  className="text-center"
                >
                  <div className="flex justify-center mb-2">
                    <stat.icon className="h-8 w-8 text-forge-500" />
                  </div>
                  <div className="text-2xl font-bold">{stat.value}</div>
                  <div className="text-sm text-muted-foreground">{stat.label}</div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="section-padding">
        <div className="container-forge">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="text-center space-y-4 mb-16"
          >
            <h2 className="text-3xl sm:text-4xl font-bold">
              Everything You Need to Build AI Employees
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              From creation to deployment, monitor and optimize your AI workforce with enterprise-grade tools.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <Card className="card-hover h-full">
                  <CardHeader>
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 rounded-lg bg-muted ${feature.color}`}>
                        <feature.icon className="h-6 w-6" />
                      </div>
                      <CardTitle className="text-lg">{feature.title}</CardTitle>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-base">
                      {feature.description}
                    </CardDescription>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="section-padding bg-muted/30">
        <div className="container-forge">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="text-center space-y-4 mb-16"
          >
            <h2 className="text-3xl sm:text-4xl font-bold">
              Trusted by Enterprise Leaders
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              See how companies are achieving superhuman performance with Forge 1.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <Card className="card-hover h-full">
                  <CardContent className="pt-6">
                    <div className="space-y-4">
                      <div className="flex space-x-1">
                        {[...Array(testimonial.rating)].map((_, i) => (
                          <Star key={i} className="h-4 w-4 fill-warning-400 text-warning-400" />
                        ))}
                      </div>
                      <blockquote className="text-muted-foreground italic">
                        "{testimonial.quote}"
                      </blockquote>
                      <div>
                        <div className="font-semibold">{testimonial.author}</div>
                        <div className="text-sm text-muted-foreground">{testimonial.role}</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="section-padding bg-gradient-forge text-white">
        <div className="container-forge">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="text-center space-y-8"
          >
            <div className="space-y-4">
              <h2 className="text-3xl sm:text-4xl font-bold">
                Ready to Build Your AI Workforce?
              </h2>
              <p className="text-xl opacity-90 max-w-2xl mx-auto">
                Join hundreds of enterprises already using Forge 1 to achieve superhuman performance.
              </p>
            </div>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button size="lg" variant="secondary" className="bg-white text-forge-600 hover:bg-white/90" asChild>
                <Link href="/auth/register">
                  Start Free Trial
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
              <Button size="lg" variant="outline" className="border-white text-white hover:bg-white/10" asChild>
                <Link href="/contact">
                  Contact Sales
                </Link>
              </Button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t bg-background">
        <div className="container-forge py-12">
          <div className="grid md:grid-cols-4 gap-8">
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <Bot className="h-6 w-6 text-forge-500" />
                <span className="font-bold text-gradient">Forge 1</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Building the future of work with superhuman AI employees.
              </p>
            </div>

            <div className="space-y-4">
              <h4 className="font-semibold">Product</h4>
              <div className="space-y-2 text-sm">
                <Link href="/features" className="block text-muted-foreground hover:text-foreground">Features</Link>
                <Link href="/pricing" className="block text-muted-foreground hover:text-foreground">Pricing</Link>
                <Link href="/integrations" className="block text-muted-foreground hover:text-foreground">Integrations</Link>
                <Link href="/security" className="block text-muted-foreground hover:text-foreground">Security</Link>
              </div>
            </div>

            <div className="space-y-4">
              <h4 className="font-semibold">Company</h4>
              <div className="space-y-2 text-sm">
                <Link href="/about" className="block text-muted-foreground hover:text-foreground">About</Link>
                <Link href="/blog" className="block text-muted-foreground hover:text-foreground">Blog</Link>
                <Link href="/careers" className="block text-muted-foreground hover:text-foreground">Careers</Link>
                <Link href="/contact" className="block text-muted-foreground hover:text-foreground">Contact</Link>
              </div>
            </div>

            <div className="space-y-4">
              <h4 className="font-semibold">Support</h4>
              <div className="space-y-2 text-sm">
                <Link href="/docs" className="block text-muted-foreground hover:text-foreground">Documentation</Link>
                <Link href="/help" className="block text-muted-foreground hover:text-foreground">Help Center</Link>
                <Link href="/status" className="block text-muted-foreground hover:text-foreground">Status</Link>
                <Link href="/api" className="block text-muted-foreground hover:text-foreground">API</Link>
              </div>
            </div>
          </div>

          <div className="border-t mt-12 pt-8 flex flex-col sm:flex-row justify-between items-center">
            <p className="text-sm text-muted-foreground">
              © 2024 Forge 1. All rights reserved.
            </p>
            <div className="flex space-x-6 text-sm text-muted-foreground">
              <Link href="/privacy" className="hover:text-foreground">Privacy</Link>
              <Link href="/terms" className="hover:text-foreground">Terms</Link>
              <Link href="/cookies" className="hover:text-foreground">Cookies</Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}