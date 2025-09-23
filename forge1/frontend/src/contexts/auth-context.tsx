'use client'

import { createContext, useContext, useEffect, useState, ReactNode } from 'react'
import { useRouter } from 'next/navigation'
import { toast } from 'react-hot-toast'
import { api } from '@/lib/api'

export interface User {
  id: string
  email: string
  name: string
  role: string
  avatar?: string
  organization?: {
    id: string
    name: string
    plan: string
  }
  permissions: string[]
  preferences: {
    theme: 'light' | 'dark' | 'system'
    notifications: boolean
    language: string
  }
  createdAt: string
  lastLoginAt: string
}

interface AuthContextType {
  user: User | null
  isLoading: boolean
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<void>
  register: (data: RegisterData) => Promise<void>
  logout: () => Promise<void>
  updateUser: (data: Partial<User>) => Promise<void>
  refreshUser: () => Promise<void>
}

interface RegisterData {
  email: string
  password: string
  name: string
  organizationName?: string
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const router = useRouter()

  const isAuthenticated = !!user

  // Initialize auth state
  useEffect(() => {
    const initAuth = async () => {
      try {
        const token = localStorage.getItem('auth_token')
        if (!token) {
          setIsLoading(false)
          return
        }

        // Verify token and get user data
        const response = await api.get('/auth/me')
        setUser(response.data.user)
      } catch (error) {
        // Token is invalid, remove it
        localStorage.removeItem('auth_token')
        console.error('Auth initialization failed:', error)
      } finally {
        setIsLoading(false)
      }
    }

    initAuth()
  }, [])

  const login = async (email: string, password: string) => {
    try {
      setIsLoading(true)
      const response = await api.post('/auth/login', { email, password })
      
      const { user: userData, token } = response.data
      
      // Store token
      localStorage.setItem('auth_token', token)
      
      // Set user data
      setUser(userData)
      
      toast.success('Welcome back!')
      router.push('/dashboard')
    } catch (error: any) {
      const message = error.response?.data?.message || 'Login failed'
      toast.error(message)
      throw error
    } finally {
      setIsLoading(false)
    }
  }

  const register = async (data: RegisterData) => {
    try {
      setIsLoading(true)
      const response = await api.post('/auth/register', data)
      
      const { user: userData, token } = response.data
      
      // Store token
      localStorage.setItem('auth_token', token)
      
      // Set user data
      setUser(userData)
      
      toast.success('Account created successfully!')
      router.push('/dashboard')
    } catch (error: any) {
      const message = error.response?.data?.message || 'Registration failed'
      toast.error(message)
      throw error
    } finally {
      setIsLoading(false)
    }
  }

  const logout = async () => {
    try {
      setIsLoading(true)
      
      // Call logout endpoint
      await api.post('/auth/logout')
      
      // Clear local state
      localStorage.removeItem('auth_token')
      setUser(null)
      
      toast.success('Logged out successfully')
      router.push('/')
    } catch (error) {
      // Even if the API call fails, clear local state
      localStorage.removeItem('auth_token')
      setUser(null)
      router.push('/')
    } finally {
      setIsLoading(false)
    }
  }

  const updateUser = async (data: Partial<User>) => {
    try {
      const response = await api.patch('/auth/profile', data)
      setUser(response.data.user)
      toast.success('Profile updated successfully')
    } catch (error: any) {
      const message = error.response?.data?.message || 'Update failed'
      toast.error(message)
      throw error
    }
  }

  const refreshUser = async () => {
    try {
      const response = await api.get('/auth/me')
      setUser(response.data.user)
    } catch (error) {
      console.error('Failed to refresh user data:', error)
    }
  }

  const value: AuthContextType = {
    user,
    isLoading,
    isAuthenticated,
    login,
    register,
    logout,
    updateUser,
    refreshUser,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}