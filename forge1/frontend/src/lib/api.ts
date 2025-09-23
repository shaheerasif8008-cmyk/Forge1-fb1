import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'
import { toast } from 'react-hot-toast'

// API configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Create axios instance
export const api: AxiosInstance = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
api.interceptors.response.use(
  (response: AxiosResponse) => {
    return response
  },
  (error) => {
    const { response } = error

    if (response?.status === 401) {
      // Unauthorized - clear token and redirect to login
      localStorage.removeItem('auth_token')
      if (typeof window !== 'undefined') {
        window.location.href = '/auth/login'
      }
      return Promise.reject(error)
    }

    if (response?.status === 403) {
      toast.error('You do not have permission to perform this action')
      return Promise.reject(error)
    }

    if (response?.status === 429) {
      toast.error('Too many requests. Please try again later.')
      return Promise.reject(error)
    }

    if (response?.status >= 500) {
      toast.error('Server error. Please try again later.')
      return Promise.reject(error)
    }

    // For other errors, let the component handle them
    return Promise.reject(error)
  }
)

// API endpoints
export const endpoints = {
  // Authentication
  auth: {
    login: '/auth/login',
    register: '/auth/register',
    logout: '/auth/logout',
    me: '/auth/me',
    refresh: '/auth/refresh',
    forgotPassword: '/auth/forgot-password',
    resetPassword: '/auth/reset-password',
    verifyEmail: '/auth/verify-email',
  },

  // Users
  users: {
    list: '/users',
    create: '/users',
    get: (id: string) => `/users/${id}`,
    update: (id: string) => `/users/${id}`,
    delete: (id: string) => `/users/${id}`,
    profile: '/users/profile',
  },

  // Organizations
  organizations: {
    list: '/organizations',
    create: '/organizations',
    get: (id: string) => `/organizations/${id}`,
    update: (id: string) => `/organizations/${id}`,
    delete: (id: string) => `/organizations/${id}`,
    members: (id: string) => `/organizations/${id}/members`,
    invite: (id: string) => `/organizations/${id}/invite`,
  },

  // AI Employees
  aiEmployees: {
    list: '/ai-employees',
    create: '/ai-employees',
    get: (id: string) => `/ai-employees/${id}`,
    update: (id: string) => `/ai-employees/${id}`,
    delete: (id: string) => `/ai-employees/${id}`,
    deploy: (id: string) => `/ai-employees/${id}/deploy`,
    stop: (id: string) => `/ai-employees/${id}/stop`,
    metrics: (id: string) => `/ai-employees/${id}/metrics`,
    logs: (id: string) => `/ai-employees/${id}/logs`,
    test: (id: string) => `/ai-employees/${id}/test`,
  },

  // Templates
  templates: {
    list: '/templates',
    create: '/templates',
    get: (id: string) => `/templates/${id}`,
    update: (id: string) => `/templates/${id}`,
    delete: (id: string) => `/templates/${id}`,
    categories: '/templates/categories',
  },

  // Workflows
  workflows: {
    list: '/workflows',
    create: '/workflows',
    get: (id: string) => `/workflows/${id}`,
    update: (id: string) => `/workflows/${id}`,
    delete: (id: string) => `/workflows/${id}`,
    execute: (id: string) => `/workflows/${id}/execute`,
    history: (id: string) => `/workflows/${id}/history`,
  },

  // Integrations
  integrations: {
    connectors: '/integrations/connectors',
    createConnector: '/integrations/connectors',
    getConnector: (id: string) => `/integrations/connectors/${id}`,
    deleteConnector: (id: string) => `/integrations/connectors/${id}`,
    executeRequest: (id: string) => `/integrations/connectors/${id}/request`,
    graphql: (id: string) => `/integrations/connectors/${id}/graphql`,
    soap: (id: string) => `/integrations/connectors/${id}/soap`,
    endpoints: (id: string) => `/integrations/connectors/${id}/endpoints`,
    schema: (id: string) => `/integrations/connectors/${id}/schema`,
    metrics: '/integrations/metrics',
    apiTypes: '/integrations/api-types',
    authMethods: '/integrations/auth-methods',
  },

  // Automation
  automation: {
    connectors: '/automation/connectors',
    createConnector: '/automation/connectors',
    getConnector: (id: string) => `/automation/connectors/${id}`,
    deleteConnector: (id: string) => `/automation/connectors/${id}`,
    workflows: (id: string) => `/automation/connectors/${id}/workflows`,
    executeWorkflow: (connectorId: string, workflowId: string) => 
      `/automation/connectors/${connectorId}/workflows/${workflowId}/execute`,
    webhooks: (id: string) => `/automation/connectors/${id}/webhooks`,
    metrics: '/automation/metrics',
    platforms: '/automation/platforms',
    triggerTypes: '/automation/trigger-types',
    actionTypes: '/automation/action-types',
  },

  // Analytics
  analytics: {
    dashboard: '/analytics/dashboard',
    performance: '/analytics/performance',
    usage: '/analytics/usage',
    costs: '/analytics/costs',
    roi: '/analytics/roi',
    reports: '/analytics/reports',
    export: '/analytics/export',
  },

  // Monitoring
  monitoring: {
    health: '/monitoring/health',
    metrics: '/monitoring/metrics',
    alerts: '/monitoring/alerts',
    logs: '/monitoring/logs',
    traces: '/monitoring/traces',
    status: '/monitoring/status',
  },

  // Settings
  settings: {
    general: '/settings/general',
    security: '/settings/security',
    notifications: '/settings/notifications',
    billing: '/settings/billing',
    api: '/settings/api',
    webhooks: '/settings/webhooks',
  },
}

// Utility functions for common API patterns
export const apiUtils = {
  // Generic CRUD operations
  list: <T>(endpoint: string, params?: Record<string, any>) =>
    api.get<{ data: T[]; total: number; page: number; limit: number }>(endpoint, { params }),

  get: <T>(endpoint: string) =>
    api.get<{ data: T }>(endpoint),

  create: <T>(endpoint: string, data: any) =>
    api.post<{ data: T }>(endpoint, data),

  update: <T>(endpoint: string, data: any) =>
    api.patch<{ data: T }>(endpoint, data),

  delete: (endpoint: string) =>
    api.delete(endpoint),

  // File upload
  upload: (endpoint: string, file: File, onProgress?: (progress: number) => void) => {
    const formData = new FormData()
    formData.append('file', file)

    return api.post(endpoint, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress(progress)
        }
      },
    })
  },

  // Download file
  download: async (endpoint: string, filename?: string) => {
    const response = await api.get(endpoint, {
      responseType: 'blob',
    })

    const url = window.URL.createObjectURL(new Blob([response.data]))
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', filename || 'download')
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.URL.revokeObjectURL(url)
  },

  // Paginated requests
  paginated: <T>(
    endpoint: string,
    page: number = 1,
    limit: number = 20,
    params?: Record<string, any>
  ) =>
    api.get<{
      data: T[]
      pagination: {
        page: number
        limit: number
        total: number
        totalPages: number
        hasNext: boolean
        hasPrev: boolean
      }
    }>(endpoint, {
      params: {
        page,
        limit,
        ...params,
      },
    }),

  // Search requests
  search: <T>(endpoint: string, query: string, filters?: Record<string, any>) =>
    api.get<{ data: T[]; total: number }>(endpoint, {
      params: {
        q: query,
        ...filters,
      },
    }),
}

// WebSocket connection
export const createWebSocketConnection = (path: string) => {
  const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'
  const token = localStorage.getItem('auth_token')
  
  const wsUrl = `${WS_BASE_URL}${path}${token ? `?token=${token}` : ''}`
  
  return new WebSocket(wsUrl)
}

export default api