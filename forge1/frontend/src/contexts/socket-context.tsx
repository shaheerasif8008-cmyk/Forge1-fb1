'use client'

import { createContext, useContext, useEffect, useState, ReactNode } from 'react'
import { useAuth } from '@/hooks/use-auth'

interface SocketContextType {
  socket: WebSocket | null
  isConnected: boolean
  sendMessage: (message: any) => void
  subscribe: (event: string, callback: (data: any) => void) => () => void
}

const SocketContext = createContext<SocketContextType | undefined>(undefined)

export function SocketProvider({ children }: { children: ReactNode }) {
  const { user, isAuthenticated } = useAuth()
  const [socket, setSocket] = useState<WebSocket | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [eventListeners, setEventListeners] = useState<Map<string, Set<(data: any) => void>>>(new Map())

  useEffect(() => {
    if (!isAuthenticated || !user) {
      return
    }

    const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'
    const token = localStorage.getItem('auth_token')
    const wsUrl = `${WS_URL}/ws?token=${token}`

    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      console.log('WebSocket connected')
      setIsConnected(true)
    }

    ws.onclose = () => {
      console.log('WebSocket disconnected')
      setIsConnected(false)
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setIsConnected(false)
    }

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        const { type, data } = message

        // Notify all listeners for this event type
        const listeners = eventListeners.get(type)
        if (listeners) {
          listeners.forEach(callback => callback(data))
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error)
      }
    }

    setSocket(ws)

    return () => {
      ws.close()
      setSocket(null)
      setIsConnected(false)
    }
  }, [isAuthenticated, user, eventListeners])

  const sendMessage = (message: any) => {
    if (socket && isConnected) {
      socket.send(JSON.stringify(message))
    }
  }

  const subscribe = (event: string, callback: (data: any) => void) => {
    setEventListeners(prev => {
      const newMap = new Map(prev)
      if (!newMap.has(event)) {
        newMap.set(event, new Set())
      }
      newMap.get(event)!.add(callback)
      return newMap
    })

    // Return unsubscribe function
    return () => {
      setEventListeners(prev => {
        const newMap = new Map(prev)
        const listeners = newMap.get(event)
        if (listeners) {
          listeners.delete(callback)
          if (listeners.size === 0) {
            newMap.delete(event)
          }
        }
        return newMap
      })
    }
  }

  const value: SocketContextType = {
    socket,
    isConnected,
    sendMessage,
    subscribe,
  }

  return <SocketContext.Provider value={value}>{children}</SocketContext.Provider>
}

export function useSocket() {
  const context = useContext(SocketContext)
  if (context === undefined) {
    throw new Error('useSocket must be used within a SocketProvider')
  }
  return context
}