// forge1/frontend/src/app/health/route.ts
import { NextResponse } from 'next/server'

export async function GET() {
  return NextResponse.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    service: 'forge1-frontend',
    version: '1.0.0'
  })
}