import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Providers } from '@/components/providers'
import { Toaster } from '@/components/ui/toaster'
import { cn } from '@/lib/utils'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Forge 1 - AI Employee Builder',
  description: 'Build superhuman AI employees that deliver 5x-50x performance improvements',
  keywords: ['AI', 'automation', 'enterprise', 'employees', 'productivity'],
  authors: [{ name: 'Forge 1 Team' }],
  creator: 'Forge 1',
  publisher: 'Forge 1',
  robots: {
    index: true,
    follow: true,
  },
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://forge1.com',
    title: 'Forge 1 - AI Employee Builder',
    description: 'Build superhuman AI employees that deliver 5x-50x performance improvements',
    siteName: 'Forge 1',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Forge 1 - AI Employee Builder',
    description: 'Build superhuman AI employees that deliver 5x-50x performance improvements',
    creator: '@forge1ai',
  },
  viewport: {
    width: 'device-width',
    initialScale: 1,
    maximumScale: 1,
  },
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#0ea5e9' },
    { media: '(prefers-color-scheme: dark)', color: '#0284c7' },
  ],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="icon" href="/favicon.ico" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
        <link rel="manifest" href="/manifest.json" />
      </head>
      <body className={cn(inter.className, 'min-h-screen bg-background antialiased')}>
        <Providers>
          <div className="relative flex min-h-screen flex-col">
            <div className="flex-1">
              {children}
            </div>
          </div>
          <Toaster />
        </Providers>
      </body>
    </html>
  )
}