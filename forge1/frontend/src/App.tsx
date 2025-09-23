// forge1/frontend/src/App.tsx
/**
 * Forge 1 Frontend Application
 * 
 * Enhanced React application extending Microsoft's frontend with:
 * - AI Employee Builder interface
 * - Premium Client Dashboard
 * - Performance monitoring
 */

import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { FluentProvider, webLightTheme } from '@fluentui/react-components';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Import enhanced components
import { HomePage } from './pages/HomePage';
import { EmployeeBuilderPage } from './pages/EmployeeBuilderPage';
import { DashboardPage } from './pages/DashboardPage';
import { Header } from './components/Header';
import { ErrorBoundary } from './components/ErrorBoundary';

// Create query client for data fetching
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

function App() {
  return (
    <FluentProvider theme={webLightTheme}>
      <QueryClientProvider client={queryClient}>
        <ErrorBoundary>
          <Router>
            <div className="forge1-app">
              <Header />
              <main className="main-content">
                <Routes>
                  <Route path="/" element={<HomePage />} />
                  <Route path="/builder" element={<EmployeeBuilderPage />} />
                  <Route path="/dashboard" element={<DashboardPage />} />
                </Routes>
              </main>
            </div>
          </Router>
        </ErrorBoundary>
      </QueryClientProvider>
    </FluentProvider>
  );
}

export default App;