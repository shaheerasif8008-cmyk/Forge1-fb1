// forge1/frontend/src/components/Header.tsx
/**
 * Application Header Component
 */

import React from 'react';
import { Button } from '@fluentui/react-components';
import { useNavigate, useLocation } from 'react-router-dom';

export const Header: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <header className="app-header">
      <div className="header-content">
        <div className="logo">
          <h1>Forge 1</h1>
        </div>
        
        <nav className="navigation">
          <Button 
            appearance={location.pathname === '/' ? 'primary' : 'subtle'}
            onClick={() => navigate('/')}
          >
            Home
          </Button>
          
          <Button 
            appearance={location.pathname === '/builder' ? 'primary' : 'subtle'}
            onClick={() => navigate('/builder')}
          >
            Employee Builder
          </Button>
          
          <Button 
            appearance={location.pathname === '/dashboard' ? 'primary' : 'subtle'}
            onClick={() => navigate('/dashboard')}
          >
            Dashboard
          </Button>
        </nav>
      </div>
    </header>
  );
};