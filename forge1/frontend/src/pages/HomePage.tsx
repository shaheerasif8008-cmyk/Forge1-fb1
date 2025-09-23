// forge1/frontend/src/pages/HomePage.tsx
/**
 * Forge 1 Home Page
 * 
 * Landing page for the Forge 1 platform with navigation to key features.
 */

import React from 'react';
import { Button, Title1, Body1 } from '@fluentui/react-components';
import { useNavigate } from 'react-router-dom';

export const HomePage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="home-page">
      <div className="hero-section">
        <Title1>Welcome to Forge 1</Title1>
        <Body1>
          Enterprise AI Employee Builder Platform - Create superhuman AI employees 
          that outperform professional humans by 5x-50x.
        </Body1>
        
        <div className="action-buttons">
          <Button 
            appearance="primary" 
            size="large"
            onClick={() => navigate('/builder')}
          >
            Build AI Employee
          </Button>
          
          <Button 
            appearance="secondary" 
            size="large"
            onClick={() => navigate('/dashboard')}
          >
            View Dashboard
          </Button>
        </div>
      </div>
      
      <div className="features-section">
        <div className="feature-card">
          <h3>Superhuman Performance</h3>
          <p>AI employees that deliver 5x-50x performance improvements</p>
        </div>
        
        <div className="feature-card">
          <h3>Multi-Model Intelligence</h3>
          <p>Intelligent routing across GPT-4o/5, Claude, Gemini</p>
        </div>
        
        <div className="feature-card">
          <h3>Enterprise Security</h3>
          <p>Full compliance with GDPR, CCPA, HIPAA, SOX</p>
        </div>
      </div>
    </div>
  );
};