#!/bin/bash

# Forge 1 Platform Configuration Script
# This script helps you set up the essential environment variables for actual performance

set -e

echo "🚀 Forge 1 Platform Configuration"
echo "=================================="
echo ""
echo "This script will help you configure Forge 1 for actual superhuman performance."
echo "You'll need API keys from AI providers for full functionality."
echo ""

# Create .env files if they don't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created main .env file"
fi

if [ ! -f backend/.env ]; then
    cp backend/.env.example backend/.env
    echo "✅ Created backend .env file"
fi

if [ ! -f frontend/.env.local ]; then
    cp frontend/.env.example frontend/.env.local
    echo "✅ Created frontend .env.local file"
fi

echo ""
echo "🔑 AI Model Configuration (REQUIRED for actual performance)"
echo "=========================================================="

# Function to prompt for API key
prompt_for_key() {
    local service=$1
    local var_name=$2
    local description=$3
    local url=$4
    
    echo ""
    echo "📝 $service Configuration"
    echo "   $description"
    echo "   Get your key from: $url"
    echo ""
    read -p "Enter your $service API key (or press Enter to skip): " api_key
    
    if [ ! -z "$api_key" ]; then
        # Update backend .env file
        if grep -q "^$var_name=" backend/.env; then
            sed -i.bak "s|^$var_name=.*|$var_name=$api_key|" backend/.env
        else
            echo "$var_name=$api_key" >> backend/.env
        fi
        echo "✅ $service API key configured"
    else
        echo "⚠️  $service API key skipped (you can add it later in backend/.env)"
    fi
}

# Configure AI providers
prompt_for_key "OpenAI" "OPENAI_API_KEY" "Required for GPT-4o and GPT-4 Turbo models" "https://platform.openai.com/api-keys"
prompt_for_key "Anthropic" "ANTHROPIC_API_KEY" "Required for Claude 3 Opus and Sonnet models" "https://console.anthropic.com/"
prompt_for_key "Google AI" "GOOGLE_AI_API_KEY" "Required for Gemini Pro and Ultra models" "https://makersuite.google.com/app/apikey"

echo ""
echo "🔐 Security Configuration"
echo "========================"

# Generate secure keys
echo "Generating secure keys..."

# Generate JWT secret
jwt_secret=$(openssl rand -base64 32 2>/dev/null || python3 -c "import secrets; print(secrets.token_urlsafe(32))" 2>/dev/null || echo "change-this-jwt-secret-$(date +%s)")
sed -i.bak "s|^JWT_SECRET_KEY=.*|JWT_SECRET_KEY=$jwt_secret|" backend/.env
sed -i.bak "s|^NEXTAUTH_SECRET=.*|NEXTAUTH_SECRET=$jwt_secret|" frontend/.env.local

# Generate application secret
app_secret=$(openssl rand -base64 32 2>/dev/null || python3 -c "import secrets; print(secrets.token_urlsafe(32))" 2>/dev/null || echo "change-this-app-secret-$(date +%s)")
sed -i.bak "s|^SECRET_KEY=.*|SECRET_KEY=$app_secret|" backend/.env

# Generate encryption key
encryption_key=$(openssl rand -base64 32 2>/dev/null || python3 -c "import secrets; print(secrets.token_urlsafe(32))" 2>/dev/null || echo "change-this-encryption-key-$(date +%s)")
sed -i.bak "s|^ENCRYPTION_KEY=.*|ENCRYPTION_KEY=$encryption_key|" backend/.env

echo "✅ Security keys generated and configured"

echo ""
echo "🌐 Network Configuration"
echo "======================="

# Configure URLs
sed -i.bak "s|^NEXT_PUBLIC_API_URL=.*|NEXT_PUBLIC_API_URL=http://localhost:8000|" frontend/.env.local
sed -i.bak "s|^NEXT_PUBLIC_WS_URL=.*|NEXT_PUBLIC_WS_URL=ws://localhost:8000|" frontend/.env.local
sed -i.bak "s|^NEXTAUTH_URL=.*|NEXTAUTH_URL=http://localhost:3000|" frontend/.env.local

echo "✅ Network URLs configured"

echo ""
echo "💳 Optional: Payment Configuration"
echo "================================="

read -p "Do you want to configure Stripe for billing? (y/N): " configure_stripe

if [[ $configure_stripe =~ ^[Yy]$ ]]; then
    echo ""
    echo "📝 Stripe Configuration"
    echo "   Get your keys from: https://dashboard.stripe.com/apikeys"
    echo ""
    
    read -p "Enter your Stripe Publishable Key (pk_test_...): " stripe_pub_key
    read -p "Enter your Stripe Secret Key (sk_test_...): " stripe_secret_key
    
    if [ ! -z "$stripe_pub_key" ] && [ ! -z "$stripe_secret_key" ]; then
        # Update backend .env
        sed -i.bak "s|^STRIPE_PUBLISHABLE_KEY=.*|STRIPE_PUBLISHABLE_KEY=$stripe_pub_key|" backend/.env
        sed -i.bak "s|^STRIPE_SECRET_KEY=.*|STRIPE_SECRET_KEY=$stripe_secret_key|" backend/.env
        
        # Update frontend .env.local
        sed -i.bak "s|^NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=.*|NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=$stripe_pub_key|" frontend/.env.local
        
        echo "✅ Stripe configuration completed"
    else
        echo "⚠️  Stripe configuration skipped"
    fi
else
    echo "⚠️  Stripe configuration skipped"
fi

echo ""
echo "🎯 Feature Configuration"
echo "======================="

# Enable core features
sed -i.bak "s|^MULTI_MODEL_ROUTING_ENABLED=.*|MULTI_MODEL_ROUTING_ENABLED=true|" backend/.env
sed -i.bak "s|^SUPERHUMAN_AGENTS_ENABLED=.*|SUPERHUMAN_AGENTS_ENABLED=true|" backend/.env
sed -i.bak "s|^PERFORMANCE_MONITORING_ENABLED=.*|PERFORMANCE_MONITORING_ENABLED=true|" backend/.env
sed -i.bak "s|^COMPLIANCE_ENGINE_ENABLED=.*|COMPLIANCE_ENGINE_ENABLED=true|" backend/.env

# Enable frontend features
sed -i.bak "s|^NEXT_PUBLIC_ENABLE_AI_CHAT=.*|NEXT_PUBLIC_ENABLE_AI_CHAT=true|" frontend/.env.local
sed -i.bak "s|^NEXT_PUBLIC_ENABLE_REAL_TIME_UPDATES=.*|NEXT_PUBLIC_ENABLE_REAL_TIME_UPDATES=true|" frontend/.env.local

echo "✅ Core features enabled"

# Clean up backup files
rm -f backend/.env.bak frontend/.env.local.bak .env.bak 2>/dev/null || true

echo ""
echo "🎉 Configuration Complete!"
echo "========================="
echo ""
echo "Your Forge 1 platform is now configured for actual performance!"
echo ""
echo "📋 What's been configured:"
echo "   ✅ Environment files created"
echo "   ✅ Security keys generated"
echo "   ✅ Network URLs configured"
echo "   ✅ Core features enabled"

# Check if AI keys were configured
ai_keys_configured=false
if grep -q "^OPENAI_API_KEY=sk-" backend/.env 2>/dev/null; then
    ai_keys_configured=true
fi

if [ "$ai_keys_configured" = true ]; then
    echo "   ✅ AI model keys configured"
    echo ""
    echo "🚀 Ready to start with full AI capabilities!"
else
    echo "   ⚠️  AI model keys not configured"
    echo ""
    echo "⚠️  To enable AI functionality, add your API keys to backend/.env:"
    echo "   - OPENAI_API_KEY=sk-your-key-here"
    echo "   - ANTHROPIC_API_KEY=sk-ant-your-key-here"
    echo "   - GOOGLE_AI_API_KEY=your-key-here"
fi

echo ""
echo "🚀 Next Steps:"
echo "   1. Start the platform: ./quick-start.sh"
echo "   2. Check health: ./health-check.sh"
echo "   3. Access UI: http://localhost:3000"
echo "   4. View API docs: http://localhost:8000/docs"
echo ""
echo "📚 For more configuration options, see:"
echo "   - SETUP_GUIDE.md (comprehensive setup guide)"
echo "   - backend/.env (backend configuration)"
echo "   - frontend/.env.local (frontend configuration)"
echo ""
echo "Happy building with Forge 1! 🎯"