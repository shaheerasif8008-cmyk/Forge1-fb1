# 🔑 Forge 1 API Keys Guide
## Complete Guide to Getting All Required API Keys

This guide provides step-by-step instructions for obtaining all API keys needed for Forge 1's superhuman performance.

## 🚀 Critical AI Model Providers (REQUIRED)

### 1. OpenAI (GPT-4o, GPT-4 Turbo) - **ESSENTIAL**

**Why needed:** Core AI functionality, most advanced reasoning capabilities

**Steps to get API key:**
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in to your account
3. Navigate to [API Keys](https://platform.openai.com/api-keys)
4. Click "Create new secret key"
5. Copy the key (starts with `sk-`)

**Environment variables:**
```bash
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_ORG_ID=org-your-organization-id-here  # Optional
```

**Cost:** Pay-per-use, ~$0.01-0.06 per 1K tokens
**Free tier:** $5 credit for new accounts

---

### 2. Anthropic (Claude 3 Opus, Sonnet) - **HIGHLY RECOMMENDED**

**Why needed:** Superior reasoning, longer context, ethical AI responses

**Steps to get API key:**
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Sign up for an account
3. Navigate to [API Keys](https://console.anthropic.com/settings/keys)
4. Click "Create Key"
5. Copy the key (starts with `sk-ant-`)

**Environment variables:**
```bash
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
```

**Cost:** Pay-per-use, ~$0.015-0.075 per 1K tokens
**Free tier:** $5 credit for new accounts

---

### 3. Google AI (Gemini Pro, Ultra) - **RECOMMENDED**

**Why needed:** Cost-effective, multimodal capabilities, fast inference

**Steps to get API key:**
1. Go to [Google AI Studio](https://makersuite.google.com/)
2. Sign in with Google account
3. Click "Get API Key"
4. Create new project or select existing
5. Copy the API key

**Environment variables:**
```bash
GOOGLE_AI_API_KEY=your-google-ai-api-key-here
GOOGLE_PROJECT_ID=your-google-cloud-project-id
```

**Cost:** Pay-per-use, ~$0.001-0.01 per 1K tokens
**Free tier:** Generous free quota

---

## 🏢 Enterprise Integrations (OPTIONAL)

### CRM Systems

#### Salesforce
**Steps:**
1. Log in to [Salesforce](https://login.salesforce.com/)
2. Go to Setup → Apps → App Manager
3. Create "New Connected App"
4. Enable OAuth settings
5. Copy Client ID and Secret

```bash
SALESFORCE_CLIENT_ID=your-salesforce-client-id
SALESFORCE_CLIENT_SECRET=your-salesforce-client-secret
SALESFORCE_INSTANCE_URL=https://your-instance.salesforce.com
```

#### HubSpot
**Steps:**
1. Go to [HubSpot Developers](https://developers.hubspot.com/)
2. Create account or log in
3. Go to "Manage Apps" → "Create App"
4. Copy API key from app settings

```bash
HUBSPOT_API_KEY=your-hubspot-api-key
HUBSPOT_CLIENT_ID=your-hubspot-client-id
HUBSPOT_CLIENT_SECRET=your-hubspot-client-secret
```

### Support Systems

#### Zendesk
**Steps:**
1. Log in to your Zendesk instance
2. Go to Admin → Channels → API
3. Enable Token Access
4. Generate new token

```bash
ZENDESK_SUBDOMAIN=your-zendesk-subdomain
ZENDESK_EMAIL=your-zendesk-email
ZENDESK_API_TOKEN=your-zendesk-api-token
```

#### Intercom
**Steps:**
1. Go to [Intercom Developer Hub](https://developers.intercom.com/)
2. Create new app
3. Get Access Token from app settings

```bash
INTERCOM_ACCESS_TOKEN=your-intercom-access-token
INTERCOM_APP_ID=your-intercom-app-id
```

### Communication

#### Slack
**Steps:**
1. Go to [Slack API](https://api.slack.com/apps)
2. Create "New App" → "From scratch"
3. Add Bot Token Scopes
4. Install app to workspace
5. Copy Bot User OAuth Token

```bash
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
SLACK_SIGNING_SECRET=your-slack-signing-secret
SLACK_CLIENT_ID=your-slack-client-id
SLACK_CLIENT_SECRET=your-slack-client-secret
```

## 💳 Payment Processing (FOR BILLING)

### Stripe
**Steps:**
1. Go to [Stripe Dashboard](https://dashboard.stripe.com/)
2. Create account or log in
3. Go to Developers → API Keys
4. Copy Publishable and Secret keys

```bash
STRIPE_PUBLISHABLE_KEY=pk_test_your-stripe-publishable-key
STRIPE_SECRET_KEY=sk_test_your-stripe-secret-key
STRIPE_WEBHOOK_SECRET=whsec_your-stripe-webhook-secret
```

**Note:** Use test keys for development, live keys for production

## 🗄️ Vector Databases (FOR ENHANCED MEMORY)

### Pinecone (Recommended)
**Steps:**
1. Go to [Pinecone](https://www.pinecone.io/)
2. Sign up for free account
3. Create new project
4. Go to API Keys section
5. Copy API key and environment

```bash
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-pinecone-environment
PINECONE_INDEX_NAME=forge1-embeddings
```

**Free tier:** 1 index, 5M vectors

### Weaviate (Alternative)
**Steps:**
1. Go to [Weaviate Cloud](https://console.weaviate.cloud/)
2. Create free cluster
3. Get cluster URL and API key

```bash
WEAVIATE_URL=https://your-cluster.weaviate.network
WEAVIATE_API_KEY=your-weaviate-api-key
```

## 📊 Monitoring & Analytics (OPTIONAL)

### Sentry (Error Tracking)
**Steps:**
1. Go to [Sentry](https://sentry.io/)
2. Create account and project
3. Copy DSN from project settings

```bash
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
```

### Google Analytics
**Steps:**
1. Go to [Google Analytics](https://analytics.google.com/)
2. Create property
3. Copy Measurement ID

```bash
NEXT_PUBLIC_GA_MEASUREMENT_ID=G-XXXXXXXXXX
```

## 🔐 Authentication Providers (OPTIONAL)

### Google OAuth
**Steps:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create project → APIs & Services → Credentials
3. Create OAuth 2.0 Client ID
4. Add authorized redirect URIs

```bash
GOOGLE_CLIENT_ID=your-google-oauth-client-id
GOOGLE_CLIENT_SECRET=your-google-oauth-client-secret
```

### Microsoft OAuth
**Steps:**
1. Go to [Azure Portal](https://portal.azure.com/)
2. Azure Active Directory → App registrations
3. New registration
4. Copy Application ID and create client secret

```bash
MICROSOFT_CLIENT_ID=your-microsoft-oauth-client-id
MICROSOFT_CLIENT_SECRET=your-microsoft-oauth-client-secret
```

## 💰 Cost Estimation

### Minimal Setup (AI Only)
- **OpenAI**: ~$10-50/month for moderate usage
- **Anthropic**: ~$10-30/month for moderate usage
- **Google AI**: ~$5-20/month for moderate usage
- **Total**: ~$25-100/month

### Full Enterprise Setup
- **AI Models**: ~$100-500/month
- **Vector Database**: ~$20-100/month
- **Monitoring**: ~$10-50/month
- **Integrations**: Usually free tiers available
- **Total**: ~$130-650/month

## 🚀 Quick Setup Priority

### Priority 1 (Essential - 5 minutes)
1. **OpenAI API Key** - Core functionality
2. **Security keys** - Generated automatically by configure.sh

### Priority 2 (Enhanced - 10 minutes)
1. **Anthropic API Key** - Better reasoning
2. **Google AI API Key** - Cost optimization
3. **Pinecone API Key** - Enhanced memory

### Priority 3 (Enterprise - 30 minutes)
1. **Stripe Keys** - Billing functionality
2. **CRM Integration** - Business automation
3. **Support Integration** - Customer service automation

## 🛠️ Configuration Tools

### Automated Configuration
```bash
# Run the configuration script
./configure.sh
```

### Manual Configuration
```bash
# Copy environment files
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local

# Edit files with your API keys
nano backend/.env
nano frontend/.env.local
```

### Verification
```bash
# Test your configuration
./health-check.sh

# Test AI functionality
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"description": "Test AI functionality"}'
```

## 🔒 Security Best Practices

### API Key Security
- ✅ Never commit API keys to version control
- ✅ Use environment variables only
- ✅ Use different keys for dev/staging/production
- ✅ Rotate keys regularly
- ✅ Monitor usage and set billing alerts

### Access Control
- ✅ Use least privilege principle
- ✅ Enable MFA where available
- ✅ Monitor API usage logs
- ✅ Set up usage alerts

## 🆘 Troubleshooting

### Common Issues

#### "Invalid API Key" Errors
- Verify key is correctly copied (no extra spaces)
- Check if key has proper permissions
- Ensure key hasn't expired
- Verify environment variable name is correct

#### "Rate Limit Exceeded"
- Check your usage limits
- Implement exponential backoff
- Consider upgrading your plan
- Use multiple providers for load balancing

#### "Insufficient Quota"
- Add billing information to your account
- Increase spending limits
- Monitor usage dashboards
- Set up billing alerts

## 📞 Support Resources

### AI Providers
- **OpenAI**: [Help Center](https://help.openai.com/)
- **Anthropic**: [Documentation](https://docs.anthropic.com/)
- **Google AI**: [Support](https://developers.generativeai.google/)

### Integration Providers
- **Salesforce**: [Trailhead](https://trailhead.salesforce.com/)
- **HubSpot**: [Developer Docs](https://developers.hubspot.com/)
- **Slack**: [API Documentation](https://api.slack.com/)

### Forge 1 Platform
- Check `SETUP_GUIDE.md` for detailed setup instructions
- Run `./health-check.sh` for system diagnostics
- View logs: `docker-compose logs forge1-backend`

---

**Ready to configure your API keys?** Run `./configure.sh` to get started! 🚀