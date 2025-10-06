# 🚀 Forge 1 Access Guide

## **Quick Start (2 minutes)**

### **Step 1: Start the Platform**
```bash
# Navigate to the Forge 1 directory
cd forge1

# Start everything with one command
./quick-start.sh
```

### **Step 2: Access the UI**
Open your browser and go to: **http://localhost:3000**

### **Step 3: Login**
- **Email:** `admin@cognisia.com`
- **Password:** `admin123`

**That's it! You're now in the Forge 1 platform! 🎉**

---

## **What You'll See**

### **🏠 Dashboard Overview**
When you first login, you'll see:
- **AI Employee Status**: Overview of all your AI employees
- **Recent Activity**: Latest tasks and workflows
- **Performance Metrics**: Success rates and efficiency scores
- **System Health**: Platform status and alerts

### **🤖 AI Employees Section**
Click "AI Employees" to see:
- **6 Pre-built Verticals**: CX, RevOps, Finance, Legal, IT Ops, Software Engineering
- **Create New Employee**: Wizard to set up specialized AI workers
- **Employee Performance**: Individual metrics and capabilities
- **Task Assignment**: Assign work to specific employees

### **🔗 Integrations Hub**
Click "Integrations" to configure:
- **CRM Systems**: Salesforce, HubSpot, Pipedrive
- **Support Tools**: Zendesk, Intercom, Freshdesk
- **Communication**: Slack, Microsoft Teams, Email
- **Development**: GitHub, Jira, GitLab

### **⚙️ Workflows & Automation**
Click "Workflows" to:
- **Create Multi-Agent Workflows**: Chain AI employees together
- **Set Approval Gates**: Human-in-the-loop for critical decisions
- **Monitor Execution**: Real-time workflow status
- **Performance Analytics**: Workflow efficiency metrics

---

## **🧪 Test Scenarios (Try These!)**

### **Scenario 1: Customer Support Automation (5 minutes)**
1. **Go to AI Employees → Create New**
2. **Select "Customer Experience"**
3. **Name it "Sarah - Support Specialist"**
4. **Configure Skills**: Ticket triage, response generation
5. **Test with Sample Ticket**:
   ```
   Subject: "Billing Issue - Overcharged"
   Message: "I was charged twice for my subscription this month. Please help!"
   ```
6. **Watch Sarah**:
   - Automatically categorize as "Billing"
   - Generate empathetic response
   - Route to appropriate team
   - Create follow-up tasks

### **Scenario 2: Sales Pipeline Optimization (5 minutes)**
1. **Create RevOps AI Employee**: "Marcus - Revenue Analyst"
2. **Connect to CRM** (use demo data)
3. **Upload Sample Deals**:
   ```
   Deal 1: "Enterprise Corp - $50K - 30% probability"
   Deal 2: "Startup Inc - $5K - 80% probability"
   Deal 3: "Big Corp - $100K - 60% probability"
   ```
4. **Watch Marcus**:
   - Analyze deal health
   - Predict close probability
   - Recommend next actions
   - Generate forecast reports

### **Scenario 3: Multi-Agent Workflow (10 minutes)**
1. **Go to Workflows → Create New**
2. **Build "New Customer Onboarding"**:
   - **Trigger**: New customer signup
   - **Step 1**: CX employee sends welcome email
   - **Step 2**: RevOps employee creates CRM record
   - **Step 3**: Finance employee sets up billing
   - **Step 4**: IT Ops employee provisions access
3. **Test with Sample Customer**:
   ```
   Name: "Test Customer Inc"
   Email: "test@customer.com"
   Plan: "Professional"
   ```
4. **Watch the Workflow Execute** across all employees

---

## **🔍 Key Features to Explore**

### **💰 Billing & Pricing**
- **Navigate to**: Settings → Billing
- **Test**: Plan upgrades, usage tracking, cost optimization
- **See**: Real-time usage meters and cost projections

### **🛡️ Security & Compliance**
- **Navigate to**: Compliance → Dashboard
- **Test**: GDPR data requests, SOC2 controls, audit trails
- **See**: Compliance status across all frameworks

### **📊 Analytics & Reporting**
- **Navigate to**: Analytics → Performance
- **Test**: Custom reports, performance metrics, ROI calculations
- **See**: AI employee efficiency and business impact

### **🔧 System Administration**
- **Navigate to**: Admin → System Health
- **Test**: User management, system monitoring, configuration
- **See**: Real-time system metrics and alerts

---

## **🐛 Troubleshooting**

### **UI Not Loading?**
```bash
# Check if services are running
docker-compose ps

# If not, restart everything
docker-compose down
./quick-start.sh
```

### **Login Not Working?**
- **Try**: `admin@cognisia.com` / `admin123`
- **Or**: Create new account with "Sign Up"
- **Check**: Browser console for errors (F12)

### **Integrations Failing?**
- **Use Demo Mode**: Most features work with sample data
- **Check Logs**: `docker-compose logs backend`
- **Verify Config**: Settings → Integrations

### **Performance Issues?**
```bash
# Check system resources
docker stats

# Restart specific service
docker-compose restart backend

# Check logs for errors
docker-compose logs -f
```

---

## **🎉 Success Indicators**

You'll know everything is working when:
- ✅ **UI loads instantly** at http://localhost:3000
- ✅ **Login works** with provided credentials
- ✅ **AI employees respond** to test scenarios
- ✅ **Workflows execute** end-to-end
- ✅ **Integrations connect** (even in demo mode)
- ✅ **Real-time updates** appear in the dashboard

**Welcome to the future of AI-powered business automation! 🤖✨**