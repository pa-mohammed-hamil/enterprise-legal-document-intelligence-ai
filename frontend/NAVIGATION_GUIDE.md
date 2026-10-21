# Navigation Guide

## Complete Navigation Structure

All internal application pages now have a unified, comprehensive navigation system with access to all features.

---

## Navigation Structure

### Main Navigation Items

1. **Dashboard** - Overview and key metrics
2. **Documents** (Dropdown)
   - All Documents
   - Contracts
   - Reviews
   - Expiry Calendar
3. **Analytics** (Dropdown)
   - Analytics Dashboard
   - Predictive Analytics (NEW!)
   - Risk Assessment
4. **E-Signature** (NEW!) - Electronic signature management
5. **Legal Q&A** - AI-powered legal chat
6. **Settings** (Dropdown)
   - Admin Panel
   - AI Models (NEW!)
   - Collaboration (NEW!)
   - Data Security (DLP) (NEW!)
   - Notifications
   - Profile Settings
   - Logout

---

## Pages Overview

### Core Pages
- `dashboard.html` - Main dashboard with overview
- `documents.html` - Document management
- `contracts.html` - Contract management
- `reviews.html` - Contract reviews
- `risk.html` - Risk assessment
- `analytics.html` - Analytics dashboard
- `legal-chat.html` - Legal Q&A chatbot
- `expiry-calendar.html` - Contract expiration calendar
- `notifications.html` - Notifications center
- `admin.html` - Admin settings

### NEW Feature Pages
- `esignature.html` - E-Signature management (DocuSign, Adobe Sign, HelloSign)
- `predictive-analytics.html` - AI-powered contract outcome predictions
- `model-settings.html` - AI model selection and configuration
- `collaboration.html` - Real-time collaboration features
- `dlp-settings.html` - Data Leakage Prevention settings

---

## How to Update Navigation in Existing Pages

### Method 1: Copy Navigation HTML

Replace the `<nav>` element in each page with the content from `includes/nav.html`:

```html
<!-- Replace your existing nav with this -->
<nav class="navbar navbar-expand-lg navbar-dark bg-dark">
    <div class="container-fluid">
        <a class="navbar-brand" href="../index.html">
            <i class="bi bi-file-earmark-text"></i> Legal AI
        </a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
            <!-- Full navigation from includes/nav.html -->
        </div>
    </div>
</nav>
```

### Method 2: Add Navigation Script

Add this before closing `</body>` tag:

```html
<script src="../js/navigation.js"></script>
```

This script:
- Highlights the current active page
- Adds visual feedback for active menu items
- Handles dropdown active states

---

## Features by Page

### Dashboard (`dashboard.html`)
- Overview metrics
- Recent activity
- Quick actions
- Charts and visualizations

### E-Signature (`esignature.html`)
**NEW FEATURE**
- Create signature requests
- Multi-provider support (DocuSign, Adobe Sign, HelloSign)
- Track signature status
- Embedded signing
- Audit trails
- Webhooks integration

### Predictive Analytics (`predictive-analytics.html`)
**NEW FEATURE**
- Success probability predictions
- Dispute likelihood forecasting
- Renewal probability analysis
- Financial performance projections
- Risk factor analysis
- AI-powered recommendations
- Benchmark comparisons

### AI Models (`model-settings.html`)
**NEW FEATURE**
- Select from 100+ AI models
- 12 providers (OpenAI, Anthropic, Google, Cohere, Mistral, Groq, etc.)
- Model comparison
- Cost optimization
- Performance monitoring

### Collaboration (`collaboration.html`)
**NEW FEATURE**
- Real-time document collaboration
- Comments and annotations
- Presence tracking
- Change notifications
- Document sharing

### Data Security - DLP (`dlp-settings.html`)
**NEW FEATURE**
- Data leakage prevention
- Sensitive data detection (15+ patterns)
- Classification levels
- Automatic redaction
- Watermarking
- Access control
- Audit logging

### Documents (`documents.html`)
- Upload documents
- View all documents
- Search and filter
- Document details
- Version control

### Contracts (`contracts.html`)
- Contract listing
- Contract details
- Clause extraction
- Obligation tracking

### Reviews (`reviews.html`)
- Contract review workflow
- Risk identification
- Approval process
- Review history

### Risk Assessment (`risk.html`)
- Risk scoring
- Risk categories
- Risk mitigation
- Risk reports

### Analytics (`analytics.html`)
- Analytics dashboard
- Usage metrics
- Performance KPIs
- Custom reports

### Legal Q&A (`legal-chat.html`)
- AI-powered chat
- Document Q&A
- Context-aware responses
- Citation tracking

### Expiry Calendar (`expiry-calendar.html`)
- Contract expiration dates
- Renewal tracking
- Deadline alerts
- Calendar view

### Notifications (`notifications.html`)
- Activity notifications
- System alerts
- Email notifications
- Notification preferences

### Admin (`admin.html`)
- User management
- Organization settings
- System configuration
- Access control

---

## Icons Reference

All pages use Bootstrap Icons:

- `bi-speedometer2` - Dashboard
- `bi-files` - Documents
- `bi-file-text` - Contracts
- `bi-clipboard-check` - Reviews
- `bi-calendar-event` - Calendar
- `bi-graph-up` - Analytics
- `bi-lightning` - Predictive
- `bi-shield-exclamation` - Risk
- `bi-pen` - E-Signature
- `bi-chat-dots` - Chat
- `bi-cpu` - AI Models
- `bi-people` - Collaboration
- `bi-shield-lock` - Security
- `bi-bell` - Notifications
- `bi-gear` - Settings
- `bi-person-circle` - Profile

---

## Responsive Design

The navigation is fully responsive:

- **Desktop**: Full horizontal navigation with dropdowns
- **Tablet**: Horizontal with collapsible menu
- **Mobile**: Hamburger menu with expandable sections

---

## Customization

### Change Brand Name
Edit the navbar-brand:
```html
<a class="navbar-brand" href="../index.html">
    <i class="bi bi-file-earmark-text"></i> Your Company Name
</a>
```

### Add New Menu Item
```html
<li class="nav-item">
    <a class="nav-link" href="your-page.html">
        <i class="bi bi-your-icon"></i> Your Feature
    </a>
</li>
```

### Add to Dropdown
```html
<li><a class="dropdown-item" href="your-page.html">
    <i class="bi bi-your-icon"></i> Your Feature
</a></li>
```

---

## Dependencies

All pages require:

```html
<!-- Bootstrap CSS -->
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">

<!-- Bootstrap Icons -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.min.css">

<!-- Bootstrap JS (before closing body tag) -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>

<!-- Navigation JS (optional, for active highlighting) -->
<script src="../js/navigation.js"></script>
```

---

## Quick Start

1. **Copy navigation from `includes/nav.html`** to your page
2. **Add navigation script** before closing `</body>`:
   ```html
   <script src="../js/navigation.js"></script>
   ```
3. **Ensure Bootstrap and Icons are loaded**
4. **Test dropdowns and mobile menu**

---

## Navigation Checklist

For each page, ensure:

- [ ] Navigation is present
- [ ] All links are working
- [ ] Active page is highlighted
- [ ] Dropdowns expand/collapse properly
- [ ] Mobile menu works (hamburger icon)
- [ ] All icons are showing
- [ ] Links point to correct pages
- [ ] Bootstrap JS is loaded for dropdowns

---

## Page Paths

All pages are in `frontend/pages/` directory:

```
frontend/
├── index.html (Landing page)
├── includes/
│   └── nav.html (Navigation component)
├── js/
│   └── navigation.js (Navigation helper)
└── pages/
    ├── dashboard.html
    ├── documents.html
    ├── contracts.html
    ├── contract-details.html
    ├── document-details.html
    ├── reviews.html
    ├── risk.html
    ├── analytics.html
    ├── predictive-analytics.html ⭐ NEW
    ├── esignature.html ⭐ NEW
    ├── legal-chat.html
    ├── expiry-calendar.html
    ├── notifications.html
    ├── admin.html
    ├── model-settings.html ⭐ NEW
    ├── collaboration.html ⭐ NEW
    ├── dlp-settings.html ⭐ NEW
    └── login.html
```

---

## Support

For navigation issues:
1. Check console for errors
2. Verify Bootstrap JS is loaded
3. Check file paths are correct
4. Ensure navigation.js is loaded
5. Test in different browsers

---

**Last Updated**: January 2024
**Version**: 2.0 (Complete Navigation)
