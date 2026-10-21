# Frontend-Backend Integration Guide

## ✅ Backend Connection Configured!

The frontend is now configured to connect to the backend API running at **http://localhost:8000**

## Quick Start

### 1. Start the Backend Server

Make sure the backend is running:

```powershell
# From project root
python run.py
```

Or:

```powershell
.\start.bat
```

The server should start at: **http://localhost:8000**

### 2. Open the Frontend

You have two options:

#### Option A: Direct File Access (Simple)
Just open `index.html` in your browser:
```
file:///c:/project/enterprise-legal-document-intelligence-ai/frontend/index.html
```

#### Option B: Local HTTP Server (Recommended)
Serve the frontend with a local server to avoid CORS issues:

```powershell
# Navigate to frontend directory
cd frontend

# Start Python HTTP server
python -m http.server 3000
```

Then open: **http://localhost:3000**

### 3. Verify Connection

When you open the page, you should see:
- A green connection indicator at the bottom-right corner saying "Backend Connected"
- The browser console showing: "✅ Successfully connected to backend!"

Click the connection indicator to open the API documentation.

## API Configuration

The frontend uses `js/config.js` which provides:

### Available API Helper Functions

```javascript
// Check backend health
await API.checkHealth();

// Get API information
await API.getApiInfo();

// Make authenticated GET request
await API.get('/api/v1/users');

// Make POST request
await API.post('/api/v1/auth/login', { username, password });

// Make PUT request
await API.put('/api/v1/users/123', { name: 'Updated' });

// Make DELETE request
await API.delete('/api/v1/users/123');
```

### API Endpoints Available

All endpoints are defined in `API_CONFIG.ENDPOINTS`:

```javascript
// Authentication
API_CONFIG.ENDPOINTS.AUTH.LOGIN
API_CONFIG.ENDPOINTS.AUTH.LOGOUT

// Users
API_CONFIG.ENDPOINTS.USERS.LIST
API_CONFIG.ENDPOINTS.USERS.GET(id)

// Search
API_CONFIG.ENDPOINTS.SEARCH.DOCUMENTS
API_CONFIG.ENDPOINTS.SEARCH.SEMANTIC

// Chat
API_CONFIG.ENDPOINTS.CHAT.MESSAGE
API_CONFIG.ENDPOINTS.CHAT.HISTORY

// Risk
API_CONFIG.ENDPOINTS.RISK.LIST
API_CONFIG.ENDPOINTS.RISK.ANALYZE

// And many more...
```

## Connection Status Indicator

The page automatically shows a connection status indicator:

- **Green with ● symbol**: Backend connected successfully
  - Click to open API documentation
  - Auto-hides after 5 seconds

- **Red with ⚠ symbol**: Backend not available
  - Shows "Start server: python run.py"
  - Click to retry connection

## Using the API in Your Pages

### Example 1: Fetch Documents

```javascript
// In any page script
async function loadDocuments() {
    try {
        const documents = await API.get('/api/v1/documents');
        console.log('Documents:', documents);
        
        // Update UI with documents
        displayDocuments(documents);
    } catch (error) {
        console.error('Failed to load documents:', error);
        showError('Could not load documents');
    }
}
```

### Example 2: Search Documents

```javascript
async function searchDocuments(query) {
    try {
        const results = await API.post('/api/v1/search/semantic', {
            query: query,
            limit: 10
        });
        
        return results;
    } catch (error) {
        console.error('Search failed:', error);
        return [];
    }
}
```

### Example 3: Send Chat Message

```javascript
async function sendChatMessage(message) {
    try {
        const response = await API.post('/api/v1/chat/message', {
            message: message,
            context: 'legal_analysis'
        });
        
        return response;
    } catch (error) {
        console.error('Chat failed:', error);
        throw error;
    }
}
```

### Example 4: Check Risk

```javascript
async function analyzeRisk(documentId) {
    try {
        const risk = await API.post('/api/v1/risk/analyze', {
            document_id: documentId
        });
        
        return risk;
    } catch (error) {
        console.error('Risk analysis failed:', error);
        throw error;
    }
}
```

## Authentication

### Login Flow

```javascript
async function login(username, password) {
    try {
        const response = await API.post('/api/v1/auth/login', {
            username,
            password
        });
        
        // Token is automatically stored
        API.setToken(response.access_token);
        
        return response;
    } catch (error) {
        console.error('Login failed:', error);
        throw error;
    }
}
```

### Logout

```javascript
function logout() {
    API.removeToken();
    window.location.href = 'login.html';
}
```

### Check Authentication

```javascript
async function checkAuth() {
    const token = API.getToken();
    
    if (!token) {
        // Not logged in
        window.location.href = 'login.html';
        return;
    }
    
    try {
        // Verify token is valid
        const user = await API.get('/api/v1/auth/me');
        return user;
    } catch (error) {
        // Token invalid, redirect to login
        API.removeToken();
        window.location.href = 'login.html';
    }
}
```

## Available Backend Endpoints

### ✅ Working Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation
- `POST /api/v1/auth/*` - Authentication endpoints
- `GET /api/v1/users` - User management
- `GET /api/v1/organizations` - Organization management
- `POST /api/v1/search/*` - Document search
- `POST /api/v1/chat/*` - Legal AI chat
- `GET /api/v1/risk` - Risk intelligence
- `GET /api/v1/expiry` - Expiry management
- `GET /api/v1/renewals` - Renewal tracking
- `GET /api/v1/reviews` - Review workflows
- `GET /api/v1/notifications` - Notifications
- `GET /api/v1/audit` - Audit logs
- `GET /api/v1/models` - AI model management

### ⏳ Temporarily Disabled

These will be re-enabled after fixing schema dependencies:
- `/api/v1/documents` - Document CRUD
- `/api/v1/contracts` - Contract management
- `/api/v1/clauses` - Clause extraction
- `/api/v1/analytics` - Analytics dashboards

## Troubleshooting

### Connection Indicator Shows "Disconnected"

1. **Check if backend is running:**
   ```powershell
   # Should show process running on port 8000
   netstat -ano | findstr :8000
   ```

2. **Start the backend:**
   ```powershell
   python run.py
   ```

3. **Check backend health manually:**
   ```powershell
   curl http://localhost:8000/health
   ```

### CORS Errors in Browser Console

If you see CORS errors:

1. **Use HTTP server instead of file:// protocol:**
   ```powershell
   cd frontend
   python -m http.server 3000
   ```

2. **Check backend CORS settings** in `backend/app/config.py`

### API Requests Failing with 401 Unauthorized

You need to authenticate first:

```javascript
// Login to get token
const response = await API.post('/api/v1/auth/login', {
    username: 'your_username',
    password: 'your_password'
});

// Token is automatically stored and used for subsequent requests
```

### Network Timeout Errors

The default timeout is 30 seconds. For long-running operations, you can increase it in `js/config.js`:

```javascript
API_CONFIG.TIMEOUT = 60000; // 60 seconds
```

## Development Tips

### 1. Open Browser DevTools

Always keep the browser console open (F12) to see:
- Connection status messages
- API request/response logs
- Error messages

### 2. Use API Documentation

Visit http://localhost:8000/docs to:
- See all available endpoints
- Test API calls interactively
- View request/response schemas
- Understand authentication requirements

### 3. Check Network Tab

In browser DevTools → Network tab:
- See all API requests
- Check request headers (including auth tokens)
- View response data
- Debug failed requests

### 4. Test with Swagger UI

Before integrating in frontend:
1. Go to http://localhost:8000/docs
2. Click "Authorize" button
3. Enter your credentials
4. Test endpoints interactively

## Next Steps

### 1. Add Authentication to Pages

Update each page to check authentication:

```javascript
document.addEventListener('DOMContentLoaded', async () => {
    // Check if user is authenticated
    const user = await checkAuth();
    
    // Load page data
    await loadPageData();
});
```

### 2. Implement Real Data Loading

Replace mock data with real API calls:

```javascript
// Before: Mock data
const documents = [{ id: 1, name: 'Mock' }];

// After: Real API
const documents = await API.get('/api/v1/documents');
```

### 3. Add Error Handling

Show user-friendly error messages:

```javascript
try {
    const data = await API.get('/api/v1/data');
    showSuccess('Data loaded successfully');
} catch (error) {
    showError('Failed to load data. Please try again.');
    console.error(error);
}
```

### 4. Implement Loading States

Show loading indicators during API calls:

```javascript
async function loadData() {
    showLoader();
    
    try {
        const data = await API.get('/api/v1/data');
        displayData(data);
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoader();
    }
}
```

## Resources

- **Backend API Docs**: http://localhost:8000/docs
- **Backend ReDoc**: http://localhost:8000/redoc
- **Backend Health**: http://localhost:8000/health
- **API Config**: `frontend/js/config.js`
- **Server Status**: See `SERVER_STATUS.md` in project root

## Support

For backend issues:
- Check backend logs in the terminal
- Review `QUICKSTART.md`
- Check `FIX_IMPORTS.md` for troubleshooting

For frontend issues:
- Check browser console (F12)
- Verify backend is running
- Test endpoints in Swagger UI first

---

**Last Updated:** 2026-10-10  
**Backend Version:** 1.0.0  
**Status:** ✅ Connected and Ready
