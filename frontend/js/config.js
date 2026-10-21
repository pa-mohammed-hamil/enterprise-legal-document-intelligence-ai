/**
 * Frontend Configuration
 * 
 * Connects the frontend to the backend API running at http://localhost:8000
 */

// API Configuration
const API_CONFIG = {
    // Backend API base URL
    BASE_URL: 'http://localhost:8000',
    
    // API endpoints
    ENDPOINTS: {
        // Root
        ROOT: '/',
        HEALTH: '/health',
        
        // API v1 base
        API_V1: '/api/v1',
        
        // Authentication
        AUTH: {
            LOGIN: '/api/v1/auth/login',
            LOGOUT: '/api/v1/auth/logout',
            REFRESH: '/api/v1/auth/refresh',
            ME: '/api/v1/auth/me'
        },
        
        // Users
        USERS: {
            LIST: '/api/v1/users',
            GET: (id) => `/api/v1/users/${id}`,
            CREATE: '/api/v1/users',
            UPDATE: (id) => `/api/v1/users/${id}`,
            DELETE: (id) => `/api/v1/users/${id}`
        },
        
        // Organizations
        ORGANIZATIONS: {
            LIST: '/api/v1/organizations',
            GET: (id) => `/api/v1/organizations/${id}`,
            CREATE: '/api/v1/organizations',
            UPDATE: (id) => `/api/v1/organizations/${id}`,
            DELETE: (id) => `/api/v1/organizations/${id}`
        },
        
        // Search
        SEARCH: {
            DOCUMENTS: '/api/v1/search/documents',
            SEMANTIC: '/api/v1/search/semantic',
            HYBRID: '/api/v1/search/hybrid',
            ADVANCED: '/api/v1/search/advanced'
        },
        
        // Legal AI Chat
        CHAT: {
            MESSAGE: '/api/v1/chat/message',
            HISTORY: '/api/v1/chat/history',
            SESSION: (id) => `/api/v1/chat/session/${id}`
        },
        
        // Risk Intelligence
        RISK: {
            LIST: '/api/v1/risk',
            GET: (id) => `/api/v1/risk/${id}`,
            ANALYZE: '/api/v1/risk/analyze',
            DASHBOARD: '/api/v1/risk/dashboard'
        },
        
        // Expiry Management
        EXPIRY: {
            LIST: '/api/v1/expiry',
            GET: (id) => `/api/v1/expiry/${id}`,
            UPCOMING: '/api/v1/expiry/upcoming',
            CALENDAR: '/api/v1/expiry/calendar'
        },
        
        // Renewals
        RENEWALS: {
            LIST: '/api/v1/renewals',
            GET: (id) => `/api/v1/renewals/${id}`,
            CREATE: '/api/v1/renewals',
            UPDATE: (id) => `/api/v1/renewals/${id}`
        },
        
        // Reviews
        REVIEWS: {
            LIST: '/api/v1/reviews',
            GET: (id) => `/api/v1/reviews/${id}`,
            CREATE: '/api/v1/reviews',
            APPROVE: (id) => `/api/v1/reviews/${id}/approve`,
            REJECT: (id) => `/api/v1/reviews/${id}/reject`
        },
        
        // Notifications
        NOTIFICATIONS: {
            LIST: '/api/v1/notifications',
            GET: (id) => `/api/v1/notifications/${id}`,
            MARK_READ: (id) => `/api/v1/notifications/${id}/read`,
            MARK_ALL_READ: '/api/v1/notifications/read-all'
        },
        
        // Audit Logs
        AUDIT: {
            LIST: '/api/v1/audit',
            GET: (id) => `/api/v1/audit/${id}`,
            EXPORT: '/api/v1/audit/export'
        },
        
        // AI Models
        MODELS: {
            LIST: '/api/v1/models',
            GET: (id) => `/api/v1/models/${id}`,
            STATUS: '/api/v1/models/status'
        }
    },
    
    // Request timeout (30 seconds)
    TIMEOUT: 30000,
    
    // Retry configuration
    RETRY: {
        MAX_RETRIES: 3,
        RETRY_DELAY: 1000
    }
};

// API Helper Functions
const API = {
    /**
     * Build full URL from endpoint path
     */
    url(path) {
        return `${API_CONFIG.BASE_URL}${path}`;
    },
    
    /**
     * Get authentication token from storage
     */
    getToken() {
        return localStorage.getItem('auth_token');
    },
    
    /**
     * Set authentication token
     */
    setToken(token) {
        localStorage.setItem('auth_token', token);
    },
    
    /**
     * Remove authentication token
     */
    removeToken() {
        localStorage.removeItem('auth_token');
    },
    
    /**
     * Get default headers for API requests
     */
    getHeaders() {
        const headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        };
        
        const token = this.getToken();
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        
        return headers;
    },
    
    /**
     * Make GET request
     */
    async get(endpoint, options = {}) {
        try {
            const response = await fetch(this.url(endpoint), {
                method: 'GET',
                headers: this.getHeaders(),
                ...options
            });
            
            if (!response.ok) {
                throw new Error(`API Error: ${response.status} ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API GET Error:', error);
            throw error;
        }
    },
    
    /**
     * Make POST request
     */
    async post(endpoint, data = {}, options = {}) {
        try {
            const response = await fetch(this.url(endpoint), {
                method: 'POST',
                headers: this.getHeaders(),
                body: JSON.stringify(data),
                ...options
            });
            
            if (!response.ok) {
                throw new Error(`API Error: ${response.status} ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API POST Error:', error);
            throw error;
        }
    },
    
    /**
     * Make PUT request
     */
    async put(endpoint, data = {}, options = {}) {
        try {
            const response = await fetch(this.url(endpoint), {
                method: 'PUT',
                headers: this.getHeaders(),
                body: JSON.stringify(data),
                ...options
            });
            
            if (!response.ok) {
                throw new Error(`API Error: ${response.status} ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API PUT Error:', error);
            throw error;
        }
    },
    
    /**
     * Make DELETE request
     */
    async delete(endpoint, options = {}) {
        try {
            const response = await fetch(this.url(endpoint), {
                method: 'DELETE',
                headers: this.getHeaders(),
                ...options
            });
            
            if (!response.ok) {
                throw new Error(`API Error: ${response.status} ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API DELETE Error:', error);
            throw error;
        }
    },
    
    /**
     * Check backend health
     */
    async checkHealth() {
        try {
            const response = await this.get(API_CONFIG.ENDPOINTS.HEALTH);
            console.log('✅ Backend API is healthy:', response);
            return response;
        } catch (error) {
            console.error('❌ Backend API health check failed:', error);
            throw error;
        }
    },
    
    /**
     * Get API information
     */
    async getApiInfo() {
        try {
            const response = await this.get(API_CONFIG.ENDPOINTS.ROOT);
            console.log('📡 API Information:', response);
            return response;
        } catch (error) {
            console.error('❌ Failed to get API info:', error);
            throw error;
        }
    }
};

// Initialize API connection on page load
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 Initializing frontend connection to backend API...');
    console.log('📍 Backend URL:', API_CONFIG.BASE_URL);
    
    try {
        // Check if backend is available
        const apiInfo = await API.getApiInfo();
        const health = await API.checkHealth();
        
        console.log('✅ Successfully connected to backend!');
        console.log('   - Name:', apiInfo.name);
        console.log('   - Version:', apiInfo.version);
        console.log('   - Environment:', apiInfo.environment);
        console.log('   - Status:', health.status);
        console.log('   - API Docs:', API.url('/docs'));
        
        // Add connection status indicator to page
        addConnectionStatus(true, apiInfo);
        
    } catch (error) {
        console.error('❌ Failed to connect to backend:', error);
        console.log('⚠️  Make sure the backend server is running at:', API_CONFIG.BASE_URL);
        console.log('    Run: python run.py or start.bat');
        
        // Add connection status indicator showing error
        addConnectionStatus(false);
    }
});

/**
 * Add connection status indicator to the page
 */
function addConnectionStatus(connected, apiInfo = null) {
    const indicator = document.createElement('div');
    indicator.id = 'backend-status';
    indicator.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 12px 18px;
        border-radius: 10px;
        background: ${connected ? '#3d866f' : '#bb665c'};
        color: white;
        font-size: 12px;
        font-weight: 700;
        z-index: 9999;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        cursor: pointer;
        transition: transform 0.2s;
    `;
    
    if (connected && apiInfo) {
        indicator.innerHTML = `
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 14px;">●</span>
                <div>
                    <div>Backend Connected</div>
                    <div style="font-size: 10px; opacity: 0.9; margin-top: 2px;">
                        ${apiInfo.name} v${apiInfo.version}
                    </div>
                </div>
            </div>
        `;
        
        indicator.onclick = () => {
            window.open(API.url('/docs'), '_blank');
        };
        
        indicator.title = 'Click to open API documentation';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            indicator.style.transform = 'translateY(100px)';
            setTimeout(() => indicator.remove(), 300);
        }, 5000);
        
    } else {
        indicator.innerHTML = `
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 14px;">⚠</span>
                <div>
                    <div>Backend Disconnected</div>
                    <div style="font-size: 10px; opacity: 0.9; margin-top: 2px;">
                        Start server: python run.py
                    </div>
                </div>
            </div>
        `;
        
        indicator.onclick = () => {
            window.location.reload();
        };
        
        indicator.title = 'Click to retry connection';
    }
    
    indicator.onmouseenter = () => {
        indicator.style.transform = 'translateY(-3px)';
    };
    
    indicator.onmouseleave = () => {
        indicator.style.transform = 'translateY(0)';
    };
    
    document.body.appendChild(indicator);
}

// Export for use in other scripts
window.API_CONFIG = API_CONFIG;
window.API = API;

console.log('✨ API configuration loaded and ready!');
