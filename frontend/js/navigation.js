/**
 * Navigation Helper
 * 
 * Loads the navigation component and highlights the active page
 */

document.addEventListener('DOMContentLoaded', function() {
    // Highlight active nav item based on current page
    const currentPage = window.location.pathname.split('/').pop();
    const navLinks = document.querySelectorAll('.nav-link, .dropdown-item');
    
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href && href.includes(currentPage)) {
            link.classList.add('active');
            
            // If it's in a dropdown, also highlight the dropdown toggle
            const dropdown = link.closest('.dropdown');
            if (dropdown) {
                const toggle = dropdown.querySelector('.dropdown-toggle');
                if (toggle) {
                    toggle.classList.add('active');
                }
            }
        }
    });
    
    // Add active state styling
    const style = document.createElement('style');
    style.textContent = `
        .nav-link.active,
        .dropdown-item.active {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }
        .dropdown-item.active {
            background-color: #0d6efd;
            color: white;
        }
    `;
    document.head.appendChild(style);
});
