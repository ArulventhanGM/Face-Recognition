/**
 * Auto-refresh functionality for the Face Recognition System
 * This script automatically refreshes pages to show updated information
 */

document.addEventListener('DOMContentLoaded', function() {
    // Configuration
    const refreshablePages = [
        '/dashboard',
        '/attendance',
        '/students',
        '/recognition'
    ];
    
    // Function to check if current page should auto-refresh
    function shouldAutoRefresh() {
        const currentPath = window.location.pathname;
        return refreshablePages.some(page => currentPath.startsWith(page));
    }
    
    // Auto-refresh functionality has been removed
    
    // Empty placeholder functions for backwards compatibility
    async function getRefreshConfig() {
        return { enabled: false, interval: 0 };
    }
    
    async function setupAutoRefresh() {
        // Auto-refresh functionality has been removed
        return;
    }
    
    // No auto-refresh will be started
    
    // Store form data in session storage before page unload (for form persistence)
    window.addEventListener('beforeunload', function() {
        if (document.querySelector('form')) {
            const forms = document.querySelectorAll('form');
            const allFormData = {};
            
            forms.forEach((form, index) => {
                const formData = {};
                const formElements = form.elements;
                
                for (let i = 0; i < formElements.length; i++) {
                    const element = formElements[i];
                    if (element.name && element.type !== 'submit') {
                        if (element.type === 'checkbox' || element.type === 'radio') {
                            formData[element.name] = element.checked;
                        } else {
                            formData[element.name] = element.value;
                        }
                    }
                }
                
                allFormData[`form_${index}`] = formData;
            });
            
            sessionStorage.setItem('formData', JSON.stringify(allFormData));
        }
    });
    
    // Restore form data from session storage after page load
    if (sessionStorage.getItem('formData')) {
        try {
            const allFormData = JSON.parse(sessionStorage.getItem('formData'));
            const forms = document.querySelectorAll('form');
            
            forms.forEach((form, index) => {
                const formData = allFormData[`form_${index}`];
                if (formData) {
                    Object.keys(formData).forEach(name => {
                        const element = form.elements[name];
                        if (element) {
                            if (element.type === 'checkbox' || element.type === 'radio') {
                                element.checked = formData[name];
                            } else {
                                element.value = formData[name];
                            }
                        }
                    });
                }
            });
        } catch (e) {
            console.error('Error restoring form data', e);
        }
    }
});