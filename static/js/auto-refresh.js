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
    
    // Form persistence functionality - updated to be more robust
    window.addEventListener('beforeunload', function() {
        try {
            if (document.querySelector('form')) {
                const forms = document.querySelectorAll('form');
                const allFormData = {};
                
                forms.forEach((form, index) => {
                    if (form && form.elements) {
                        const formData = {};
                        const formElements = form.elements;
                        
                        for (let i = 0; i < formElements.length; i++) {
                            const element = formElements[i];
                            if (element && element.name && element.type !== 'submit') {
                                if (element.type === 'checkbox' || element.type === 'radio') {
                                    formData[element.name] = element.checked;
                                } else {
                                    formData[element.name] = element.value;
                                }
                            }
                        }
                        
                        allFormData[`form_${index}`] = formData;
                    }
                });
                
                // Only store if we have valid data
                if (Object.keys(allFormData).length > 0) {
                    sessionStorage.setItem('formData', JSON.stringify(allFormData));
                }
            }
        } catch (e) {
            console.error('Error saving form data:', e);
            // Clear any corrupted data
            sessionStorage.removeItem('formData');
        }
    });
    
    // Restore form data from session storage after page load
    try {
        const storedData = sessionStorage.getItem('formData');
        if (storedData) {
            const allFormData = JSON.parse(storedData);
            if (allFormData && typeof allFormData === 'object') {
                const forms = document.querySelectorAll('form');
                
                forms.forEach((form, index) => {
                    const formKey = `form_${index}`;
                    const formData = allFormData[formKey];
                    
                    if (form && formData && form.elements) {
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
            }
        }
    } catch (e) {
        console.error('Error restoring form data:', e);
        // Clear corrupted data
        sessionStorage.removeItem('formData');
    }
});