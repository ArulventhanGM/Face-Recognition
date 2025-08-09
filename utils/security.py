import re
import csv
import pandas as pd
from typing import List, Dict, Any
import html
import os

class CSVSecurityError(Exception):
    """Custom exception for CSV security violations"""
    pass

class SecureCSVHandler:
    """Secure CSV handler with protection against CSV injection and validation"""
    
    # CSV injection patterns
    DANGEROUS_PATTERNS = [
        r'^[=@+\-]',  # Formulas starting with =, @, +, -
        r'^\s*[=@+\-]',  # Formulas with leading whitespace
        r'cmd\s*\|',  # Command injection
        r'powershell',  # PowerShell commands
        r'<script',  # Script tags
        r'javascript:',  # JavaScript protocol
        r'data:',  # Data URLs
    ]
    
    @staticmethod
    def sanitize_csv_value(value: str) -> str:
        """Sanitize a single CSV value"""
        if not isinstance(value, str):
            value = str(value)
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # HTML escape
        value = html.escape(value)
        
        # Check for dangerous patterns
        for pattern in SecureCSVHandler.DANGEROUS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                # Prefix with single quote to prevent formula execution
                value = "'" + value
                break
        
        # Limit length
        if len(value) > 1000:
            value = value[:1000] + "..."
        
        return value
    
    @staticmethod
    def validate_csv_data(data: List[Dict[str, Any]], allowed_columns: List[str]) -> List[Dict[str, Any]]:
        """Validate and sanitize CSV data"""
        if len(data) > 10000:  # Limit number of rows
            raise CSVSecurityError("Too many rows in CSV data")
        
        sanitized_data = []
        for row in data:
            sanitized_row = {}
            for key, value in row.items():
                # Validate column names
                if key not in allowed_columns:
                    continue
                
                # Sanitize values
                sanitized_row[key] = SecureCSVHandler.sanitize_csv_value(str(value))
            
            sanitized_data.append(sanitized_row)
        
        return sanitized_data
    
    @staticmethod
    def safe_read_csv(file_path: str) -> pd.DataFrame:
        """Safely read CSV file with validation"""
        try:
            if not os.path.exists(file_path):
                return pd.DataFrame()
            
            # Read with size limit
            file_size = os.path.getsize(file_path)
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                raise CSVSecurityError("CSV file too large")
            
            # Handle empty files
            if file_size == 0:
                return pd.DataFrame()
            
            # Read CSV with student_id as string to prevent type conversion issues
            df = pd.read_csv(file_path, dtype={'student_id': str})
            
            # Limit rows
            if len(df) > 10000:
                raise CSVSecurityError("Too many rows in CSV file")
            
            return df
        
        except pd.errors.EmptyDataError:
            # Handle empty CSV files gracefully
            return pd.DataFrame()
        except Exception as e:
            raise CSVSecurityError(f"Error reading CSV: {str(e)}")
    
    @staticmethod
    def safe_write_csv(data: List[Dict[str, Any]], file_path: str, allowed_columns: List[str]):
        """Safely write CSV file with validation"""
        try:
            # Validate and sanitize data
            sanitized_data = SecureCSVHandler.validate_csv_data(data, allowed_columns)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write CSV with proper headers even if data is empty
            if not sanitized_data:
                # Create empty DataFrame with column headers
                df = pd.DataFrame(columns=allowed_columns)
            else:
                df = pd.DataFrame(sanitized_data)
            
            df.to_csv(file_path, index=False)
            
        except Exception as e:
            raise CSVSecurityError(f"Error writing CSV: {str(e)}")
    
    @staticmethod
    def append_to_csv(data: Dict[str, Any], file_path: str, allowed_columns: List[str]):
        """Safely append data to CSV file"""
        try:
            # Sanitize single row
            sanitized_data = SecureCSVHandler.validate_csv_data([data], allowed_columns)[0]
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Check if file exists
            file_exists = os.path.exists(file_path)
            
            # Write header if file doesn't exist
            with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=allowed_columns)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(sanitized_data)
                
        except Exception as e:
            raise CSVSecurityError(f"Error appending to CSV: {str(e)}")

def validate_student_id(student_id: str) -> bool:
    """Validate student ID format"""
    # Allow alphanumeric and some special characters
    pattern = r'^[A-Za-z0-9\-_]{3,20}$'
    return bool(re.match(pattern, student_id))

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove path traversal attempts
    filename = os.path.basename(filename)
    
    # Remove dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    
    return filename
