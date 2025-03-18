"""
AI Financial Advisor - Main runner script

This script launches the AI Financial Advisor application.
"""

import os
import sys
import subprocess

def main():
    """Main function to run the application"""
    try:
        # Check if required packages are installed
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        
        # Run the Streamlit app
        print("🚀 Launching AI Financial Advisor...")
        os.system(f"{sys.executable} -m streamlit run app/main.py")
        
    except Exception as e:
        print(f"❌ Error starting application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 