import os
from pathlib import Path
from dotenv import load_dotenv

def load_api_key():
    """Load API key from .env file or environment variable"""
    # Print current working directory for debugging
    print(f"Current working directory: {Path.cwd()}")
    
    # Try to load from environment first
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key
    
    # Try to load from .env file
    env_path = Path(__file__).parent.parent / '.env'
    print(f"Looking for .env file at: {env_path}")
    
    if env_path.exists():
        print(".env file found")
        load_dotenv(env_path)
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            return api_key
        else:
            print("GROQ_API_KEY not found in .env file")
    else:
        print(".env file not found")
    
    raise ValueError(
        "GROQ_API_KEY not found in environment or .env file.\n"
        f"Please create a .env file at {env_path} with GROQ_API_KEY=your_key_here"
    ) 