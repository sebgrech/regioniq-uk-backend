import os
from dotenv import load_dotenv
import psycopg2

# Load variables from .env file
load_dotenv()

SUPABASE_URI = os.getenv("SUPABASE_URI")

if not SUPABASE_URI:
    raise ValueError("SUPABASE_URI not set in .env or shell environment.")

try:
    conn = psycopg2.connect(SUPABASE_URI)
    cur = conn.cursor()
    cur.execute("SELECT now();")
    print("✅ Connected! Server time:", cur.fetchone()[0])
    cur.close()
    conn.close()
except Exception as e:
    print("❌ Connection failed:", e)
