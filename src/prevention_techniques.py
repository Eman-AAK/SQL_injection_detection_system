# prevention_techniques.py
# -----------------------------------------------------------
# SQL Injection Prevention Techniques Demonstration
# -----------------------------------------------------------
# Demonstrates the difference between:
#   ❌ Vulnerable string-concatenated queries
#   ✅ Safe parameterized queries (prepared statements)
# -----------------------------------------------------------

import sqlite3

def unsafe_query(user_input: str):
    """ ❌ Vulnerable: directly concatenating user input into the SQL string """
    query = f"SELECT * FROM users WHERE username = '{user_input}'"
    print("\n[VULNERABLE QUERY EXECUTED]")
    print("Generated SQL:", query)
    print("→ This query can be manipulated by an attacker!")
    print("Example effect: it can return all users instead of just one.")


def safe_query(user_input: str):
    """ ✅ Safe: uses parameterized query to prevent injection """
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("CREATE TABLE users (id INTEGER, username TEXT)")
    cur.execute("INSERT INTO users VALUES (?, ?)", (1, "admin"))
    conn.commit()

    print("\n[SAFE QUERY EXECUTION]")
    print("Using parameterized query: SELECT * FROM users WHERE username = ?")
    cur.execute("SELECT * FROM users WHERE username = ?", (user_input,))
    result = cur.fetchall()
    print("Query result:", result)
    if not result:
        print("→ Input treated as literal text — no injection occurred.")
    conn.close()


if __name__ == "__main__":
    # Example demonstration with an injection attempt
    test_input = "admin' OR '1'='1"

    print("=== SQL Injection Prevention Demonstration ===")
    print(f"User input: {test_input}")

    unsafe_query(test_input)
    safe_query(test_input)

    print("\nSummary:")
    print(" - ❌ The unsafe version runs arbitrary injected code.")
    print(" - ✅ The safe version uses placeholders to neutralize it.")
