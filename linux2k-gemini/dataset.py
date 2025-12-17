import re
import pandas as pd
import random

# 1. Define the Regex Patterns to capture "Real Data"
# These are tuned for your specific 'Linux_2k.log' file format.
patterns = {
    # Captures: rhost (IP/Domain) AND user (if present)
    'sshd_auth_fail': re.compile(r'sshd.*authentication failure;.*rhost=(\S+)(?:.*user=(\w+))?'),
    
    # Captures: just the event (no variables needed)
    'sshd_check_pass': re.compile(r'sshd.*check pass; user unknown'),
    
    # Captures: action (opened/closed) and user
    'su_session': re.compile(r'su.*session (opened|closed) for user (\w+)'),
    
    # Captures: IP address
    'ftpd_connection': re.compile(r'ftpd\[\d+\]: connection from ([\d\.]+)')
}

# 2. Define the Templates (The "Human" explanations)
# We give the AI multiple ways to say the same thing to prevent memorization.
templates = {
    'sshd_auth_fail': [
        "Security Alert: An authentication failure occurred for user '{user}' from host {ip}.",
        "Failed SSH login attempt detected from remote host {ip} (User: {user}).",
        "The system rejected a login request from {ip} for user account '{user}'.",
        "Unauthorized access attempt: {user} failed to authenticate via SSH from {ip}."
    ],
    'sshd_check_pass': [
        "The system checked a password for an unknown user.",
        "Login failed: The username provided does not exist in the system records.",
        "Authentication error: A connection attempt was made with an invalid username.",
    ],
    'su_session_opened': [
        "A new privileged session was opened for user {user}.",
        "User {user} has successfully elevated privileges (su command).",
        "System Activity: Session started for {user} via su."
    ],
    'su_session_closed': [
        "The privileged session for user {user} has been closed.",
        "User {user} dropped their elevated privileges (session ended).",
        "System Activity: Session finished for {user}."
    ],
    'ftpd_connection': [
        "Incoming FTP connection request from {ip}.",
        "File Transfer Protocol (FTP) service received a connection from {ip}.",
        "Network Event: Remote host {ip} connected to the FTP server."
    ]
}

# 3. Process the File
dataset = []

with open('Linux_2k.log', 'r') as f:
    for line in f:
        line = line.strip()
        
        # --- Pattern 1: SSH Auth Failures ---
        m = patterns['sshd_auth_fail'].search(line)
        if m:
            ip = m.group(1)
            # If 'user=' is missing in the log, we call it 'unknown'
            user = m.group(2) if m.group(2) else "unknown" 
            
            # Pick a random template
            template = random.choice(templates['sshd_auth_fail'])
            dataset.append([line, template.format(ip=ip, user=user)])
            continue

        # --- Pattern 2: SSH Invalid Users ---
        if patterns['sshd_check_pass'].search(line):
            template = random.choice(templates['sshd_check_pass'])
            dataset.append([line, template])
            continue

        # --- Pattern 3: SU (Sudo/User Switch) ---
        m = patterns['su_session'].search(line)
        if m:
            action = m.group(1) # 'opened' or 'closed'
            user = m.group(2)
            # Select template based on action
            template = random.choice(templates[f'su_session_{action}'])
            dataset.append([line, template.format(user=user)])
            continue

        # --- Pattern 4: FTP Connections ---
        m = patterns['ftpd_connection'].search(line)
        if m:
            ip = m.group(1)
            template = random.choice(templates['ftpd_connection'])
            dataset.append([line, template.format(ip=ip)])
            continue

# 4. Save to CSV
df_result = pd.DataFrame(dataset, columns=['input_text', 'target_text'])
df_result.to_csv('better_training_data.csv', index=False)

print(f"Success! Created a dataset with {len(df_result)} rows.")
print(df_result.head())