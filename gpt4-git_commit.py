import subprocess
import sys
import openai
import os

# Set your OpenAI API key as an environment variable for security
openai.api_key = os.environ.get("OPENAI_API_KEY")

def generate_commit_message(diff):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates concise, clear git commit messages."},
            {"role": "user", "content": f"Generate a commit message for this diff:\n\n{diff}"}
        ],
        max_tokens=50
    )
    return response['choices'][0]['message']['content'].strip()

def git_add_commit_and_push():
    # Get list of modified and new files
    status_output = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    files_to_add = []
    for line in status_output.split('\n'):
        if line.strip() and not line.startswith('??') and not '__pycache__' in line:
            files_to_add.append(line.split()[-1])
        elif line.startswith('??') and not '__pycache__' in line:
            files_to_add.append(line.split()[-1])
    
    if not files_to_add:
        print("No changes to commit.")
        return

    # Add files
    for file in files_to_add:
        subprocess.run(["git", "add", file])

    # Get the diff of staged changes
    diff = subprocess.run(["git", "diff", "--cached"], capture_output=True, text=True).stdout

    # Generate commit message
    commit_message = generate_commit_message(diff)

    # Commit changes
    commit_result = subprocess.run(["git", "commit", "-m", commit_message], capture_output=True, text=True)
    
    # Print the commit output
    print(commit_result.stdout)
    if commit_result.stderr:
        print("Commit Errors:", commit_result.stderr, file=sys.stderr)

    # Push changes
    push_result = subprocess.run(["git", "push"], capture_output=True, text=True)
    
    # Print the push output
    print(push_result.stdout)
    if push_result.stderr:
        print("Push Errors:", push_result.stderr, file=sys.stderr)

if __name__ == "__main__":
    git_add_commit_and_push()
