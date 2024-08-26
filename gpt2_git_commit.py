import subprocess
import sys

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

    # Generate commit message
    if len(files_to_add) == 1:
        commit_message = f"Update {files_to_add[0]}"
    else:
        commit_message = f"Update {len(files_to_add)} files"

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