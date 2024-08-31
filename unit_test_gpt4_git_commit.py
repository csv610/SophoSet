import unittest
from unittest.mock import patch, MagicMock, call
import io
import sys
import os

# Import the functions from your script
from your_script_name import generate_commit_message, git_add_commit_and_push

class TestGitCommitScript(unittest.TestCase):

    def setUp(self):
        # Save the original stdout and stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        # Create StringIO objects to capture output
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        sys.stdout = self.stdout
        sys.stderr = self.stderr

    def tearDown(self):
        # Restore original stdout and stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    @patch('openai.ChatCompletion.create')
    def test_generate_commit_message(self, mock_openai):
        # Test normal case
        mock_openai.return_value = {
            'choices': [{'message': {'content': 'Test commit message'}}]
        }
        diff = "Sample diff content"
        result = generate_commit_message(diff)
        self.assertEqual(result, 'Test commit message')
        mock_openai.assert_called_once()

        # Test empty diff
        mock_openai.reset_mock()
        result = generate_commit_message("")
        self.assertTrue(result)  # Ensure we get a non-empty message even for empty diff

        # Test API error
        mock_openai.side_effect = Exception("API Error")
        with self.assertRaises(Exception):
            generate_commit_message(diff)

    @patch('subprocess.run')
    @patch('your_script_name.generate_commit_message')
    def test_git_add_commit_and_push(self, mock_generate_message, mock_subprocess):
        # Test normal case
        mock_subprocess.side_effect = [
            MagicMock(stdout="M file1.py\n?? file2.py"),  # git status
            MagicMock(),  # git add file1.py
            MagicMock(),  # git add file2.py
            MagicMock(stdout="Sample diff"),  # git diff
            MagicMock(stdout="Commit successful"),  # git commit
            MagicMock(stdout="Push successful")  # git push
        ]
        mock_generate_message.return_value = "Test commit message"

        git_add_commit_and_push()

        self.assertEqual(mock_subprocess.call_count, 6)
        mock_generate_message.assert_called_once()
        self.assertIn("Commit successful", self.stdout.getvalue())
        self.assertIn("Push successful", self.stdout.getvalue())

    @patch('subprocess.run')
    def test_git_add_commit_and_push_no_changes(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(stdout="")
        git_add_commit_and_push()
        self.assertEqual(mock_subprocess.call_count, 1)
        self.assertIn("No changes to commit.", self.stdout.getvalue())

    @patch('subprocess.run')
    @patch('your_script_name.generate_commit_message')
    def test_git_add_commit_and_push_with_errors(self, mock_generate_message, mock_subprocess):
        mock_subprocess.side_effect = [
            MagicMock(stdout="M file1.py"),  # git status
            MagicMock(),  # git add
            MagicMock(stdout="Sample diff"),  # git diff
            MagicMock(stdout="", stderr="Commit failed"),  # git commit
            MagicMock(stdout="", stderr="Push failed")  # git push
        ]
        mock_generate_message.return_value = "Test commit message"

        git_add_commit_and_push()

        self.assertIn("Commit Errors: Commit failed", self.stderr.getvalue())
        self.assertIn("Push Errors: Push failed", self.stderr.getvalue())

    @patch('subprocess.run')
    @patch('your_script_name.generate_commit_message')
    def test_git_add_commit_and_push_with_pycache(self, mock_generate_message, mock_subprocess):
        mock_subprocess.side_effect = [
            MagicMock(stdout="M file1.py\n?? __pycache__/file.pyc\nM file2.py"),  # git status
            MagicMock(),  # git add file1.py
            MagicMock(),  # git add file2.py
            MagicMock(stdout="Sample diff"),  # git diff
            MagicMock(stdout="Commit successful"),  # git commit
            MagicMock(stdout="Push successful")  # git push
        ]
        mock_generate_message.return_value = "Test commit message"

        git_add_commit_and_push()

        # Check that __pycache__ files were not added
        add_calls = [call(['git', 'add', 'file1.py']), call(['git', 'add', 'file2.py'])]
        mock_subprocess.assert_has_calls(add_calls, any_order=True)

    @patch('os.environ.get')
    def test_missing_api_key(self, mock_env_get):
        mock_env_get.return_value = None
        with self.assertRaises(ValueError):
            generate_commit_message("Sample diff")

    @patch('subprocess.run')
    def test_git_command_failure(self, mock_subprocess):
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, 'git status')
        with self.assertRaises(subprocess.CalledProcessError):
            git_add_commit_and_push()

if __name__ == '__main__':
    unittest.main()
